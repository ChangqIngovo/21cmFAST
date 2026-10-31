"""Audit continuous RT fields and test successive numerical refinements.

Convergence compares RT to RT at identical cosmic times. Native-cumulative RT
shares the native coeval stellar photon budget, including its initial inventory.
The explicitly named legacy instantaneous-rate mode has a different budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.pc_field_emulator.evaluation import shell_spectra

AXES = ("directions", "refinement", "time_substeps", "source_stride", "cfl")
DECREASING = {"source_stride", "cfl"}
LEDGER_TOLERANCE = 2e-10


def _sum(value):
    return np.asarray(value).sum(dtype=np.longdouble)


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name]) for name in data.files}


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _primary_band(spectra, dimension, box_len):
    k = np.asarray(spectra["k_cMpc_inv"])
    minimum = 2 * np.pi / box_len
    maximum = min(4 * minimum, .5 * np.pi * dimension / box_len)
    selected = (k >= minimum * (1 - 1e-12)) & (k <= maximum * (1 + 1e-12))
    actual = k[selected]
    modes = np.asarray(spectra["modes"])[selected]
    return {"minimum_bound_cMpc_inv": float(minimum), "maximum_bound_cMpc_inv": float(maximum),
            "bin_count": len(actual), "shell_centres_cMpc_inv": actual.tolist(),
            "modes_per_bin": modes.tolist(),
            "minimum_tested_shell_cMpc_inv": float(actual.min()) if len(actual) else None,
            "maximum_tested_shell_cMpc_inv": float(actual.max()) if len(actual) else None}


def _setting_key(setting):
    return json.dumps(setting, sort_keys=True, allow_nan=False)


def _audit_case(root, case, box_len):
    if case.get("source_mode", "instantaneous") not in ("instantaneous", "native_cumulative"):
        raise ValueError("Unknown RT source convention")
    frames, rows = {}, []
    first_atoms = None
    previous_time = -np.inf
    previous_source = previous_ions = np.longdouble(0)
    for point in case["checkpoints"]:
        key = int(point["node"])
        time = float(point["time_seconds"])
        if key in frames or not np.isfinite(time) or time <= previous_time:
            raise ValueError("Checkpoint nodes and times must be unique and ordered")
        previous_time = time
        path = Path(root) / point["fields_path"]
        if point.get("fields_sha256") is not None and _digest(path) != point["fields_sha256"]:
            raise ValueError("Saved checkpoint file hash differs from its receipt")
        arrays = _load(path)
        capacity = arrays["capacity"]
        if capacity.ndim != 3 or len(set(capacity.shape)) != 1 or len(capacity) < 2:
            raise ValueError("Saved capacity must be a cubic grid")
        for name in ("capacity", "ionized", "radiation", "cumulative_source",
                     "rt_xhi", "rt_brightness", "native_xhi", "native_brightness"):
            value = arrays[name]
            if value.shape != capacity.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid saved {name}")
            if name in ("capacity", "ionized", "radiation", "cumulative_source") and (value < 0).any():
                raise ValueError(f"Negative saved {name}")
        if case.get("source_mode", "instantaneous") == "native_cumulative":
            for name in ("source_native_current", "source_sampled_current"):
                if (name not in arrays or arrays[name].shape != capacity.shape
                        or not np.isfinite(arrays[name]).all() or (arrays[name] < 0).any()):
                    raise ValueError(f"Invalid saved {name}")
            if (str(arrays.get("source_mode", np.asarray("")).item()) != "native_cumulative"
                    or str(arrays.get("source_budget_sha256", np.asarray("")).item()) != case.get("source_budget_sha256")):
                raise ValueError("Saved checkpoint source identity differs from its case")
        matched = {"matched_es_xhi", "matched_es_brightness"}
        if matched & set(arrays):
            if not matched <= set(arrays):
                raise ValueError("Matched ES must include both neutral fraction and brightness")
            for name in matched:
                value = arrays[name]
                if value.shape != capacity.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid saved {name}")
            if np.any(arrays["matched_es_xhi"] < 0) or np.any(arrays["matched_es_xhi"] > 1):
                raise ValueError("Matched ES neutral fraction lies outside [0,1]")
        ions, radiation, source = (arrays[k] for k in ("ionized", "radiation", "cumulative_source"))
        if np.any(ions - capacity > 2e-11 * np.maximum(capacity, 1e-300)):
            raise ValueError("Saved ion counts exceed atom capacity")
        xhi = np.divide(capacity - ions, capacity, out=np.ones_like(capacity), where=capacity > 0)
        if (np.max(np.abs(xhi - arrays["rt_xhi"])) > 2e-7
                or np.any(arrays["native_xhi"] < 0) or np.any(arrays["native_xhi"] > 1)):
            raise ValueError("Saved neutral fractions are inconsistent or out of bounds")
        atoms, emitted, absorbed, remaining = map(_sum, (capacity, source, ions, radiation))
        if atoms <= 0:
            raise ValueError("Positive atom inventory required")
        if first_atoms is None:
            first_atoms = atoms
        atom_error = abs(atoms - first_atoms) / first_atoms
        photon_error = abs(emitted - absorbed - remaining) / max(emitted, np.longdouble(np.finfo(float).tiny))
        if atom_error > LEDGER_TOLERANCE or photon_error > LEDGER_TOLERANCE:
            raise ValueError("Independent saved-field atom/photon ledger failed")
        if (emitted + LEDGER_TOLERANCE * atoms < previous_source
                or absorbed + LEDGER_TOLERANCE * atoms < previous_ions):
            raise ValueError("Source or ion inventory decreased in no-recombination evolution")
        previous_source, previous_ions = emitted, absorbed
        spectral_fields = {"teacher": arrays["rt_brightness"], "native_es": arrays["native_brightness"]}
        if "matched_es_brightness" in arrays:
            spectral_fields["matched_es"] = arrays["matched_es_brightness"]
        spectra = shell_spectra(spectral_fields, box_len)
        frame = dict(point, xHI_mass=float(1 - absorbed / atoms), xHI_volume=float(xhi.mean()),
                     native_xHI_volume=float(arrays["native_xhi"].mean(dtype=float)),
                     source_per_atom=float(emitted / atoms), radiation_per_atom=float(remaining / atoms),
                     photon_ledger_relative_residual=float(photon_error),
                     atom_ledger_relative_residual=float(atom_error), spectra=spectra,
                     checkpoint_file_hash_verified=point.get("fields_sha256") is not None,
                     primary_power_band=_primary_band(spectra, len(capacity), box_len))
        if "matched_es_xhi" in arrays:
            frame.update(matched_es_xHI_volume=float(arrays["matched_es_xhi"].mean()),
                         matched_es_xHI_mass=float(_sum(capacity * arrays["matched_es_xhi"]) / atoms))
        rows.append(frame)
        frames[key] = (frame, arrays)
    if not rows:
        raise ValueError("A case needs saved checkpoints")
    return {"case_id": case["case_id"], "seed": case["seed"], "setting": case["setting"],
            "source_mode": case.get("source_mode", "instantaneous"),
            "source_budget_sha256": case.get("source_budget_sha256"),
            "history_sha256": case["history_sha256"], "all_saved_field_ledgers_passed": True,
            "all_interval_ledgers_passed": case.get("all_ledgers_passed") is True,
            "checkpoints": rows}, frames


def audit_case(root, case, box_len):
    """Independently recompute saved counts, bounds and physical spectra."""
    return _audit_case(root, case, box_len)[0]


def _pair(coarse, fine, frame_map, axis, box_len, tolerance):
    if coarse["history_sha256"] != fine["history_sha256"]:
        raise ValueError("A convergence pair must use the same archived native history")
    native_budget = coarse.get("source_mode", "instantaneous") == "native_cumulative"
    if coarse.get("source_mode", "instantaneous") != fine.get("source_mode", "instantaneous"):
        raise ValueError("A convergence pair must use the same source convention")
    if native_budget and coarse.get("source_budget_sha256") != fine.get("source_budget_sha256"):
        raise ValueError("A convergence pair must use the same native cumulative source archive")
    left, right = frame_map[coarse["case_id"]], frame_map[fine["case_id"]]
    if set(left) != set(right):
        raise ValueError("Convergence cases have different checkpoint nodes")
    values = []
    worst_field = worst_power = worst_source = 0.
    undefined = eligible_bins = active = 0
    for node in sorted(left):
        a, aa = left[node]
        b, bb = right[node]
        if a["time_seconds"] != b["time_seconds"] or a["redshift"] != b["redshift"]:
            raise ValueError("Convergence cases must share actual cosmic timestamps")
        if not np.allclose(aa["capacity"], bb["capacity"], rtol=2e-11, atol=2e-12):
            raise ValueError("Numerical refinements changed the underlying coarse gas field")
        if not np.array_equal(aa["native_xhi"], bb["native_xhi"]):
            raise ValueError("Convergence pair has different native comparison fields")
        if axis != "source_stride":
            same_source = (np.isclose(_sum(aa["cumulative_source"]), _sum(bb["cumulative_source"]),
                                      rtol=2e-11, atol=0.) if native_budget else
                           np.allclose(aa["cumulative_source"], bb["cumulative_source"], rtol=2e-11, atol=2e-12))
            if not same_source:
                raise ValueError("Numerical refinements changed the supplied source budget")
        if not (.05 <= a["xHI_mass"] <= .9 and .05 <= b["xHI_mass"] <= .9):
            continue
        active += 1
        k = np.asarray(b["spectra"]["k_cMpc_inv"])
        if not np.array_equal(k, np.asarray(a["spectra"]["k_cMpc_inv"])):
            raise ValueError("Convergence spectra have different wave numbers")
        maximum = min(4 * 2 * np.pi / box_len, .5 * np.pi * len(bb["capacity"]) / box_len)
        primary = (k >= 2 * np.pi / box_len * (1 - 1e-12)) & (k <= maximum * (1 + 1e-12))
        pa = np.asarray(a["spectra"]["P_teacher"])
        pb = np.asarray(b["spectra"]["P_teacher"])
        floor = max(float(np.max(pb)) * 1e-12, np.finfo(float).eps**2 * box_len**3)
        valid = primary & (pa > floor) & (pb > floor)
        undefined_here = int(np.count_nonzero(primary & ~valid))
        undefined += undefined_here
        eligible_bins += int(np.count_nonzero(primary))
        errors = np.abs(pa[valid] / pb[valid] - 1)
        power = float(np.max(errors)) if errors.size else None
        field = float(np.mean(np.abs(aa["rt_xhi"] - bb["rt_xhi"])))
        source_error = float(np.sum(np.abs(aa["cumulative_source"].astype(np.longdouble)
                                             - bb["cumulative_source"]))
                             / max(_sum(bb["cumulative_source"]), np.longdouble(np.finfo(float).tiny)))
        worst_field = max(worst_field, field)
        worst_power = max(worst_power, power or 0.)
        worst_source = max(worst_source, source_error)
        values.append({"node": node, "redshift": b["redshift"], "field_MAE": field,
                       "maximum_primary_power_error": power, "undefined_primary_bins": undefined_here,
                       "primary_bin_count": int(np.count_nonzero(primary)),
                       "source_L1_relative_error": source_error})
    passed = bool(active and eligible_bins and undefined == 0
                  and worst_field <= tolerance and worst_power <= tolerance
                  and (native_budget or worst_source <= tolerance))
    return {"coarse": coarse["case_id"], "fine": fine["case_id"], "axis": axis,
            "active_checkpoints": active, "primary_power_bin_checks": eligible_bins,
            "undefined_active_power_bins": undefined, "worst_field_MAE": worst_field,
            "worst_primary_power_error": worst_power if eligible_bins and not undefined else None,
            "worst_source_L1_relative_error": worst_source,
            "source_map_difference_role": ("Diagnostic: moving-emitter quadrature changes historical emission locations; each map is independently audited and global budgets must agree"
                                           if native_budget else "Convergence criterion on the integrated endpoint-rate map"),
            "primary_power_band": next(iter(right.values()))[0]["primary_power_band"],
            "passed": passed, "checkpoints": values}


def convergence(cases, frame_map, box_len, tolerance):
    """Require two successive refinements along every axis at one finest anchor."""
    output = []
    for seed in sorted({c["seed"] for c in cases}):
        selected = [c for c in cases if c["seed"] == seed]
        keys = [_setting_key(c["setting"]) for c in selected]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate settings for one seed")
        anchor = {axis: (min if axis in DECREASING else max)(c["setting"][axis] for c in selected)
                  for axis in AXES}
        by_axis = {}
        for axis in AXES:
            ladder = [c for c in selected if all(c["setting"][other] == anchor[other]
                                                for other in AXES if other != axis)]
            ladder.sort(key=lambda c: c["setting"][axis], reverse=axis in DECREASING)
            pairs = [_pair(a, b, frame_map, axis, box_len, tolerance)
                     for a, b in zip(ladder, ladder[1:])]
            by_axis[axis] = {"adjacent_comparisons": len(pairs),
                             "two_adjacent_steps_available": len(pairs) >= 2,
                             "passed": bool(len(pairs) >= 2 and all(p["passed"] for p in pairs)),
                             "comparisons": pairs}
        output.append({"seed": seed, "finest_anchor": anchor, "axes": by_axis,
                       "numerical_convergence_certified": all(a["passed"] for a in by_axis.values())})
    return output


def _plot(root, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = sorted({case["seed"] for case in report["cases"]})
    figure, axes = plt.subplots(len(seeds), 3, figsize=(14, 3.6 * len(seeds)), squeeze=False)
    for row, seed in zip(axes, seeds, strict=True):
        cases = [case for case in report["cases"] if case["seed"] == seed]
        anchor = next(r["finest_anchor"] for r in report["seeds"] if r["seed"] == seed)
        reference = next((c for c in cases if c["setting"] == anchor), cases[0])
        native_budget = reference.get("source_mode", "instantaneous") == "native_cumulative"
        control_label = ("ES on current sampled source map" if native_budget else
                         "ES on integrated emission-location map")
        for case in cases:
            points = case["checkpoints"]
            is_reference = case is reference
            color = "C0" if is_reference else ".75"
            label = "Finest RT" if is_reference else None
            row[0].plot([p["redshift"] for p in points], [p["xHI_volume"] for p in points],
                        color=color, alpha=1 if is_reference else .5, label=label)
            row[1].semilogy([p["redshift"] for p in points],
                            [max(p["photon_ledger_relative_residual"], 1e-19) for p in points], color=color)
        points = reference["checkpoints"]
        if all("matched_es_xHI_volume" in p for p in points):
            row[0].plot([p["redshift"] for p in points], [p["matched_es_xHI_volume"] for p in points],
                        "--", color="C2", label=control_label)
        row[0].plot([p["redshift"] for p in points], [p["native_xHI_volume"] for p in points],
                    "k--", label="Native coeval ES (cumulative proxy)")
        stage = min(points, key=lambda p: abs(p["xHI_volume"] - .5))
        spectrum = stage["spectra"]
        row[2].plot(spectrum["k_cMpc_inv"], spectrum["Delta2_teacher"], "o-", label="RT")
        row[2].plot(spectrum["k_cMpc_inv"], spectrum["Delta2_native_es"], "s--", color="black", label="Native coeval ES")
        if "Delta2_matched_es" in spectrum:
            row[2].plot(spectrum["k_cMpc_inv"], spectrum["Delta2_matched_es"], "^--", color="C2",
                        label=control_label)
        row[2].set_title(f"Seed {seed}, same z={stage['redshift']:g}; different mean HI")
        row[2].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel=r"$\Delta^2_{21}$ [mK$^2$]")
        row[0].set_ylabel("Volume-mean neutral fraction")
        row[1].set_ylabel("Independent photon ledger residual")
        row[0].set_ylim(-.02, 1.02)
        row[0].legend(fontsize=7)
        row[2].legend(fontsize=8)
        for panel in row[:2]:
            panel.set_xlabel("Redshift")
            panel.invert_xaxis()
            panel.set_title(f"Seed {seed}")
        for panel in row:
            panel.grid(alpha=.2)
    figure.suptitle("Continuous RT and native coeval ES: source convention recorded in report")
    figure.tight_layout()
    figure.savefig(root / "causal_rt_comparison.png", dpi=160)
    plt.close(figure)



def _native_budget_history(history, budget, setting):
    """Reconstruct emission locations directly from conserved packet increments.

    This does not use workflow state, its source-rate helper, or its fine-grid
    source arrays. The coarse CIC integral is independent of RT refinement.
    """
    from .gas import deposit
    from .native_history import background_at_time

    times = np.asarray(history["times_seconds"], dtype=float)
    q = np.asarray(budget["cumulative"], dtype=float)
    stride = setting["source_stride"]
    substeps = setting["time_substeps"]
    if (q.shape[0] != len(times) or not np.array_equal(budget["times_seconds"], times)
            or not np.array_equal(budget["redshifts"], history["redshifts"])
            or not np.array_equal(budget["growth_factors"], history["growth_factors"])
            or type(stride) is not int or stride < 1 or (len(times) - 1) % stride
            or type(substeps) is not int or substeps < 1):
        raise ValueError("Native cumulative source archive and history coordinates differ")
    sampled = q.copy()
    for left in range(0, len(times) - 1, stride):
        right = left + stride
        for index in range(left + 1, right):
            w = (times[index] - times[left]) / (times[right] - times[left])
            sampled[index] = (1 - w) * q[left] + w * q[right]
    if (np.diff(sampled, axis=0) < 0).any():
        raise ValueError("A cumulative source packet decreases; no negative-photon clipping is allowed")
    packets = budget["packets"]
    n = packets["coarse_dimension"]
    initial_growth = packets["growth_initial"]

    def positions(growth):
        return (packets["q"] + (growth - initial_growth) * packets["v1"]
                + (3. / 7.) * (growth**2 - initial_growth**2) * packets["v2"])

    # Count the initial cumulative stellar budget exactly once, including when
    # it is initially present as radiation rather than ionized atoms.
    emitted = deposit(sampled[0].ravel(), positions(history["growth_factors"][0]), n)
    cumulative = [emitted.astype(np.longdouble)]
    current_native, current_sampled = [], []
    for node in range(len(times)):
        endpoint = positions(history["growth_factors"][node])
        current_native.append(deposit(q[node].ravel(), endpoint, n))
        current_sampled.append(deposit(sampled[node].ravel(), endpoint, n))
        if node == 0:
            continue
        increment = (sampled[node] - sampled[node - 1]).ravel() / substeps
        step = np.zeros((n,) * 3, dtype=np.longdouble)
        for sub in range(substeps):
            growth = background_at_time(history, node, (sub + .5) / substeps)["growth_factor"]
            step += deposit(increment, positions(growth), n)
        cumulative.append(cumulative[-1] + step)
    return {"cumulative": np.asarray(cumulative), "sampled": sampled,
            "current_native": np.asarray(current_native),
            "current_sampled": np.asarray(current_sampled)}


def _history_sources(root, summary, cases, frames):
    """Reconstruct native packet budgets or explicitly legacy rate integrals.

    A missing archive prevents certification. A supplied but corrupt archive or
    source mismatch is an error, not an uncertified-success condition.
    """
    histories = {}
    for row in summary.get("histories", []):
        seed = row["seed"]
        if seed in histories:
            raise ValueError("Duplicate native history seed")
        path = root / row["path"]
        digest = _digest(path)
        if digest != row["sha256"]:
            raise ValueError("Archived native history hash differs from its receipt")
        history = _load(path)
        times, zs, rates, mass = (history[k] for k in
                                 ("times_seconds", "redshifts", "rates", "packet_mass"))
        if (times.ndim != 1 or len(times) < 2 or times[0] != 0
                or zs.shape != times.shape or rates.shape[0] != len(times)
                or not np.isfinite(times).all() or not (np.diff(times) > 0).all()
                or not np.isfinite(zs).all() or not (np.diff(zs) < 0).all()
                or not np.isfinite(rates).all() or (rates < 0).any()
                or mass.ndim != 1 or not np.isfinite(mass).all() or (mass < 0).any()
                or _sum(mass) <= 0):
            raise ValueError("Invalid archived native source timeline or atom inventory")
        histories[seed] = (history, digest, path, row)
    cache = {}
    for case in cases:
        if case["seed"] not in histories:
            case["native_history_audit"] = {"passed": False, "reason": "Native rate/atom archive absent"}
            continue
        history, digest, history_path, history_row = histories[case["seed"]]
        if digest != case["history_sha256"]:
            raise ValueError("Case does not use the same archived native history")
        stride = case["setting"]["source_stride"]
        native_budget = case.get("source_mode", "instantaneous") == "native_cumulative"
        if native_budget and case.get("source_budget_sha256") != history_row.get("source_budget_sha256"):
            raise ValueError("Native cumulative source archive differs from its case")
        key = (case["seed"], stride, case["setting"]["time_substeps"] if native_budget else None,
               case.get("source_mode", "instantaneous"))
        times = history["times_seconds"]
        if type(stride) is not int or stride < 1 or (len(times) - 1) % stride:
            raise ValueError("Source stride does not divide the original native history")
        if native_budget and key not in cache:
            from .native_history import load_history
            from .source_budget import load_source_budget
            if "source_budget_path" not in history_row or "source_budget_sha256" not in history_row:
                raise ValueError("Native cumulative run is missing its source-budget archive")
            budget_path = root / history_row["source_budget_path"]
            if (_digest(budget_path) != history_row["source_budget_sha256"]
                    or case.get("source_budget_sha256") != history_row["source_budget_sha256"]):
                raise ValueError("Native cumulative source archive differs from its receipt or case")
            budget = load_source_budget(budget_path)
            if budget["metadata"]["history_sha256"] != digest:
                raise ValueError("Source-budget archive references a different native history")
            full_history = load_history(history_path)
            reconstructed = _native_budget_history(full_history, budget, case["setting"])
            cache[key] = {**reconstructed, "budget": budget, "history": full_history}
        elif not native_budget and key not in cache:
            knots = np.arange(0, len(times), stride)
            upper = np.searchsorted(knots, np.arange(len(times)), side="right")
            upper = np.minimum(upper, len(knots) - 1)
            lower = np.maximum(upper - 1, 0)
            left, right = knots[lower], knots[upper]
            weight = ((times - times[left]) / (times[right] - times[left]))[:, None, None, None]
            rates = ((1 - weight) * history["rates"][left] + weight * history["rates"][right]).astype(np.longdouble)
            increments = (rates[:-1] + rates[1:]) * (np.diff(times).astype(np.longdouble)[:, None, None, None] / 2)
            cache[key] = np.concatenate((np.zeros_like(rates[:1]), np.cumsum(increments, axis=0)), axis=0)
        cumulative = cache[key]["cumulative"] if native_budget else cache[key]
        total_atoms = _sum(history["packet_mass"])
        maximum_source = maximum_atoms = maximum_density = 0.
        maximum_budget = maximum_current = maximum_native_roundoff = maximum_sampling = 0.
        maximum_native_map_roundoff = 0.
        for node, (point, arrays) in frames[case["case_id"]].items():
            if (not 0 <= node < len(times) or point["time_seconds"] != float(times[node])
                    or point["redshift"] != float(history["redshifts"][node])):
                raise ValueError("Checkpoint does not match actual cosmic timestamps in native history")
            expected = cumulative[node]
            if expected.shape != arrays["cumulative_source"].shape:
                raise ValueError("Native source grid differs from the saved coarse grid")
            source_error = float(np.sum(np.abs(expected - arrays["cumulative_source"]))
                                 / max(_sum(expected), np.longdouble(np.finfo(float).tiny)))
            atom_error = float(abs(_sum(arrays["capacity"]) - total_atoms) / total_atoms)
            maximum_source = max(maximum_source, source_error)
            maximum_atoms = max(maximum_atoms, atom_error)
            if source_error > 2e-11 or atom_error > 2e-11:
                raise ValueError("Saved source/atom inventory differs from independently integrated native history")
            if native_budget:
                reconstruction = cache[key]
                target = _sum(reconstruction["sampled"][node])
                budget_error = float(abs(_sum(arrays["cumulative_source"]) - target)
                                     / max(target, np.longdouble(np.finfo(float).tiny)))
                current_error = 0.
                for field, expected_current in (("source_native_current", reconstruction["current_native"][node]),
                                                ("source_sampled_current", reconstruction["current_sampled"][node])):
                    if field not in arrays or arrays[field].shape != expected_current.shape:
                        raise ValueError("Saved current-location source field is missing or inconsistent")
                    error = float(_sum(np.abs(arrays[field] - expected_current))
                                  / max(_sum(expected_current), np.longdouble(np.finfo(float).tiny)))
                    current_error = max(current_error, error)
                if budget_error > 2e-11 or current_error > 2e-11:
                    raise ValueError("Saved source differs from native packet cumulative budget or current source map")
                budget = reconstruction["budget"]
                native_total = np.longdouble(budget["native_totals"][node])
                native_record = reconstruction["history"]["records"][node]
                native_root = Path(budget["metadata"]["native_archive_root"]).resolve()
                native_path = (native_root / native_record["fields_path"]).resolve()
                if not native_path.is_relative_to(native_root):
                    raise ValueError("Archived native source path escapes its root")
                with np.load(native_path, allow_pickle=False) as original:
                    native_source = np.array(original["source"])
                native_hash = hashlib.sha256(np.ascontiguousarray(native_source).tobytes()).hexdigest()
                if native_hash != native_record["source_sha256"]:
                    raise ValueError("Original native source array hash changed")
                if native_source.shape != reconstruction["current_native"][node].shape:
                    raise ValueError("Native source grid differs from current packet deposition")
                native_map_roundoff = float(_sum(np.abs(native_source - reconstruction["current_native"][node]))
                                           / max(_sum(native_source), np.longdouble(np.finfo(float).tiny)))
                if (native_map_roundoff > 1e-6 or
                        not np.isclose(native_total, _sum(native_source), rtol=2e-15, atol=0.)):
                    raise ValueError("Native cumulative source deposition or total differs from original source beyond float roundoff")
                native_roundoff = float(abs(_sum(budget["cumulative"][node]) - native_total)
                                        / max(native_total, np.longdouble(np.finfo(float).tiny)))
                if native_roundoff > 1e-6:
                    raise ValueError("Native cumulative total differs from original native source beyond float roundoff")
                sampling_difference = float(abs(target - _sum(budget["cumulative"][node]))
                                            / max(_sum(budget["cumulative"][node]), np.longdouble(np.finfo(float).tiny)))
                maximum_budget = max(maximum_budget, budget_error)
                maximum_current = max(maximum_current, current_error)
                maximum_native_roundoff = max(maximum_native_roundoff, native_roundoff)
                maximum_native_map_roundoff = max(maximum_native_map_roundoff, native_map_roundoff)
                maximum_sampling = max(maximum_sampling, sampling_difference)
                point.update(source_budget_target_total=float(target),
                             source_budget_relative_residual=budget_error,
                             source_native_total=float(native_total),
                             source_native_total_relative_roundoff=native_roundoff,
                             source_native_map_L1_relative_roundoff=native_map_roundoff,
                             source_sampling_relative_difference=sampling_difference)
            if "native_capacity" in history:
                density_error = float(np.max(np.abs(arrays["capacity"] - history["native_capacity"][node])))
                maximum_density = max(maximum_density, density_error)
                if density_error > 3e-6:
                    raise ValueError("Coarse RT density differs from original native density beyond float roundoff")
        case["native_history_audit"] = {"passed": True, "history_sha256": digest,
                                        "maximum_source_L1_relative_error": maximum_source,
                                        "maximum_atom_inventory_relative_error": maximum_atoms,
                                        "maximum_native_density_absolute_error": maximum_density}
        if native_budget:
            case["native_history_audit"].update(
                source_mode="native_cumulative", source_budget_sha256=case["source_budget_sha256"],
                initial_source_total=float(_sum(cache[key]["sampled"][0])),
                maximum_native_cumulative_budget_relative_residual=maximum_budget,
                maximum_current_source_L1_relative_error=maximum_current,
                maximum_native_source_total_relative_roundoff=maximum_native_roundoff,
                maximum_native_source_L1_relative_roundoff=maximum_native_map_roundoff,
                maximum_source_sampling_relative_difference=maximum_sampling,
                cumulative_map_definition="Initial photon inventory plus historical emission locations; not the current native source map")


def build_report(root, *, make_plot=True):
    """Audit actual saved fields and publish narrowly scoped numerical tests."""
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    config = summary["configuration"]
    tolerance = float(config.get("tolerance", .01))
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Convergence tolerance must lie in (0,1)")
    box_len = float(config["box_len"])
    cases, frames = [], {}
    for item in summary["cases"]:
        if item["case_id"] in frames:
            raise ValueError("Duplicate case identity")
        report, data = _audit_case(root, item, box_len)
        cases.append(report)
        frames[item["case_id"]] = data
    if not cases:
        raise ValueError("Cannot certify an empty run")
    _history_sources(root, summary, cases, frames)
    seeds = convergence(cases, frames, box_len, tolerance)
    native_budget = all(c.get("source_mode", "instantaneous") == "native_cumulative" for c in cases)
    report = {"schema": 1, "scope": "Numerical convergence of the recorded continuous effective-atom, no-recombination RT equations; not full H/He/thermal physics validation",
              "source_modes": sorted({c.get("source_mode", "instantaneous") for c in cases}),
              "native_es_comparison": ("Shared original native cumulative stellar budget, allowing native float32 deposition roundoff and declared source-knot interpolation; finite-c propagation and spatial allocation differ"
                                       if native_budget else "Context only: legacy native coeval cumulative n_ion differs from the integrated instantaneous RT source history"),
              "matched_es_comparison": ("Static ES on the current-location sampled native cumulative source map and current gas; shares its total budget, but is not a causal ES history"
                                        if native_budget else "Static ES on the integrated emission-location map and current gas; shares accumulated emission budget, but is not a causal ES history"),
              "all_native_history_audits_passed": all(c["native_history_audit"]["passed"] for c in cases),
              "actual_primary_power_bands": {c["case_id"]: c["checkpoints"][0]["primary_power_band"] for c in cases},
              "tolerance": tolerance,
              "primary_power_band": "Shell centres from fundamental to min(4*fundamental, half native-grid Nyquist)",
              "active_window": "Both mass-weighted neutral fractions in [0.05,0.9]",
              "all_saved_field_ledgers_passed": True,
              "all_interval_ledgers_passed": all(c["all_interval_ledgers_passed"] for c in cases),
              "maximum_saved_photon_ledger_relative_residual": max(p["photon_ledger_relative_residual"] for c in cases for p in c["checkpoints"]),
              "maximum_saved_atom_ledger_relative_residual": max(p["atom_ledger_relative_residual"] for c in cases for p in c["checkpoints"]),
              "maximum_native_cumulative_budget_relative_residual": (max(c["native_history_audit"].get("maximum_native_cumulative_budget_relative_residual", 0.) for c in cases)
                                                                      if native_budget else None),
              "convergence_certified": bool(all(s["numerical_convergence_certified"] for s in seeds)
                                            and all(c["all_interval_ledgers_passed"] for c in cases)
                                            and all(c["native_history_audit"]["passed"] for c in cases)),
              "seeds": seeds, "cases": cases}
    destination = root / "causal_rt_report.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    if make_plot:
        _plot(root, report)
    return report
