"""Streaming comparison of independent coeval PC, original SCRIPT, and native ES.

The two allocators receive identical archived extensive cumulative photon and
atom arrays. This is an instantaneous coeval closure, not finite-speed RT.
No causal history NPZ is opened. Each completed snapshot has its own receipt.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from research.pc_field_emulator.evaluation import field_metrics, shell_spectra
from research.pc_script_reference.run import archive_path, digest, verify_files, write_json


DEFAULT_REDSHIFTS = "10,9,8,7.5,7.25,7,6.75,6.5,6.375,6"


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def discover(input_run, redshifts):
    """Read small receipts only, selecting exact archived redshifts per seed."""
    input_run = Path(input_run).resolve()
    root = input_run / "results" if (input_run / "results").is_dir() else input_run
    manifest = json.loads((root / "manifest.json").read_text())
    requested = sorted(set(float(z) for z in redshifts), reverse=True)
    if not requested or not np.isfinite(requested).all() or min(requested) < 0:
        raise ValueError("Require finite nonnegative requested redshifts")
    snapshots = []
    for path in sorted((root / "histories").glob("seed*/history_receipt.json")):
        history_root = path.parent
        history_receipt = json.loads(path.read_text())
        verify_files(history_root, history_receipt, ("history_manifest.json",))
        if json.loads((history_root / "history_manifest.json").read_text()) != history_receipt["identity"]:
            raise ValueError("History manifest differs from its receipt")
        selected = {}
        for name in history_receipt["files"]:
            if not name.startswith("native/") or not name.endswith("/receipt.json"):
                continue
            # Small snapshot receipts provide z; the large arrays remain unopened.
            receipt_path = archive_path(history_root, name)
            receipt = json.loads(receipt_path.read_text())
            z = float(receipt["record"]["redshift"])
            if z not in requested:
                continue
            verify_files(history_root, history_receipt, (name,))
            if z in selected:
                raise ValueError(f"Duplicate snapshot at z={z}: {history_root}")
            record = receipt["record"]
            if history_root.name != f"seed{record['seed']}":
                raise ValueError("Snapshot seed differs from its history folder")
            selected[z] = {"history_root": str(history_root), "receipt_path": str(receipt_path),
                           "history_receipt_sha256": digest(path),
                           "receipt_sha256": digest(receipt_path), "record": record}
        missing = sorted(set(requested) - set(selected), reverse=True)
        if missing:
            raise ValueError(f"Requested redshifts absent in {history_root}: {missing}; select existing snapshots")
        snapshots.extend(selected[z] for z in requested)
    if not snapshots:
        raise ValueError("No completed native source snapshot receipts found")
    return root, manifest, snapshots


def load_snapshot(snapshot, runtime):
    """Validate one NPZ against its native H5 and the original binary."""
    from py21cmfast.drivers.coeval import Coeval
    from research.pc_native_rt.workflow import _input_dict
    from research.pc_reconsidered.native_driver import from_native_coeval

    receipt_path = Path(snapshot["receipt_path"])
    if digest(receipt_path) != snapshot["receipt_sha256"]:
        raise ValueError("Input snapshot receipt changed during run")
    receipt = json.loads(receipt_path.read_text())
    verify_files(receipt_path.parent, receipt, ("source.npz", "native.json", "native.h5"))
    if receipt["record"] != snapshot["record"]:
        raise ValueError("Native record changed")
    if receipt["identity"]["extension_sha256"] != runtime["native_extension_sha256"]:
        raise ValueError("Use the input archive's original native extension")
    root = Path(snapshot["history_root"])
    record = snapshot["record"]
    fields_path = archive_path(root, record["fields_path"])
    h5_path = archive_path(root, record["h5_path"])
    if fields_path != receipt_path.parent / "source.npz" or h5_path != receipt_path.parent / "native.h5":
        raise ValueError("Record fields do not identify the verified files")
    with np.load(fields_path, allow_pickle=False) as saved:
        fields = {name: np.array(saved[name]) for name in
                  ("source", "capacity", "native_es_xhi", "native_brightness")}
    native = Coeval.from_file(h5_path)
    if _input_dict(native.inputs) != receipt["identity"]["inputs"]:
        raise ValueError("Native H5 inputs differ from receipt")
    if float(native.redshift) != record["redshift"]:
        raise ValueError("Native redshift differs from receipt")
    extracted = from_native_coeval(native)  # Also enforces no Park correction / restricted physics.
    for name, actual in (("source", extracted.source), ("capacity", extracted.capacity),
                         ("native_es_xhi", native.ionized_box.get("neutral_fraction")),
                         ("native_brightness", native.brightness_temp)):
        if not np.array_equal(fields[name], actual):
            raise ValueError(f"Native H5 and NPZ disagree: {name}")
    if (fields["capacity"] <= 0).any():
        raise ValueError("SCRIPT requires strictly positive capacity in every cell")
    return native, fields


def native_brightness(native, ionized, capacity):
    from research.pc_reconsidered.native_driver import _native_box
    from py21cmfast.drivers.single_field import brightness_temperature
    box = _native_box(native, ionized, capacity)
    result = brightness_temperature(ionized_box=box, perturbed_field=native.perturbed_field,
                                    regenerate=True, write=False)
    return np.array(result.get("brightness_temp"))


def compare_fields(source, capacity, pc_xhi, script_xhi, es_xhi, temperatures, box_len):
    """Budget, morphology and native brightness spectra at one common redshift."""
    atoms = np.sum(capacity, dtype=np.longdouble)
    photons = np.sum(source, dtype=np.longdouble)
    expected = float(max(0, 1 - photons / atoms))
    mass = {name: float(np.sum(capacity * value, dtype=np.longdouble) / atoms)
            for name, value in (("pc", pc_xhi), ("script", script_xhi), ("native_es", es_xhi))}
    spectra = shell_spectra({"teacher": temperatures["script"], "pc": temperatures["pc"],
                             "native_es": temperatures["native_es"]}, box_len)
    spectra = {key.replace("teacher", "script"): value for key, value in spectra.items()}
    spectra["ratio_reference"] = "SCRIPT; null where reference power is numerically zero"
    spectra["below_half_nyquist"] = [k <= .5 * spectra["k_max_cMpc_inv"] for k in spectra["k_cMpc_inv"]]
    spectra["scale_caveat"] = "Half-Nyquist is a diagnostic band, not a certified accuracy cut; grid convergence remains required"
    return {"source_per_atom": float(photons / atoms), "expected_xHI_mass": expected,
            "xHI_mass": mass,
            "xHI_volume": {name: float(np.mean(value)) for name, value in
                           (("pc", pc_xhi), ("script", script_xhi), ("native_es", es_xhi))},
            "mass_budget_absolute_difference": {name: abs(value - expected) for name, value in mass.items()},
            "pc_vs_script": field_metrics(1-pc_xhi, 1-script_xhi, capacity),
            "native_es_vs_script": field_metrics(1-es_xhi, 1-script_xhi, capacity),
            "spectra_21cm": spectra}


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.cpp"))
    for relative in ("research/pc_script_reference/kernel.py", "research/pc_script_reference/run.py",
                     "research/pc_reconsidered/native_driver.py", "research/pc_native_rt/workflow.py",
                     "research/pc_field_emulator/evaluation.py"):
        paths.append(root / relative)
    return {str(path.relative_to(root)): digest(path) for path in sorted(set(paths))}


def freeze_identity(path, identity):
    """Preserve original bytes on resume, including equivalent JSON numbers."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != identity:
            raise ValueError("Inputs, settings, or implementation changed; use a fresh output directory")
        # 120 and 120.0 compare equal but have different serialized hashes.
        # Snapshot receipts refer to the originally written byte identity.
        return saved
    write_json(path, identity)
    return identity


def plot_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    seeds = sorted({row["seed"] for row in report["snapshots"]})
    for seed in seeds:
        rows = [row for row in report["snapshots"] if row["seed"] == seed]
        z = [row["redshift"] for row in rows]
        for method, style in (("pc", "-"), ("script", "--"), ("native_es", ":")):
            axes[0].plot(z, [row["xHI_mass"][method] for row in rows], style, label=f"{method} seed {seed}")
        axes[1].plot(z, [row["pc_vs_script"]["MAE"] for row in rows], "o-", label=f"PC vs SCRIPT, {seed}")
        # Pick an active midpoint, not a fully ionized zero-power spectrum.
        active = [row for row in rows if .05 < row["xHI_mass"]["script"] < .95]
        if active:
            row = min(active, key=lambda item: abs(item["xHI_mass"]["script"] - .5))
            s = row["spectra_21cm"]
            axes[2].plot(s["k_cMpc_inv"], s["ratio_pc"], label=f"seed {seed}, z={row['redshift']:g}")
    axes[0].set(xlabel="Redshift", ylabel="Mass-weighted neutral fraction")
    axes[1].set(xlabel="Redshift", ylabel="Neutral-field MAE: PC vs SCRIPT")
    axes[2].set(xlabel=r"$k$ [cMpc$^{-1}$]", ylabel="21-cm P_PC / P_SCRIPT")
    axes[2].axhline(1, color="gray", lw=.7)
    for ax in axes[:2]:
        ax.invert_xaxis()
    for ax in axes:
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    fig.suptitle("Same native coeval photon and atom arrays; instantaneous PC vs original SCRIPT")
    fig.tight_layout()
    temporary = Path(output) / "coeval_comparison.tmp.png"
    fig.savefig(temporary, dpi=170)
    plt.close(fig)
    temporary.replace(Path(output) / "coeval_comparison.png")


def _run(input_run, output, *, threads, redshifts, tolerance, max_iterations,
         max_interactions, script_timeout):
    from research.pc_native_rt.workflow import _runtime_identity
    from research.pc_script_reference import kernel
    from research.pc_coeval.solver import build_identity, solve

    root, manifest, snapshots = discover(input_run, redshifts)
    runtime = _runtime_identity()
    if runtime["native_extension_sha256"] != manifest["native_runtime"]["native_extension_sha256"]:
        raise ValueError("Native binary differs from input archive; use its original runtime")
    settings = dict(threads=threads, tolerance=tolerance, max_iterations=max_iterations,
                    max_interactions=max_interactions, script_timeout=script_timeout)
    _, script_build = kernel.build(output / "_build" / "script")
    identity = {"input_root": str(root), "input_manifest_sha256": digest(root / "manifest.json"),
                "runtime": runtime, "source_hashes": implementation_hashes(),
                "pc_build": build_identity(), "script_build": script_build,
                "script_provenance": kernel.provenance(), "settings": settings, "snapshots": snapshots}
    identity_path = output / "identity.json"
    identity = freeze_identity(identity_path, identity)
    (output / "COMPLETED").unlink(missing_ok=True)
    report = {"schema": 1, "model": "independent coeval synchronous radial photon allocation",
              "source_convention": "raw native cumulative n_ion, identical source/capacity bytes for PC and SCRIPT",
              "propagation": "instantaneous frozen-gas periodic redistribution; NOT finite-speed causal RT",
              "native_physics_scope": "L-INTEGRAL ACG only; no recombinations, MCGs, thermal evolution or photoheating feedback",
              "native_es_photon_correction": "no-photoncons (Park correction disabled)",
              "brightness": "original native brightness_temperature with archived native inputs",
              "comparison_is_truth_error": False, "convergence_certified": False,
              "history_caveat": "Identical mass-mean histories follow from a shared completed budget, not morphological accuracy; volume means can differ",
              "early_epoch_caveat": "Native ES can retain RECFAST residual ions beyond the stellar source budget",
              "timing_scope": "PC in-process allocation and audit; SCRIPT executable including process startup and binary IO; compiler and common native loading/brightness excluded. Small-grid times are not a pure-kernel speedup measurement",
              "precision": "PC float64; unmodified SCRIPT REAL32 with both original-input and executed-input ledgers",
              "identity": identity, "snapshots": []}
    identity_hash = digest(identity_path)
    for snapshot in snapshots:
        record = snapshot["record"]
        z, seed = record["redshift"], record["seed"]
        folder = output / f"seed{seed}" / ("z" + f"{z:g}".replace(".", "p"))
        folder.mkdir(parents=True, exist_ok=True)
        receipt_path = folder / "receipt.json"
        node_identity = {"run_identity_sha256": identity_hash, "input_receipt_sha256": snapshot["receipt_sha256"]}
        if receipt_path.is_file():
            input_receipt_path = Path(snapshot["receipt_path"])
            if digest(input_receipt_path) != snapshot["receipt_sha256"]:
                raise ValueError("Input snapshot receipt changed during resume")
            verify_files(input_receipt_path.parent, json.loads(input_receipt_path.read_text()),
                         ("source.npz", "native.json", "native.h5"))
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != node_identity:
                raise ValueError("Saved snapshot identity changed")
            verify_files(folder, receipt)
            row = json.loads((folder / "metrics.json").read_text())
            if not row["pc_audit"]["complete"] or not row["script_ledger"]["passed"] or not row["budget_checks_passed"]:
                raise ValueError("Cannot resume an uncertified snapshot")
            print(f"reuse seed={seed} z={z:g}", flush=True)
        else:
            started = time.perf_counter()
            native, fields = load_snapshot(snapshot, runtime)
            if float(native.inputs.simulation_options.BOX_LEN) != float(manifest["configuration"]["box_len"]):
                raise ValueError("Native box length differs from spectrum configuration")
            source, capacity = fields["source"], fields["capacity"]
            source.flags.writeable = capacity.flags.writeable = False
            common = {"source_sha256_float64": array_digest(source.astype(np.float64, copy=False)),
                      "capacity_sha256_float64": array_digest(capacity.astype(np.float64, copy=False)),
                      "shape": list(source.shape)}
            print(f"start seed={seed} z={z:g} shape={source.shape} "
                  f"photons_per_atom={source.sum()/capacity.sum():.6g}", flush=True)
            pc = solve(source, capacity, threads=threads, tolerance=tolerance,
                       max_iterations=max_iterations, max_interactions=max_interactions)
            if not pc["audit"]["complete"]:
                write_json(folder / "incomplete_pc.json", pc["audit"])
                raise RuntimeError(f"PC allocation incomplete at seed={seed} z={z}; see {folder / 'incomplete_pc.json'}")
            qi_script, ledger = kernel.run(source, capacity, build_dir=output / "_build" / "script",
                                           timeout_seconds=script_timeout)
            if not ledger["passed"] or not np.isfinite(qi_script).all() or (qi_script < 0).any() or (qi_script > 1).any():
                raise RuntimeError("Original SCRIPT failed its independent photon/bounds audit")
            qi_pc = pc["ionized"] / capacity
            if not np.isfinite(qi_pc).all() or (qi_pc < 0).any() or (qi_pc > 1).any():
                raise RuntimeError("PC returned invalid ionization; no clipping applied")
            temperatures = {"pc": native_brightness(native, pc["ionized"], capacity),
                            "script": native_brightness(native, capacity * qi_script, capacity),
                            "native_es": fields["native_brightness"]}
            row = compare_fields(source, capacity, 1-qi_pc, 1-qi_script, fields["native_es_xhi"],
                                 temperatures, manifest["configuration"]["box_len"])
            row.update(seed=seed, redshift=z, common_input=common,
                       pc_audit=pc["audit"], script_ledger=ledger,
                       total_snapshot_seconds=time.perf_counter()-started,
                       process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
            row["budget_checks_passed"] = (row["mass_budget_absolute_difference"]["pc"] <= max(tolerance*2, 1e-12)
                                           and row["mass_budget_absolute_difference"]["script"] <= ledger["tolerance"]*2)
            if not row["budget_checks_passed"]:
                write_json(folder / "failed_budget.json", row)
                raise RuntimeError("Mass-mean history failed analytic common-budget check")
            temporary = folder / "fields.tmp.npz"
            np.savez(temporary, source=source, capacity=capacity, pc_xhi=1-qi_pc, script_xhi=1-qi_script,
                     native_es_xhi=fields["native_es_xhi"], **{name+"_brightness": tb for name, tb in temperatures.items()})
            temporary.replace(folder / "fields.npz")
            write_json(folder / "metrics.json", row)
            write_json(receipt_path, {"identity": node_identity,
                                     "files": {name: digest(folder / name) for name in ("fields.npz", "metrics.json")}})
            print(f"seed={seed} z={z:g} PC={row['xHI_mass']['pc']:.6f} SCRIPT={row['xHI_mass']['script']:.6f} "
                  f"MAE={row['pc_vs_script']['MAE']:.5g} PC_s={pc['audit']['elapsed_seconds']:.3g} "
                  f"SCRIPT_s={ledger['elapsed_seconds']:.3g}", flush=True)
            del native, fields, source, capacity, pc, qi_pc, qi_script, temperatures
            gc.collect()
        report["snapshots"].append(row)
        write_json(output / "coeval_report.json", report)
    plot_report(output, report)
    write_json(output / "report_receipt.json", {"identity_sha256": identity_hash,
               "files": {name: digest(output / name) for name in ("coeval_report.json", "coeval_comparison.png")}})
    (output / "COMPLETED").write_text(identity_hash + "\n")
    return report


def run(input_run, output, *, threads=8, redshifts=(10, 9, 8, 7.5, 7.25, 7, 6.75, 6.5, 6.375, 6),
        tolerance=1e-8, max_iterations=1000, max_interactions=1_000_000_000_000, script_timeout=900):
    input_run, output = Path(input_run).resolve(), Path(output).resolve()
    if output.is_relative_to(input_run) or input_run.is_relative_to(output):
        raise ValueError("Output must be separate from the readonly input archive")
    if not np.isfinite(script_timeout) or script_timeout <= 0:
        raise ValueError("SCRIPT timeout must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(input_run, output, threads=threads, redshifts=redshifts, tolerance=tolerance,
                    max_iterations=max_iterations, max_interactions=max_interactions, script_timeout=script_timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--redshifts", default=DEFAULT_REDSHIFTS)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-interactions", type=int, default=1_000_000_000_000)
    parser.add_argument("--script-timeout", type=float, default=900)
    args = parser.parse_args()
    run(args.input_run, args.output, threads=args.threads, redshifts=[float(z) for z in args.redshifts.split(",")],
        tolerance=args.tolerance, max_iterations=args.max_iterations, max_interactions=args.max_interactions,
        script_timeout=args.script_timeout)


if __name__ == "__main__":
    main()
