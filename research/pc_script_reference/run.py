"""Run the author's unmodified SCRIPT kernel on existing native source archives.

No new source history, RT evolution, or native extension is built. The original
coeval n_ion proxy is used, not the causal experiment's integrated emissivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from research.pc_field_emulator.evaluation import shell_spectra


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def archive_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes its root: {relative}")
    return path


def verify_files(root, receipt, names=None):
    files = receipt["files"]
    for name in files if names is None else names:
        if name not in files or digest(archive_path(root, name)) != files[name]:
            raise ValueError(f"Archive checksum failed: {root / name}")


def selected_nodes(count, stride):
    if type(stride) is not int or stride < 1 or count < 2:
        raise ValueError("Positive checkpoint stride and at least two nodes required")
    return sorted(set(range(0, count, stride)) | {count - 1})


def summarize(source, capacity, xhi, native_xhi):
    atoms = capacity.sum(dtype=np.longdouble)
    photons = source.sum(dtype=np.longdouble)
    es_ions = np.sum(capacity * (1 - native_xhi.astype(float)), dtype=np.longdouble)
    return {
        "source_per_atom": float(photons / atoms),
        "photon_budget_expected_xHI_mass": float(max(0, 1 - photons / atoms)),
        "script_xHI_mass": float(np.sum(capacity * xhi, dtype=np.longdouble) / atoms),
        "script_xHI_volume": float(xhi.mean()),
        "native_es_xHI_mass": float(1 - es_ions / atoms),
        "native_es_xHI_volume": float(native_xhi.mean(dtype=float)),
        "native_es_source_minus_ions_per_atom": float((photons - es_ions) / atoms),
        "field_MAE_script_vs_native_es": float(np.mean(np.abs(xhi - native_xhi))),
    }


def _native_snapshot(history_root, record, history, node, runtime):
    from py21cmfast.drivers.coeval import Coeval
    from research.pc_native_rt.workflow import _input_dict
    from research.pc_reconsidered.native_driver import from_native_coeval

    if (record["seed"] != history["metadata"]["seed"]
            or record["redshift"] != float(history["redshifts"][node])):
        raise ValueError("Native record does not match the history seed/timeline")
    fields_path = archive_path(history_root, record["fields_path"])
    receipt = json.loads((fields_path.parent / "receipt.json").read_text())
    verify_files(fields_path.parent, receipt, ("source.npz", "native.json", "native.h5"))
    if receipt["record"] != record:
        raise ValueError("Native snapshot record differs from its history")
    if receipt["identity"]["extension_sha256"] != runtime["native_extension_sha256"]:
        raise ValueError("Use the input run's original Python/native extension for brightness")
    if receipt["identity"]["inputs"] != history["metadata"]["inputs"]:
        raise ValueError("Native snapshot inputs differ from its history")
    with np.load(fields_path, allow_pickle=False) as saved:
        fields = {name: np.array(saved[name]) for name in saved.files}
    for name, hist_name in (("capacity", "native_capacity"),
                            ("native_es_xhi", "native_xhi"),
                            ("native_brightness", "native_brightness")):
        if not np.array_equal(fields[name], history[hist_name][node]):
            raise ValueError(f"Native snapshot/history disagree: {name}")
    native = Coeval.from_file(archive_path(history_root, record["h5_path"]))
    if _input_dict(native.inputs) != receipt["identity"]["inputs"]:
        raise ValueError("Native H5 inputs differ from its receipt")
    source = from_native_coeval(native)  # Enforces supported physics, no density multiplier.
    for field, value in (("source", source.source), ("capacity", source.capacity),
                         ("native_es_xhi", native.ionized_box.get("neutral_fraction")),
                         ("native_brightness", native.brightness_temp)):
        if not np.array_equal(fields[field], value):
            raise ValueError(f"Native H5/NPZ disagree: {field}")
    if float(native.redshift) != record["redshift"]:
        raise ValueError("Native H5 redshift differs from record")
    return native, fields, receipt


def plot_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for seed in report["seeds"]:
        rows = seed["snapshots"]
        z = [r["redshift"] for r in rows]
        line, = axes[0].plot(z, [r["script_xHI_mass"] for r in rows], label=f"SCRIPT seed {seed['seed']}")
        color = line.get_color()
        axes[0].plot(z, [r["native_es_xHI_mass"] for r in rows], "--", color=color,
                     label=f"native ES seed {seed['seed']}")
        axes[1].plot(z, [r["source_per_atom"] for r in rows], color=color,
                     label=f"native coeval n_ion, {seed['seed']}")
        axes[1].plot(z, [r["context_causal_integrated_photons_per_atom"] for r in rows],
                     ":", color=color, label=f"causal time integral, {seed['seed']}")
        axes[2].plot(z, [r["field_MAE_script_vs_native_es"] for r in rows],
                     color=color, label=f"seed {seed['seed']}")
    for ax, label in zip(axes, ("Mass-weighted neutral fraction", "Effective photons / atom",
                               "Neutral-field MAE: SCRIPT vs native ES")):
        ax.set(xlabel="Redshift", ylabel=label)
        ax.invert_xaxis()
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    axes[1].axhline(1, color="gray", linewidth=.7)
    fig.suptitle("Original SCRIPT on native coeval sources; differences are model comparisons")
    fig.tight_layout()
    fig.savefig(output / "script_comparison.png", dpi=180)
    plt.close(fig)


def _run_unlocked(input_run, output):
    from research.pc_causal_rt.native_history import load_history
    from research.pc_native_rt.workflow import _runtime_identity
    from research.pc_reconsidered.native_driver import _native_box
    from py21cmfast.drivers.single_field import brightness_temperature
    from research.pc_script_reference import kernel

    input_run, output = Path(input_run).resolve(), Path(output).resolve()
    root = input_run / "results" if (input_run / "results").is_dir() else input_run
    if output.is_relative_to(input_run) or input_run.is_relative_to(output):
        raise ValueError("Reference output must be separate from the readonly input run")
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["configuration"]
    runtime = _runtime_identity()
    if runtime["native_extension_sha256"] != manifest["native_runtime"]["native_extension_sha256"]:
        raise ValueError("Native binary differs from archived run; use its python_path.txt")
    histories = sorted((root / "histories").glob("seed*/history_receipt.json"))
    if not histories:
        raise ValueError("No completed native source histories found")
    output.mkdir(parents=True, exist_ok=True)
    provenance = kernel.provenance()
    identity = {"input_root": str(root), "input_manifest_sha256": digest(root / "manifest.json"),
                "native_runtime": runtime, "script_provenance": provenance,
                "runner_sha256": digest(__file__), "kernel_sha256": digest(kernel.__file__)}
    old_identity = output / "identity.json"
    if old_identity.exists() and json.loads(old_identity.read_text()) != identity:
        raise ValueError("Reference inputs or implementation changed; use a fresh output directory")
    (output / "COMPLETED").unlink(missing_ok=True)
    write_json(old_identity, identity)
    report = {"schema": 1, "solver": "author original SCRIPT Fortran kernel",
              "physical_model_authority": "21cmFAST; SCRIPT is an external comparison only, not a production physics replacement",
              "source_convention": "native coeval cumulative n_ion; no extra density factor or rate integral",
              "scope": "restricted native ACG/no recombination/no thermal/no photoheating-feedback coevals",
              "geometry": "SCRIPT full periodic redistribution; native ES retains its own filter and R_BUBBLE_MAX",
              "comparison_is_truth_error": False, "convergence_certified": False,
              "native_early_epoch_caveat": "Native low-emissivity early exit can include RECFAST residual ions outside the stellar n_ion budget; source-minus-ions is not solely photon loss at those epochs",
              "brightness": "original archived native binary brightness_temperature, real-space, saturated spin temperature",
              "context_rate_integral": "trapezoidal original source-stride-1 rates, shown only as context; not SCRIPT input",
              "identity": identity, "seeds": []}
    started = time.perf_counter()
    for receipt_path in histories:
        history_root = receipt_path.parent
        receipt = json.loads(receipt_path.read_text())
        verify_files(history_root, receipt, ("history.npz", "history_manifest.json"))
        if json.loads((history_root / "history_manifest.json").read_text()) != receipt["identity"]:
            raise ValueError("History manifest differs from its receipt identity")
        history = load_history(history_root / "history.npz")
        seed = history["metadata"]["seed"]
        if seed not in config["seeds"] or len(history["records"]) != len(history["redshifts"]):
            raise ValueError("Native history does not match experiment configuration")
        rates = history["rates"].sum(axis=(1, 2, 3), dtype=np.longdouble)
        emitted = np.r_[np.longdouble(0), np.cumsum(.5 * (rates[1:] + rates[:-1])
                                                   * np.diff(history["times_seconds"]))]
        seed_report = {"seed": seed, "history_sha256": receipt["files"]["history.npz"], "snapshots": []}
        report["seeds"].append(seed_report)
        for node in selected_nodes(len(history["redshifts"]), config["checkpoint_stride"]):
            record = history["records"][node]
            native_receipt_path = Path(record["fields_path"]).with_name("receipt.json")
            verify_files(history_root, receipt, (str(native_receipt_path),))
            native, fields, native_receipt = _native_snapshot(history_root, record, history, node, runtime)
            node_identity = {"history_sha256": seed_report["history_sha256"],
                             "native_record": record, "native_files": native_receipt["files"],
                             "reference_identity": identity}
            folder = output / f"seed{seed}" / f"node{node:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            result_receipt = folder / "receipt.json"
            if result_receipt.is_file():
                saved = json.loads(result_receipt.read_text())
                if saved["identity"] != node_identity:
                    raise ValueError("Saved SCRIPT reference identity changed")
                verify_files(folder, saved)
                row = json.loads((folder / "metrics.json").read_text())
                if row["ledger"].get("passed") is not True:
                    raise ValueError("Saved SCRIPT ledger did not pass")
            else:
                qi, ledger = kernel.run(fields["source"], fields["capacity"], build_dir=output / "_build")
                if not np.isfinite(qi).all() or (qi < 0).any() or (qi > 1).any():
                    raise ValueError("Original SCRIPT returned invalid ionized fractions; no clipping applied")
                if ledger.get("passed") is not True:
                    raise ValueError(f"Original SCRIPT photon ledger failed: {ledger}")
                xhi = 1 - qi
                ionized = _native_box(native, fields["capacity"] * qi, fields["capacity"])
                brightness = brightness_temperature(ionized_box=ionized,
                    perturbed_field=native.perturbed_field, regenerate=True, write=False)
                tb = np.array(brightness.get("brightness_temp"))
                spectra = shell_spectra({"teacher": tb, "native_es": fields["native_brightness"]},
                                        history["metadata"]["box_len_cMpc"])
                spectra = {key.replace("teacher", "script"): value for key, value in spectra.items()}
                spectra["ratio_definition"] = "P_native_es / P_script; null for undefined reference power"
                row = {"node": node, "redshift": record["redshift"], "ledger": ledger,
                       **summarize(fields["source"], fields["capacity"], xhi, fields["native_es_xhi"]),
                       "context_causal_integrated_photons_per_atom": float(emitted[node] / fields["capacity"].sum()),
                       "spectra": spectra}
                with (folder / "fields.npz.tmp").open("wb") as stream:
                    np.savez_compressed(stream, **fields, script_xhi=xhi, script_brightness=tb)
                (folder / "fields.npz.tmp").replace(folder / "fields.npz")
                write_json(folder / "metrics.json", row)
                write_json(result_receipt, {"identity": node_identity,
                    "files": {name: digest(folder / name) for name in ("fields.npz", "metrics.json")}})
            seed_report["snapshots"].append(row)
            print(f"seed={seed} z={row['redshift']:g} SCRIPT xHI_mass={row['script_xHI_mass']:.6g} "
                  f"ES={row['native_es_xHI_mass']:.6g}", flush=True)
    report["all_script_ledgers_passed"] = all(r["ledger"]["passed"] for s in report["seeds"] for r in s["snapshots"])
    report["elapsed_seconds_including_audit_and_brightness"] = time.perf_counter() - started
    plot_report(output, report)
    write_json(output / "script_report.json", report)
    write_json(output / "completion.json", {"identity": identity, "files": {
        name: digest(output / name) for name in ("script_report.json", "script_comparison.png")}})
    (output / "COMPLETED").touch()
    return report


def run(input_run, output):
    import fcntl
    input_run, output = Path(input_run).resolve(), Path(output).resolve()
    if output.is_relative_to(input_run) or input_run.is_relative_to(output):
        raise ValueError("Reference output must be separate from the readonly input run")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reference.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_unlocked(input_run, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.input_run, args.output)


if __name__ == "__main__":
    main()
