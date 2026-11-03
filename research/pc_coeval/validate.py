"""Read-only independent photon audit and geometric MFP postprocessing.

This never calls either allocation solver or imports the native extension.
MFP means a uniform phase-volume point's isotropic first-exit distance on a
periodic voxel mask, not an absorption cross-section or bubble radius.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

METHODS = ("pc", "script", "native_es")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_files(root, receipt):
    root = Path(root).resolve()
    if not receipt.get("files"):
        raise ValueError("Empty receipt")
    for name, expected in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"Archive checksum failed: {path}")


def read_archive(input_run):
    """Only small metadata; array checks happen one snapshot at a time."""
    root = Path(input_run).resolve()
    if (root / "results" / "coeval_report.json").is_file():
        root = root / "results"
    identity_path = root / "identity.json"
    identity_hash = digest(identity_path)
    if (root / "COMPLETED").read_text().strip() != identity_hash:
        raise ValueError("Input coeval run is not completed with this identity")
    receipt = json.loads((root / "report_receipt.json").read_text())
    if receipt["identity_sha256"] != identity_hash or "coeval_report.json" not in receipt["files"]:
        raise ValueError("Report receipt identity mismatch")
    verify_files(root, receipt)
    report = json.loads((root / "coeval_report.json").read_text())
    identity = json.loads(identity_path.read_text())
    if report["identity"] != identity:
        raise ValueError("Report differs from input identity")
    manifest_path = Path(identity["input_root"]) / "manifest.json"
    if digest(manifest_path) != identity["input_manifest_sha256"]:
        raise ValueError("Original native manifest checksum failed")
    manifest = json.loads(manifest_path.read_text())
    box_len = float(manifest["configuration"]["box_len"])
    if not np.isfinite(box_len) or box_len <= 0:
        raise ValueError("Invalid physical box length")
    expected_nodes = {(s["record"]["seed"], s["record"]["redshift"]): s for s in identity["snapshots"]}
    if len(expected_nodes) != len(identity["snapshots"]):
        raise ValueError("Duplicate identity snapshots")
    snapshots, seen = [], set()
    for row in report["snapshots"]:
        key = row["seed"], row["redshift"]
        if key in seen or key not in expected_nodes:
            raise ValueError("Duplicate or unexpected report snapshot")
        seen.add(key)
        folder = root / f"seed{key[0]}" / ("z" + f"{key[1]:g}".replace(".", "p"))
        node_receipt_path = folder / "receipt.json"
        node_receipt = json.loads(node_receipt_path.read_text())
        expected_identity = {"run_identity_sha256": identity_hash,
                             "input_receipt_sha256": expected_nodes[key]["receipt_sha256"]}
        if node_receipt["identity"] != expected_identity:
            raise ValueError("Snapshot identity mismatch")
        if not {"fields.npz", "metrics.json"}.issubset(node_receipt["files"]):
            raise ValueError("Snapshot receipt lacks fields or metrics")
        if digest(folder / "metrics.json") != node_receipt["files"]["metrics.json"]:
            raise ValueError("Snapshot metrics checksum failed")
        if json.loads((folder / "metrics.json").read_text()) != row:
            raise ValueError("Snapshot metrics differ from report")
        shape = row["common_input"]["shape"]
        if len(shape) != 3 or len(set(shape)) != 1:
            raise ValueError("Expected cubic coeval fields")
        spectra = row["spectra_21cm"]
        if not np.isclose(spectra["shell_width_cMpc_inv"], 2*np.pi/box_len, rtol=1e-10):
            raise ValueError("Spectrum and manifest physical box lengths disagree")
        snapshots.append({"folder": str(folder), "receipt_sha256": digest(node_receipt_path),
                          "receipt": node_receipt, "record": expected_nodes[key]["record"], "row": row})
    if not snapshots or seen != set(expected_nodes):
        raise ValueError("Report is missing expected snapshots")
    return root, report, snapshots, box_len


def load_fields(snapshot):
    folder = Path(snapshot["folder"])
    if digest(folder / "receipt.json") != snapshot["receipt_sha256"]:
        raise ValueError("Input snapshot receipt changed")
    verify_files(folder, snapshot["receipt"])
    with np.load(folder / "fields.npz", allow_pickle=False) as saved:
        fields = {name: saved[name] for name in ("source", "capacity", *(m + "_xhi" for m in METHODS))}
        if "pc_remaining_source" in saved:
            fields["pc_remaining_source"] = saved["pc_remaining_source"]
    common = snapshot["row"]["common_input"]
    shape = tuple(common["shape"])
    if any(value.shape != shape for value in fields.values()):
        raise ValueError("Field shape differs from metadata")
    for name in ("source", "capacity"):
        actual = array_digest(fields[name])
        if actual != common[name + "_sha256_float64"] or actual != snapshot["record"][name + "_sha256"]:
            raise ValueError(f"Independent input-array checksum failed: {name}")
    return fields


def implementation_hashes():
    folder = Path(__file__).parent
    return {name: digest(folder / name) for name in
            ("validate.py", "photon_audit.py", "mfp.py", "mfp_kernel.cpp")}


def plot_report(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"pc": "-", "script": "--", "native_es": ":"}
    colors = {"pc": "C0", "script": "C1", "native_es": "C2"}
    threshold = min(report["identity"]["settings"]["thresholds"], key=lambda t: abs(t-.5))
    seeds = sorted({row["seed"] for row in report["snapshots"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for seed in seeds:
        rows = [r for r in report["snapshots"] if r["seed"] == seed]
        z = [r["redshift"] for r in rows]
        for method in METHODS:
            data = [r["photon_audit"]["methods"][method] for r in rows]
            axes[0].plot(z, [d["absolute_defect_per_atom"] for d in data], styles[method],
                         label=f"{method} seed {seed}", color=colors[method])
        axes[1].plot(z, [r["source_per_atom"] for r in rows], "o-", label=f"source/atoms, {seed}")
    axes[0].set_yscale("symlog", linthresh=1e-14)
    axes[0].set(ylabel="|target ionized atoms - field ionized atoms| / atoms")
    axes[1].set(ylabel="Cumulative photons per atom")
    axes[1].axhline(1, color="gray", lw=.7)
    for ax in axes:
        ax.set(xlabel="Redshift"); ax.invert_xaxis(); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Independent field-based budget check; surplus inferred unless stored separately")
    fig.tight_layout(); fig.savefig(output / "photon_budget.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for seed in seeds:
        rows = [r for r in report["snapshots"] if r["seed"] == seed]
        for col, phase in enumerate(("ionized", "neutral")):
            for method in METHODS:
                vals = [next(v for v in r["mfp"] if v["threshold"] == threshold and v["phase"] == phase)["methods"][method]
                        for r in rows]
                z = [r["redshift"] for r in rows]
                axes[0,col].plot(z, [v.get("restricted_mean_cMpc") for v in vals], styles[method],
                                 color=colors[method], label=f"{method}, {seed}")
                axes[1,col].plot(z, [v.get("censored_fraction") for v in vals], styles[method], color=colors[method])
    for col, phase in enumerate(("ionized", "neutral")):
        axes[0,col].set(title=f"{phase} phase; xHI threshold {threshold:g}", ylabel="Restricted mean exit length [cMpc]")
        axes[1,col].set(ylabel="Fraction reaching distance cap", ylim=(-.02,1.02))
        for ax in axes[:,col]:
            ax.set(xlabel="Redshift"); ax.invert_xaxis(); ax.grid(alpha=.2)
        axes[0,col].legend(fontsize=8)
    fig.suptitle("Geometric MFP; mean includes capped rays, empty phase is undefined")
    fig.tight_layout(); fig.savefig(output / "mfp_history.png", dpi=160); plt.close(fig)
    selected = []
    for seed in seeds:
        rows = [r for r in report["snapshots"] if r["seed"] == seed and 0 < r["xHI_mass"]["pc"] < 1]
        for target in (.8, .5, .05):
            if rows:
                row = min(rows, key=lambda r: abs(r["xHI_mass"]["pc"]-target))
                if row not in selected: selected.append(row)
    if not selected:
        selected = report["snapshots"][:1]
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.3*len(selected)), squeeze=False)
    for i, row in enumerate(selected):
        for col, phase in enumerate(("ionized", "neutral")):
            item = next(v for v in row["mfp"] if v["threshold"] == threshold and v["phase"] == phase)
            for method in METHODS:
                val = item["methods"][method]
                # Empirical first-exit CDF below cap; censored rays never
                # become spurious exits at the rightmost histogram edge.
                if val.get("sample_count", 0):
                    axes[i,col].plot(val["histogram_edges_cMpc"], val["boundary_cdf_at_edges"], styles[method],
                                     color=colors[method], label=method)
            axes[i,col].set(title=f"seed {row['seed']} z={row['redshift']:g}, {phase}",
                            xlabel="First-exit length [cMpc]", ylabel="P(exit distance <= length)", ylim=(0,1.02))
            axes[i,col].grid(alpha=.2); axes[i,col].legend(fontsize=8)
    fig.suptitle(f"MFP distributions, xHI threshold {threshold:g}; same redshift, volume fractions may differ")
    fig.tight_layout(); fig.savefig(output / "mfp_distributions.png", dpi=160); plt.close(fig)


def run(input_run, output, *, rays=100000, seed=20261103, thresholds=(.1,.5,.9),
        threads=8, max_distance_boxes=1.):
    from research.pc_coeval.photon_audit import audit_snapshot
    from research.pc_coeval.mfp import make_draws, measure_mfp, compare_mfp, build_identity
    if type(rays) is not int or rays < 100 or type(threads) is not int or threads < 1:
        raise ValueError("Require >=100 rays and positive integer threads")
    if type(seed) is not int or seed < 0:
        raise ValueError("Require nonnegative integer random seed")
    thresholds = sorted(set(float(t) for t in thresholds))
    if not thresholds or not all(np.isfinite(t) and 0 < t < 1 for t in thresholds):
        raise ValueError("Thresholds must lie strictly within (0,1)")
    if not np.isfinite(max_distance_boxes) or max_distance_boxes <= 0:
        raise ValueError("Distance cap must be positive")
    root, old_report, snapshots, box_len = read_archive(input_run)
    output = Path(output).resolve()
    input_path = Path(input_run).resolve()
    input_container = root.parent if root.name == "results" else root
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (root, input_path, input_container)):
        raise ValueError("Output must be separate from readonly input")
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(rays=rays, seed=seed, thresholds=thresholds, threads=threads,
                    max_distance_boxes=float(max_distance_boxes))
    identity = {"input_root": str(root), "input_report_sha256": digest(root / "coeval_report.json"),
                "input_identity_sha256": digest(root / "identity.json"), "box_len_cMpc": box_len,
                "source_hashes": implementation_hashes(), "numpy_version": np.__version__, "mfp_build": build_identity(), "settings": settings,
                "snapshots": [{"path": s["folder"], "receipt_sha256": s["receipt_sha256"]} for s in snapshots]}
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Validation inputs or settings changed; use a new output")
        else:
            write_json(identity_path, identity)
        identity_hash = digest(identity_path)
        (output / "COMPLETED").unlink(missing_ok=True)
        report = {"schema": 1, "identity": identity, "snapshots": [],
                  "morphology_convergence_certified": False, "comparison_is_truth_error": False,
                  "definition": "Uniform phase-volume starting points, isotropic directions, periodic exact voxel first exit",
                  "mfp_caveats": ["Geometric length, not physical absorption MFP or a bubble radius",
                                  "Same-redshift comparison; no rescaling to equal volume filling fraction",
                                  "Capped rays are right-censored; restricted mean is a lower bound on the uncapped mean",
                                  "Ray count tests sampling only, not spatial resolution or cosmic variance"],
                  "references": ["https://arxiv.org/abs/0704.0946", "https://doi.org/10.1093/mnras/stw1542"]}
        draws = make_draws(rays, seed)
        for snapshot in snapshots:
            old = snapshot["row"]
            folder = output / f"seed{old['seed']}" / ("z" + f"{old['redshift']:g}".replace(".", "p"))
            folder.mkdir(parents=True, exist_ok=True)
            fields = load_fields(snapshot)  # Also recheck reused input, never trust cached metrics alone.
            node_identity = {"validation_identity_sha256": identity_hash, "input_receipt_sha256": snapshot["receipt_sha256"]}
            node_receipt = folder / "receipt.json"
            if node_receipt.exists():
                saved = json.loads(node_receipt.read_text())
                if saved["identity"] != node_identity: raise ValueError("Output node identity mismatch")
                verify_files(folder, saved)
                row = json.loads((folder / "metrics.json").read_text())
                print(f"reuse seed={old['seed']} z={old['redshift']:g}", flush=True)
            else:
                started = time.perf_counter()
                xhi = {m: fields[m + "_xhi"] for m in METHODS}
                audit = audit_snapshot(fields["source"], fields["capacity"], xhi,
                                       remaining_source=fields.get("pc_remaining_source"))
                row = {"seed": old["seed"], "redshift": old["redshift"], "shape": list(fields["source"].shape),
                       "cell_width_cMpc": box_len/fields["source"].shape[0], "photon_audit": audit,
                       "xHI_mass": {m: audit["methods"][m]["xHI_mass_recomputed"] for m in METHODS},
                       "xHI_volume": {m: audit["methods"][m]["xHI_volume_recomputed"] for m in METHODS},
                       "archived_mean_absolute_differences": {
                           m: {"mass": abs(audit["methods"][m]["xHI_mass_recomputed"]-old["xHI_mass"][m]),
                               "volume": abs(audit["methods"][m]["xHI_volume_recomputed"]-old["xHI_volume"][m])}
                           for m in METHODS},
                       "source_per_atom": float(np.sum(fields["source"],dtype=np.longdouble)/np.sum(fields["capacity"],dtype=np.longdouble)),
                       "mfp": []}
                arrays = {}
                for threshold in thresholds:
                    for phase in ("ionized", "neutral"):
                        measurements = {m: measure_mfp(xhi[m], box_len, phase=phase, threshold=threshold,
                                        draws=draws, max_distance=box_len*max_distance_boxes, threads=threads) for m in METHODS}
                        row["mfp"].append({"threshold": threshold, "phase": phase,
                                           "methods": {m: v["summary"] for m,v in measurements.items()},
                                           "comparisons": {m+"_vs_script": compare_mfp(measurements[m],measurements["script"])
                                                           for m in ("pc", "native_es")}})
                        for m,v in measurements.items():
                            name = f"t{threshold:g}_{phase}_{m}"
                            arrays[name+"_distance_cMpc"] = v["distance_cMpc"]
                            arrays[name+"_censored"] = v["censored"]
                row["elapsed_seconds"] = time.perf_counter()-started
                np.savez(folder / "ray_samples.tmp.npz", **arrays)
                (folder / "ray_samples.tmp.npz").replace(folder / "ray_samples.npz")
                write_json(folder / "metrics.json", row)
                write_json(node_receipt, {"identity": node_identity,
                           "files": {f: digest(folder/f) for f in ("metrics.json", "ray_samples.npz")}})
                print(f"seed={old['seed']} z={old['redshift']:g} budget_pass={audit['pc_and_script_allocation_passed']} "
                      f"MFP_seconds={row['elapsed_seconds']:.2f}", flush=True)
            report["snapshots"].append(row)
            write_json(output / "photon_mfp_report.json", report)
            del fields
            gc.collect()
        report["all_pc_script_allocation_budgets_passed"] = all(r["photon_audit"]["pc_and_script_allocation_passed"] for r in report["snapshots"])
        report["all_available_photon_checks_passed"] = all(r["photon_audit"]["all_available_required_checks_passed"] for r in report["snapshots"])
        write_json(output / "photon_mfp_report.json", report)
        plot_report(output, report)
        write_json(output / "report_receipt.json", {"identity_sha256": identity_hash,
                   "files": {f: digest(output/f) for f in ("photon_mfp_report.json", "photon_budget.png", "mfp_history.png", "mfp_distributions.png")}})
        # Completion describes postprocessing, never a morphology pass.
        (output / "COMPLETED").write_text(identity_hash + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rays", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--thresholds", default="0.1,0.5,0.9")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-distance-boxes", type=float, default=1.)
    args = parser.parse_args()
    result = run(args.input_run,args.output,rays=args.rays,seed=args.seed,
                 thresholds=[float(t) for t in args.thresholds.split(",")],threads=args.threads,
                 max_distance_boxes=args.max_distance_boxes)
    if not result["all_available_photon_checks_passed"]:
        raise SystemExit("Photon-budget verification failed; inspect the saved report")


if __name__ == "__main__":
    main()
