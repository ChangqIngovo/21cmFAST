"""Archive-integrity and full postprocessing regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.pc_coeval.validate import array_digest, digest, load_fields, read_archive, run, write_json


def make_archive(base, *, bad_pc=False, reservoir=False):
    """Synthetic allocations only; not a native physics benchmark."""
    base = Path(base)
    native = base / "native"
    native.mkdir()
    write_json(native / "manifest.json", {"configuration": {"box_len": 32.}})
    root = base / "input" / "results"
    root.mkdir(parents=True)
    identity = {"input_root": str(native), "input_manifest_sha256": digest(native / "manifest.json"), "snapshots": []}
    rows, fieldsets = [], []
    for z, q in ((8., .5), (6., 2.)):
        capacity = np.ones((8,8,8))
        source = capacity*q
        pc = np.zeros_like(capacity)
        script = np.zeros_like(capacity)
        if q < 1:
            pc[:4] = 1
            script[::2] = 1
            if bad_pc: pc[0,0,0] = .5
        fields = {"source": source, "capacity": capacity, "pc_xhi": pc,
                  "script_xhi": script, "native_es_xhi": np.maximum(pc, .01)}
        if reservoir: fields["pc_remaining_source"] = capacity*max(0,q-1)
        record = {"seed": 222, "redshift": z,
                  "source_sha256": array_digest(source), "capacity_sha256": array_digest(capacity)}
        identity["snapshots"].append({"record": record, "receipt_sha256": "a"*64})
        row = {"seed":222,"redshift":z,"common_input":{
                "shape":list(capacity.shape), "source_sha256_float64":array_digest(source),
                "capacity_sha256_float64":array_digest(capacity)},
               "xHI_mass": {m:float(fields[m+"_xhi"].mean()) for m in ("pc","script","native_es")},
               "xHI_volume": {m:float(fields[m+"_xhi"].mean()) for m in ("pc","script","native_es")},
               "spectra_21cm":{"shell_width_cMpc_inv":2*np.pi/32.}}
        rows.append(row); fieldsets.append(fields)
    write_json(root / "identity.json", identity)
    identity_hash = digest(root / "identity.json")
    for row, fields in zip(rows, fieldsets):
        folder = root / "seed222" / f"z{row['redshift']:g}"
        folder.mkdir(parents=True)
        np.savez(folder/"fields.npz", **fields)
        write_json(folder/"metrics.json", row)
        write_json(folder/"receipt.json", {"identity":{"run_identity_sha256":identity_hash,"input_receipt_sha256":"a"*64},
                   "files":{n:digest(folder/n) for n in ("fields.npz","metrics.json")}})
    write_json(root/"coeval_report.json", {"identity":identity,"snapshots":rows})
    write_json(root/"report_receipt.json", {"identity_sha256":identity_hash,"files":{"coeval_report.json":digest(root/"coeval_report.json")}})
    (root/"COMPLETED").write_text(identity_hash+"\n")
    return root


def read_json(path): return json.loads(Path(path).read_text())


def dummy_plots(output, report):
    for name in ("photon_budget.png","mfp_history.png","mfp_distributions.png"):
        (output/name).write_bytes(b"test-only-plot")


class ValidationTests(unittest.TestCase):
    def test_full_run_detects_shape_difference_and_censoring_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); root=make_archive(base, reservoir=True); output=base/"validation"
            before={str(p):digest(p) for p in root.rglob("*") if p.is_file()}
            report=run(root.parent,output,rays=1000,threads=2)
            self.assertTrue(report["all_available_photon_checks_passed"])
            self.assertFalse(report["morphology_convergence_certified"])
            partial=report["snapshots"][0]
            self.assertGreater(partial["mfp"][0]["comparisons"]["pc_vs_script"]["restricted_wasserstein1_cMpc"],1.)
            full=report["snapshots"][1]
            ionized=next(i for i in full["mfp"] if i["phase"]=="ionized" and i["threshold"]==.5)
            neutral=next(i for i in full["mfp"] if i["phase"]=="neutral" and i["threshold"]==.5)
            self.assertEqual(ionized["methods"]["pc"]["censored_fraction"],1.)
            self.assertIsNone(neutral["methods"]["pc"]["restricted_mean_cMpc"])
            self.assertEqual(before,{str(p):digest(p) for p in root.rglob("*") if p.is_file()})
            node_hashes={str(p):digest(p) for p in output.glob("seed*/z*/*") if p.is_file()}
            resumed=run(root.parent,output,rays=1000,threads=2,max_distance_boxes=1)
            self.assertEqual(report,resumed)
            self.assertEqual(node_hashes,{str(p):digest(p) for p in output.glob("seed*/z*/*") if p.is_file()})
            self.assertGreater((output/"mfp_distributions.png").stat().st_size,10000)

    def test_field_tampering_is_detected_before_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=make_archive(directory)
            _,_,snapshots,_=read_archive(root)
            with (root/"seed222/z8/fields.npz").open("ab") as stream: stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError,"checksum"):
                load_fields(snapshots[0])

    def test_actual_array_hash_checked_beyond_file_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=make_archive(directory); folder=root/"seed222/z8"
            with np.load(folder/"fields.npz") as saved: fields={k:saved[k] for k in saved.files}
            fields["source"]*=1.1
            np.savez(folder/"fields.npz",**fields)
            receipt=read_json(folder/"receipt.json");receipt["files"]["fields.npz"]=digest(folder/"fields.npz")
            write_json(folder/"receipt.json",receipt)
            _,_,snapshots,_=read_archive(root)
            with self.assertRaisesRegex(ValueError,"input-array checksum"):
                load_fields(snapshots[0])

    def test_failed_conservation_is_reported_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as directory, patch("research.pc_coeval.validate.plot_report",dummy_plots):
            root=make_archive(directory,bad_pc=True)
            report=run(root,Path(directory)/"out",rays=100,threads=1,thresholds=[.5])
            self.assertFalse(report["all_available_photon_checks_passed"])
            self.assertFalse(report["all_pc_script_allocation_budgets_passed"])
            self.assertGreater(report["snapshots"][0]["photon_audit"]["methods"]["pc"]["absolute_defect_per_atom"],0)

    def test_missing_reservoir_is_explicit_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory, patch("research.pc_coeval.validate.plot_report",dummy_plots):
            root=make_archive(directory)
            report=run(root,Path(directory)/"out",rays=100,threads=1,thresholds=[.5])
            self.assertTrue(report["all_available_photon_checks_passed"])
            self.assertFalse(report["snapshots"][1]["photon_audit"]["independent_reservoir_audit_available"])

    def test_bad_saved_reservoir_fails_even_when_allocation_is_correct(self):
        with tempfile.TemporaryDirectory() as directory, patch("research.pc_coeval.validate.plot_report",dummy_plots):
            root=make_archive(directory,reservoir=True); folder=root/"seed222/z6"
            with np.load(folder/"fields.npz") as saved: fields={k:saved[k] for k in saved.files}
            fields["pc_remaining_source"]*=.9
            np.savez(folder/"fields.npz",**fields)
            receipt=read_json(folder/"receipt.json");receipt["files"]["fields.npz"]=digest(folder/"fields.npz")
            write_json(folder/"receipt.json",receipt)
            report=run(root,Path(directory)/"out",rays=100,threads=1,thresholds=[.5])
            self.assertTrue(report["all_pc_script_allocation_budgets_passed"])
            self.assertFalse(report["all_available_photon_checks_passed"])

    def test_changed_settings_cannot_reuse_outputs(self):
        with tempfile.TemporaryDirectory() as directory, patch("research.pc_coeval.validate.plot_report",dummy_plots):
            root=make_archive(directory);out=Path(directory)/"out"
            run(root,out,rays=100,threads=1,thresholds=[.5])
            with self.assertRaisesRegex(ValueError,"settings changed"):
                run(root,out,rays=200,threads=1,thresholds=[.5])

    def test_no_write_inside_input_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=make_archive(directory)
            with self.assertRaisesRegex(ValueError,"separate"):
                run(root.parent,root.parent/"analysis",rays=100)

    def test_missing_completion_and_changed_native_manifest_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=make_archive(directory)
            marker=root/"COMPLETED";original=marker.read_text();marker.write_text("changed")
            with self.assertRaisesRegex(ValueError,"not completed"):read_archive(root)
            marker.write_text(original)
            (Path(directory)/"native/manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"manifest checksum"):read_archive(root)


if __name__ == "__main__":unittest.main()
