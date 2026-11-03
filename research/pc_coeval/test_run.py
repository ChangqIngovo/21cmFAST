"""Independent checks of comparison semantics and the streaming archive reader."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.pc_coeval.run import compare_fields, digest, discover, freeze_identity, run


class StreamingArchiveTests(unittest.TestCase):
    def make_archive(self, root):
        (root / "manifest.json").write_text(json.dumps({"configuration": {"seeds": [222]}}))
        history = root / "histories" / "seed222"
        history.mkdir(parents=True)
        (history / "history_manifest.json").write_text('{"schema":1}')
        # Deliberately invalid large-history data: discovery must never open it.
        (history / "history.npz").write_bytes(b"DO NOT READ THIS HISTORY")
        files = {"history_manifest.json": digest(history / "history_manifest.json"),
                 "history.npz": "unused-history-checksum"}
        for z in (8., 7., 6.):
            folder = history / "native" / "seed222" / f"z{z:g}"
            folder.mkdir(parents=True)
            receipt = folder / "receipt.json"
            receipt.write_text(json.dumps({"record": {"seed": 222, "redshift": z}}))
            files[str(receipt.relative_to(history))] = digest(receipt)
        receipt = history / "history_receipt.json"
        receipt.write_text(json.dumps({"files": files, "identity": {"schema": 1}}))
        return history

    def test_discovery_never_loads_full_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archive(root)
            with patch("numpy.load", side_effect=AssertionError("No array loads in discovery")):
                _, _, snapshots = discover(root, [6, 8])
            self.assertEqual([s["record"]["redshift"] for s in snapshots], [8, 6])

    def test_missing_requested_snapshot_is_not_silently_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archive(root)
            with self.assertRaisesRegex(ValueError, "absent"):
                discover(root, [7.25])

    def test_tampered_selected_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_archive(root)
            path = history / "native/seed222/z7/receipt.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                discover(root, [7])

    def test_history_identity_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_archive(root)
            (history / "history_manifest.json").write_text('{"schema":2}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                discover(root, [7])

    def test_output_cannot_modify_input(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "separate"):
                run(directory, Path(directory) / "new_results")

    def test_resume_preserves_identity_bytes_for_equivalent_numeric_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            freeze_identity(path, {"timeout": 120.0, "threads": 4})
            original = path.read_bytes()
            saved = freeze_identity(path, {"timeout": 120, "threads": 4})
            self.assertIsInstance(saved["timeout"], float)
            self.assertEqual(path.read_bytes(), original)

    def test_changed_identity_never_overwrites_frozen_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            freeze_identity(path, {"timeout": 120.0})
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze_identity(path, {"timeout": 900.0})
            self.assertEqual(path.read_bytes(), original)


class ComparisonTests(unittest.TestCase):
    def test_equal_mass_budget_does_not_imply_equal_morphology_or_volume(self):
        capacity = np.ones((4, 4, 4))
        capacity[0] = 2.
        source = np.full_like(capacity, .25)
        pc = np.ones_like(capacity)
        script = pc.copy()
        pc[0] = .5  # 16 atoms ionized in dense layer.
        script[1] = 0.  # 16 atoms in another layer, twice the ionized volume.
        es = script.copy()
        report = compare_fields(source, capacity, pc, script, es,
                                {"pc": capacity*pc, "script": capacity*script,
                                 "native_es": capacity*es}, 50.)
        self.assertAlmostEqual(report["xHI_mass"]["pc"], .8)
        self.assertAlmostEqual(report["xHI_mass"]["script"], .8)
        self.assertAlmostEqual(report["expected_xHI_mass"], .8)
        self.assertNotEqual(report["xHI_volume"]["pc"], report["xHI_volume"]["script"])
        self.assertGreater(report["pc_vs_script"]["MAE"], .1)

    def test_zero_power_is_undefined_not_perfect_match(self):
        capacity = np.ones((4, 4, 4))
        zero = np.zeros_like(capacity)
        report = compare_fields(capacity*2, capacity, zero, zero, zero,
                                {"pc": zero, "script": zero, "native_es": zero}, 50.)
        self.assertEqual(report["expected_xHI_mass"], 0.)
        self.assertTrue(all(x is None for x in report["spectra_21cm"]["ratio_pc"]))
        self.assertIn("P_script", report["spectra_21cm"])
        self.assertNotIn("P_teacher", report["spectra_21cm"])


if __name__ == "__main__":
    unittest.main()
