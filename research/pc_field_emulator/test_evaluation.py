"""Independent numerical checks for field-emulator inputs and observables."""

from __future__ import annotations

import inspect
import json
import unittest

import numpy as np
import pytest

try:
    from .evaluation import evaluate_fields, field_metrics, shell_spectra
    from .features import build_features
except ImportError:
    from evaluation import evaluate_fields, field_metrics, shell_spectra
    from features import build_features


class FeatureTests(unittest.TestCase):
    """Check input-only features and periodic translation covariance."""

    def test_features_have_no_teacher_input(self) -> None:
        """Ensure constant input fields produce constant input-only features."""
        assert list(inspect.signature(build_features).parameters) == [
            "capacity",
            "photons",
            "redshift",
            "box_len",
            "radii_cells",
            "radii_cMpc",
        ]
        rho = np.ones((6, 6, 6))
        photons = np.full_like(rho, 0.25)
        features, names = build_features(rho, photons, 8.0, 60.0)
        assert features.dtype == np.float32
        assert features.shape == (rho.size, len(names))
        np.testing.assert_allclose(
            features, np.broadcast_to(features[0], features.shape)
        )
        assert "log1p_emitted_per_atom" in names
        assert np.all(np.isfinite(features))

    def test_features_translate_with_fields(self) -> None:
        """Check periodic smoothing does not privilege a spatial origin."""
        rng = np.random.default_rng(7)
        rho = rng.uniform(0.1, 2.0, (8, 8, 8))
        photons = rng.uniform(0.0, 1.0, rho.shape)
        first, _ = build_features(rho, photons, 8.0, 80.0, (1.0, 2.0))
        second, _ = build_features(
            np.roll(rho, 2, axis=0),
            np.roll(photons, 2, axis=0),
            8.0,
            80.0,
            (1.0, 2.0),
        )
        np.testing.assert_allclose(
            second.reshape(*rho.shape, -1),
            np.roll(first.reshape(*rho.shape, -1), 2, axis=0),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_zero_capacity_cells_are_finite(self) -> None:
        """Keep empty cells finite and reject invalid total capacity."""
        rho = np.ones((4, 4, 4))
        rho[0] = 0.0
        features, _ = build_features(rho, np.ones_like(rho), 8.0, 40.0)
        assert np.all(np.isfinite(features))
        with pytest.raises(ValueError, match="positive finite atom capacity"):
            build_features(np.zeros_like(rho), rho, 8.0, 40.0)
        with pytest.raises(
            ValueError, match="Photon counts must be finite and nonnegative"
        ):
            build_features(rho, -rho, 8.0, 40.0)


class SpectrumTests(unittest.TestCase):
    """Check FFT normalization independently of the spectrum implementation."""

    def test_single_cosine_parseval_and_shell_mode_weights(self) -> None:
        """Recover cosine variance using independent analytic normalization."""
        n, box_len, amplitude = 8, 80.0, 3.0
        x = np.arange(n)[:, None, None]
        field = np.broadcast_to(amplitude * np.cos(2 * np.pi * x / n), (n,) * 3)
        result = shell_spectra({"teacher": field, "raw": 2 * field}, box_len)
        modes = np.asarray(result["modes"])
        power = np.asarray(result["P_teacher"])
        assert modes[0] == 18
        np.testing.assert_allclose(
            np.sum(modes * power) / box_len**3,
            amplitude**2 / 2,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(result["ratio_raw"][0], 4.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["r_raw"][0], 1.0, rtol=1e-12, atol=1e-12)

    def test_nyquist_has_single_conjugate_weight(self) -> None:
        """Avoid double counting the real-valued Nyquist plane."""
        n, box_len = 8, 8.0
        field = np.broadcast_to((-1.0) ** np.arange(n)[None, None, :], (n,) * 3)
        result = shell_spectra({"teacher": field}, box_len)
        np.testing.assert_allclose(
            sum(
                m * p for m, p in zip(result["modes"], result["P_teacher"], strict=True)
            )
            / box_len**3,
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_constant_spectrum_has_no_defined_ratios(self) -> None:
        """Represent a zero-power denominator as undefined instead of unity."""
        result = shell_spectra(
            {"teacher": np.ones((8, 8, 8)), "raw": np.zeros((8, 8, 8))},
            80.0,
        )
        assert all(x is None for x in result["ratio_raw"])
        assert all(x is None for x in result["r_raw"])
        assert all(x == 0.0 for x in result["P_teacher"])
        json.dumps(result, allow_nan=False)

    def test_phase_error_detected_by_cross_correlation(self) -> None:
        """Distinguish equal power from equal spatial phase."""
        n = 8
        x = np.arange(n)[:, None, None]
        first = np.broadcast_to(np.cos(2 * np.pi * x / n), (n,) * 3)
        second = np.broadcast_to(np.sin(2 * np.pi * x / n), (n,) * 3)
        result = shell_spectra({"teacher": first, "raw": second}, 80.0)
        np.testing.assert_allclose(result["ratio_raw"][0], 1.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["r_raw"][0], 0.0, rtol=1e-12, atol=1e-12)


class MetricsTests(unittest.TestCase):
    """Check mass weighting and undefined diagnostics."""

    def test_neutral_island_metric_detects_late_island_loss(self) -> None:
        """A high ionized IoU must not conceal a completely lost neutral island."""
        teacher = np.ones((4, 4, 4))
        teacher[0, 0, 0] = 0.0
        result = field_metrics(np.ones_like(teacher), teacher, np.ones_like(teacher))
        assert result["ionized_IoU"] == 63 / 64
        assert result["neutral_IoU"] == 0.0
        assert result["neutral_cell_count"] == 1
        assert result["neutral_cell_MAE"] == 1.0

    def test_mass_and_volume_are_distinct(self) -> None:
        """Check atom weighting and the empty partial-ionization mask."""
        rho = np.ones((2, 2, 2))
        rho[0] = 3.0
        teacher = np.zeros_like(rho)
        teacher[0] = 1.0
        result = field_metrics(teacher, teacher, rho)
        np.testing.assert_allclose(result["Q_volume"], 0.5, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["Q_mass"], 0.75, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["MAE"], 0.0, rtol=1e-12, atol=1e-12)
        assert result["partial_cell_MAE"] is None
        np.testing.assert_allclose(result["front_MAE"], 0.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["ionized_IoU"], 1.0, rtol=1e-12, atol=1e-12)

    def test_full_report_is_finite_and_contains_baseline(self) -> None:
        """Keep undefined spectra JSON-safe while including the simple baseline."""
        rho = np.ones((8, 8, 8))
        teacher = np.full_like(rho, 0.4)
        report = evaluate_fields(
            rho,
            teacher,
            teacher + 0.1,
            teacher,
            8.0,
            80.0,
            baseline=np.full_like(rho, 0.3),
        )
        assert "baseline" in report["metrics"]
        np.testing.assert_allclose(
            report["metrics"]["raw"]["MAE"], 0.1, rtol=1e-12, atol=1e-12
        )
        assert report["metrics"]["projected"]["RMSE"] == 0.0
        assert "P_baseline" in report["spectra"]
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
