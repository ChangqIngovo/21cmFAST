"""Compare fixed-physics probes on a declared target grid and the coarse RT grid.

Target-grid agreement measures the label accuracy relevant to a field emulator.
It is reported separately from unresolved fine-front differences. No spatial
convergence result is inferred from photon closure alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_build = import_module("research.c2ray_teacher.build")
_runner = import_module("research.c2ray_teacher.runner")


def cell_average(values: np.ndarray, n: int) -> np.ndarray:
    """Project piecewise-constant cubic cells by their exact volume overlaps.

    Parameters
    ----------
    values : ndarray
        Cell averages on the original uniform cubic mesh.
    n : int
        Number of cells per target axis; may be an incommensurate coarser size.
    """
    m = len(values)
    if type(n) is not int or n < 1 or n > m or values.shape != (m,) * 3:
        raise ValueError("Projection requires a cubic array and a coarser target")
    if m % n == 0:
        q = m // n
        return values.reshape(n, q, n, q, n, q).mean((1, 3, 5))
    left = np.maximum(np.arange(n)[:, None] / n, np.arange(m)[None, :] / m)
    right = np.minimum((np.arange(n)[:, None] + 1) / n, (np.arange(m)[None, :] + 1) / m)
    overlap = np.maximum(0, right - left) * n
    first = np.einsum("ia,abc->ibc", overlap, values, optimize=True)
    second = np.einsum("jb,ibc->ijc", overlap, first, optimize=True)
    return np.einsum("kc,ijc->ijk", overlap, second, optimize=True)


def _load(case: Path) -> dict:
    fixture = json.loads((case / "fixture.json").read_text())
    state = _build.read_output(case / "output.bin")
    init = _runner._native_initialization(case / "native.log")
    ledger = _runner.read_transport_diagnostics(
        case / "build/runtime/results/transport_diagnostics.jsonl",
        state,
        init["sed_table_photons_over_nominal"],
        active_spectral_ratio=init["sed_transport_photons_over_nominal"],
    )
    manifest = json.loads((case / "build/build_manifest.json").read_text())
    if not fixture["dimension"] == manifest["dimension"] == state["dimension"]:
        raise ValueError("Probe fixture, build and output dimensions disagree")
    if _build.sha256(case / "input.bin") != fixture["input_sha256"]:
        raise ValueError("Probe input hash changed")
    if not all(np.isfinite(field).all() for field in state["fields"].values()):
        raise ValueError("Non-finite probe output")
    return {"fixture": fixture, "state": state, "ledger": ledger, "build": manifest}


def build_provenance(case: Path) -> dict:
    """Read immutable input and numerical build hashes for a completed probe.

    Parameters
    ----------
    case : Path
        Directory holding fixture.json and build/build_manifest.json.
    """
    manifest = json.loads((case / "build/build_manifest.json").read_text())
    fixture = json.loads((case / "fixture.json").read_text())
    return {
        "input_sha256": fixture["input_sha256"],
        "build_manifest_sha256": _build.sha256(case / "build/build_manifest.json"),
        **{
            key: manifest[key]
            for key in (
                "upstream_commit",
                "transport",
                "angular_order",
                "numerical_algorithm_unchanged",
                "transport_patch",
                "executable_sha256",
                "driver_sha256",
            )
        },
    }


def compare_probes(
    reference: Path, candidate: Path, target_dimension: int = 16
) -> dict:
    """Compare identical physical fixtures with explicit resolution dependence.

    Parameters
    ----------
    reference, candidate : Path
        Completed case directories containing fixture.json and output.bin.
    target_dimension : int
        Fixed target grid for emulator labels, distinct from each RT grid.
    """
    a, b = _load(reference), _load(candidate)
    for key in ("configuration", "source_positions_fractional"):
        if a["fixture"][key] != b["fixture"][key]:
            raise ValueError("Comparison requires identical physical sources and gas")
    c = a["fixture"]["configuration"]
    reports = []
    for index in range(len(a["state"]["steps"])):
        fields = [item["state"]["fields"] for item in (a, b)]
        x, y = [cell_average(f["xHII"][index], target_dimension) for f in fields]
        t, u = [
            cell_average(f["temperature_K"][index], target_dimension) for f in fields
        ]
        increment = max(
            float(np.abs(y - c["initial_xHII"]).sum()), np.finfo(float).tiny
        )
        thermal = max(
            float(np.abs(u - c["initial_temperature_K"]).sum()), np.finfo(float).tiny
        )
        freq = np.fft.fftfreq(target_dimension) * target_dimension
        k2 = (
            freq[:, None, None] ** 2
            + freq[None, :, None] ** 2
            + freq[None, None, :] ** 2
        )
        mask = (k2 > 0) & (k2 <= (target_dimension / 4) ** 2)
        px, py = [
            float(np.sum(np.abs(np.fft.fftn(v - v.mean())[mask]) ** 2)) for v in (x, y)
        ]
        coarse_n = min(f["xHII"].shape[1] for f in fields)
        coarse_a, coarse_b = [cell_average(f["xHII"][index], coarse_n) for f in fields]
        fine_normalizer = max(
            float(np.abs(coarse_b - c["initial_xHII"]).sum()), np.finfo(float).tiny
        )
        row = {
            "step": index + 1,
            "target_xHII_L1_over_ionized_increment": float(
                np.abs(x - y).sum() / increment
            ),
            "target_xHII_mean_relative_change": float(
                abs(x.sum() - y.sum()) / increment
            ),
            "target_temperature_L1_over_thermal_change": float(
                np.abs(t - u).sum() / thermal
            ),
            "target_low_k_power_relative_change": abs(px / py - 1) if py else None,
            "target_low_k_definition": "Summed neutral-fraction fluctuation power, 0 < k <= half target-grid Nyquist",
            "unfiltered_RT_grid_dimension": coarse_n,
            "unfiltered_RT_xHII_L1_over_ionized_increment": float(
                np.abs(coarse_a - coarse_b).sum() / fine_normalizer
            ),
        }
        row["target_diagnostic_passed"] = (
            row["target_xHII_L1_over_ionized_increment"] < 0.01
            and row["target_xHII_mean_relative_change"] < 0.005
            and row["target_temperature_L1_over_thermal_change"] < 0.05
            and row["target_low_k_power_relative_change"] is not None
            and row["target_low_k_power_relative_change"] < 0.01
        )
        reports.append(row)
    return {
        "scope": "Fixed physical two-source fixture; target-grid convergence and unfiltered front differences reported separately",
        "target_dimension": target_dimension,
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "reference_dimension": a["state"]["dimension"],
        "candidate_dimension": b["state"]["dimension"],
        "reference_angular_order": a["build"].get("angular_order"),
        "candidate_angular_order": b["build"].get("angular_order"),
        "reference_provenance": build_provenance(reference),
        "candidate_provenance": build_provenance(candidate),
        "reference_output_sha256": _build.sha256(reference / "output.bin"),
        "candidate_output_sha256": _build.sha256(candidate / "output.bin"),
        "reference_photon_residual": a["ledger"]["cumulative"][
            "photon_residual_over_transported_emitted"
        ],
        "candidate_photon_residual": b["ledger"]["cumulative"][
            "photon_residual_over_transported_emitted"
        ],
        "target_thresholds": {
            "ionized_L1": 0.01,
            "ionized_mean": 0.005,
            "thermal_L1": 0.05,
            "low_k_band_power": 0.01,
        },
        "target_diagnostics_passed": all(
            row["target_diagnostic_passed"] for row in reports
        ),
        "universal_convergence_certified": False,
        "steps": reports,
    }


def main() -> None:
    """Compare two completed probes and save an immutable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target-dimension", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output.exists():
        raise FileExistsError(output)
    report = compare_probes(**args)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    sys.stdout.write(
        json.dumps(
            {
                "output": str(output),
                "target_diagnostics_passed": report["target_diagnostics_passed"],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
