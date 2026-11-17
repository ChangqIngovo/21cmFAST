"""Assemble one trained exposure model from its fitted heads."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from research.pc_analytic_emulator.exposure_emulator import (
    ExposureEmulator,
    implementation_hashes,
)
from research.pc_analytic_emulator.exposure_onset import OnsetHead
from research.pc_analytic_emulator.predict import _digest


def assemble(base: Path, onset: Path, depth: int, output: Path) -> None:
    """Join the fitted onset, field and budget heads and pin their inference code."""
    model = ExposureEmulator.load(base)
    head = OnsetHead.load(onset)
    if type(depth) is not int or depth < 1 or depth > model.metadata["depth"]:
        raise ValueError("Invalid post-onset layer count")
    metadata = dict(model.metadata)
    seeds = metadata["split_seeds"]
    onset_train = set(head.metadata["training_seeds"])
    if onset_train & set(seeds["validation"]):
        raise ValueError("Onset head training overlaps field-model validation")
    metadata.update(
        depth=depth,
        field_fit_depth=model.metadata["depth"],
        implementation_sha256=implementation_hashes(),
        onset_sha256=_digest(onset),
        split_seeds={
            **seeds,
            "train": sorted(set(seeds["train"]) | onset_train),
            "onset_development": head.metadata.get("excluded_observed_seeds", []),
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(base / "weights.npz", output / "weights.npz")
    shutil.copyfile(onset, output / "onset.npz")
    (output / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    ExposureEmulator.load(output)


def main() -> None:
    """Create a reproducible frozen model directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "onset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    args = parser.parse_args()
    assemble(args.base, args.onset, args.depth, args.output)


if __name__ == "__main__":
    main()
