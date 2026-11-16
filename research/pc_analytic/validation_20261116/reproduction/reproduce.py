"""Reproduce eight fitting/regression and four independent test realizations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the frozen, seed-separated diagnostic configuration in a new folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = Path(__file__).resolve().parent
    shutil.copyfile(configs / "native_template.json", output / "native_template.json")
    campaign = json.loads((configs / "teacher_config.json").read_text())
    external = json.loads((configs / "external_teacher_config.json").read_text())
    campaign["native_template_json"] = str(output / "native_template.json")
    external["native_template_json"] = str(output / "native_template.json")
    training = json.loads((configs / "training_config.json").read_text())
    training["input_root"] = str(output / "teacher_results")
    training["external_test_root"] = str(output / "external_teacher_results")
    (output / "campaign_config.json").write_text(json.dumps(campaign, indent=2) + "\n")
    (output / "external_campaign_config.json").write_text(
        json.dumps(external, indent=2) + "\n"
    )
    (output / "training_config.json").write_text(json.dumps(training, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        OMP_THREAD_LIMIT="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        OMP_DYNAMIC="FALSE",
    )
    for module, config, destination in [
        ("research.pc_analytic.campaign", "campaign_config.json", "teacher_results"),
        (
            "research.pc_analytic.campaign",
            "external_campaign_config.json",
            "external_teacher_results",
        ),
        (
            "research.pc_analytic_emulator.train",
            "training_config.json",
            "emulator_results",
        ),
    ]:
        stage_env = env.copy()
        if destination == "emulator_results":
            stage_env.update(OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with (output / f"{destination}.log").open("w") as log:
            subprocess.run(
                [
                    args.python,
                    "-m",
                    module,
                    "--config",
                    str(output / config),
                    "--output",
                    str(output / destination),
                ],
                cwd=source,
                env=stage_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    main()
