# Native teacher generation

Run this from a checkout containing the completion workflow, using a Python
environment with the supported native 21cmFAST extension already installed:

```bash
python research/pc_analytic_emulator/completion_20261116/reproduction/generate_teachers.py \
  --source "$PWD" \
  --commit "$(git rev-parse HEAD)" \
  --native-python /absolute/path/to/native-environment/bin/python \
  --output /absolute/path/to/new-teacher-campaign \
  --workers 2 --threads 2
```

The launcher creates a detached source worktree, freezes the complete native
input template and numerical configuration, records the native binary and code
hashes, and runs two seed workers with two threads each. Each snapshot retains
its original native HDF5, source/atom arrays, teacher fields and audit receipts.
Reusing the same command and commit resumes completed snapshots only when the
frozen identities still match. The native Python path must name its environment
executable; the launcher preserves virtual-environment symlinks.

| Campaign | Seeds | Redshifts | Grid and box | Snapshots |
|---|---|---|---|---:|
| Development original | 2101–2108 | 10, 9, 8, 7.5, 7, 6.5, 6 | 16³, 50 cMpc | 56 |
| Development external | 2109–2112 | Same seven | 16³, 50 cMpc | 28 |
| Expanded training | 2121–2128 | 10, 9, 8, 7.5, 7, 6.75, 6.5, 6.25, 6, 5.75 | 16³, 50 cMpc | 80 |
| Validation | 2113–2116 | Same seven | 16³, 50 cMpc | 28 |
| Blind test | 2117–2120 | Same seven | 16³, 50 cMpc | 28 |
| Resolution transfer | 2131–2132 | 8, 7.5, 6.5, 6 | 32³, 50 cMpc | 8 |
| Volume transfer | 2141–2142 | 8, 7.5, 6.5, 6 | 32³, 256 cMpc | 8 |

The default generates 236 snapshots: 220 for the primary campaign and 16 for
separate geometry-transfer checks. `--profiles validation` runs just that
campaign; other profile names are listed by `--help`.

`training_collection/collection_manifest.json` references unchanged campaign
receipts for 164 training and 28 validation snapshots. It excludes blind and
transfer campaigns. Those outputs are never used for model selection and are
evaluated only after the selected model and inference settings are frozen. Earlier seeds
2101–2112 have already been inspected and are development data.

`train_and_validate.py` compares the two trained feature configurations and the
unchanged previous checkpoint using the same validation inputs and objective.
The previous checkpoint remains an eligible fallback and wins exact ties.
Selection is recorded before the blind and transfer benchmarks begin; the
previous checkpoint also remains their fixed comparison reference. See
[GADI.md](GADI.md) for the training and evaluation launcher.

The teacher uses 96 directions, 32 exposure steps, characteristic transport and
the `source_periodic_cube` boundary, with separately recorded escaped photons.
Native initial conditions use DIM=96. The supplied template retains the tested
21cmFAST physical parameters: initially neutral frozen coeval fields, cumulative
native sources, and no recombinations, thermal evolution or photoheating
feedback. These campaigns test that declared approximation.

Logs and original completion receipts remain under each output campaign. The
launcher reports only completion and photon-budget checks; it does not evaluate
blind morphology or power spectra.
