# Emulator of the analytic coeval photon-conserving teacher

This workflow trains on completed `pc_analytic.campaign` outputs. The labels are
our **analytic-column-characteristic** neutral fields, not SCRIPT or C²-Ray.
It reads the existing teacher archive and does not rerun the simulations.

## First training run on Gadi

From a clean checkout of `matched-photon-conservation`:

```bash
(
    set -euo pipefail
    cd /g/data/qp00/sz4640/21cmFAST
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation
    module load python3/3.12.1

    unset PC_TEACHER_RUN
    PC_RUN_ROOT=/g/data/qp00/sz4640/21cmfast_runs \
    PC_TRAIN_SEEDS=1001,1002,1003,1004,1005,1006 \
    PC_VALIDATION_SEEDS=1007 PC_TEST_SEEDS=1008 \
    PC_MODEL_SEEDS=11,23,47,83 \
    PC_WORKERS=4 PC_THREADS=4 PC_NCPUS=16 \
    PC_EMULATOR_EPOCHS=200 PC_SAMPLES_PER_SNAPSHOT=8192 \
    PC_MEM=64GB PC_WALLTIME=04:00:00 PC_QUEUE=normal \
    bash research/pc_analytic_emulator/gadi_prepare.sh \
        --latest-teacher --submit
)
```

`--latest-teacher` locates the latest completed matching campaign below
`PC_RUN_ROOT`: seeds 1001–1008, seven redshifts (10, 9, 8, 7.5, 7, 6.5, 6),
N=32 and L=256 cMpc. It prints the selected input and new training directory.
To select an exact completed campaign, replace `--latest-teacher` with
`--teacher-run /absolute/path/to/analytic_teacher_run`.

The launcher freezes the committed source, input hashes, configuration and
runtime. It reuses the teacher's Python interpreter and module environment.
Training needs NumPy, SciPy and Matplotlib; it does not import the native
21cmFAST extension or require PyTorch. The job is restricted to one node.

## Data split and fitting

| Role | Simulation seeds | Snapshots |
| --- | --- | ---: |
| Training | 1001–1006 | 42 |
| Validation | 1007 | 7 |
| Final test | 1008 | 7 |

Every redshift of an initial-condition seed stays in the same split. Input and
label arrays are checked against the native and teacher receipts, and the
teacher photon ledger is independently recomputed before fitting.

The field model is a two-layer (32, 32) tanh MLP learning a bounded correction
to native ES xHII. Its 24 input features contain local and Gaussian-smoothed
source, atom and ES fields, redshift and cell size. The four smoothing widths
are 6.25, 12.5, 25 and 50 cMpc. Teacher ionization and photon reservoirs are
labels only; they are never inference features.

A separate one-layer (8) MLP predicts the absorbed fraction

```text
eta = absorbed photons / min(source photons, atom capacity).
```

This retains the teacher's possibility of unabsorbed photons. It does not force
all available photons to be absorbed while neutral atoms remain. The field is
then projected to the predicted absorbed count. The complementary predicted
unabsorbed count is fixed before auditing the resulting field. This is a
**global photon-budget constraint**, not a learned ray-by-ray or source-by-source
transport solution; the predicted unabsorbed count is not measured escape.

Each of four model initializations uses 8192 uniformly sampled cells per training
snapshot (344064 cells total), all validation cells, Adam, at most 200 epochs and
20-epoch patience. Feature normalization is fitted on training samples only.
Four spawned workers use four BLAS threads each. The field checkpoint is chosen
by raw validation MSE and the budget checkpoint by validation eta MSE. The final
initialization is selected by full-grid **constrained validation MSE**, with a
zero-correction ES/eta=1 candidate retained. Test data never select a checkpoint
or model. A baseline winning is reported explicitly.

## Results and interpretation

Within the printed `analytic_train_...` directory:

| File under `results/` | Content |
| --- | --- |
| `summary.json` | Split, selection, held-out errors and maximum global budget residual |
| `model/field_model.npz` | Fitted field network and its training normalizer |
| `model/budget_model.npz` | Fitted global absorption network and normalizer |
| `model/model.json` | Inference configuration and teacher provenance |
| `model_selection.json` | Validation scores for all initializations and baseline candidates |
| `trials/seed*/history.json` | Per-epoch field and absorption losses |
| `dataset_manifest.json` | Audited input hashes, physical settings and seed split |
| `test_metrics.csv`, `test_metrics.json` | Raw, constrained and native ES errors against the teacher |
| `test/*/predictions.npz` | Held-out fields and predicted photon reservoirs |
| `test_slices.png` | Native ES, teacher, emulator and residual slices |
| `test_power_ratios.png` | 21-cm power relative to the teacher |
| `test_errors.png` | Held-out field and mass-ionized-fraction errors |

21-cm brightness is reconstructed from the original cosmological parameters
using the explicit `BrightnessTemperatureBox.c` formula for `USE_TS_FLUCT=False`.
The reconstruction is checked against saved native and teacher brightness at
float32 rounding tolerance. Power ratios and cross-correlations with zero
reference power are undefined, not counted as perfect agreement.
The summary separately aggregates active snapshots with teacher mass-weighted
neutral fraction between 0.05 and 0.95. Per-snapshot reports also quantify the
raw-to-constrained field change and absorption/unabsorbed-count prediction errors.

The learned scope is the teacher's scope: frozen coeval density, initially
neutral gas, cumulative native n_ion, no recombinations, thermal evolution,
photoheating feedback or independent helium chemistry. Successful fitting does
not add these processes or establish a continuum solution. Held-out performance
tests new initial conditions at the sampled grid and physical parameters; this
first run does not test new cosmologies, source prescriptions or resolutions.

## Resume and local checks

Resubmit the same frozen training run using its generated command:

```bash
bash /absolute/path/to/analytic_train_run/resubmit.sh
```

Completed trials with valid receipts are reused. An interrupted trial restarts;
this is trial-level recovery, not optimizer-state recovery within an epoch.
A completed run is checksum-verified without refitting or selecting on its test
results. Changed source, teacher inputs or runtime require a newly prepared run.

For a separately prepared JSON configuration and output directory:

```bash
python -m research.pc_analytic_emulator.train \
    --config /absolute/path/to/experiment_config.json \
    --output /absolute/path/to/training_results
```

Targeted checks from a development environment:

```bash
uv run --no-project python -m pytest -q research/pc_analytic_emulator
uv run --no-project python -m ruff check research/pc_analytic_emulator
uv run --no-project python -m ruff format --check research/pc_analytic_emulator
bash -n research/pc_analytic_emulator/gadi_prepare.sh
bash -n research/pc_analytic_emulator/gadi_train.pbs
```

Test fixtures contain synthetic receipts and fields to exercise the workflow;
their scores are not physical-validation or held-out simulation results.
