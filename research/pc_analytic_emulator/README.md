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

    unset PC_TEACHER_RUN PC_EXTERNAL_TEST_ROOT PC_TRAINING_METHOD
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

## Improved training on the existing teachers

The original fitting path remains the default. Add `--improved` to select
`projected_multiscale_v2` without changing the teacher solver or input arrays:

```bash
(
    set -euo pipefail
    cd /g/data/qp00/sz4640/21cmFAST
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation
    module load python3/3.12.1

    unset PC_TEACHER_RUN PC_EXTERNAL_TEST_ROOT
    PC_RUN_ROOT=/g/data/qp00/sz4640/21cmfast_runs \
    PC_TRAIN_SEEDS=1001,1002,1003,1004,1005,1006 \
    PC_VALIDATION_SEEDS=1007 PC_TEST_SEEDS=1008 \
    PC_MODEL_SEEDS=11,23,47,83 \
    PC_WORKERS=4 PC_THREADS=4 PC_NCPUS=16 \
    PC_EMULATOR_EPOCHS=200 PC_V2_EPOCHS=80 \
    PC_SAMPLES_PER_SNAPSHOT=8192 \
    PC_MEM=64GB PC_WALLTIME=04:00:00 PC_QUEUE=normal \
    bash research/pc_analytic_emulator/gadi_prepare.sh \
        --latest-teacher --improved --submit
)
```

The v2 field head adds an unbounded linear-output correction to native ES xHII.
For latent ionization scores `u` and cell atom capacities `A`, it finds the
bounded field `q` minimizing `sum(A * (q - u)**2) / 2`, subject to the independently
predicted absorbed-photon count `sum(A*q)`. Scores outside [0, 1] retain freedom
to place cells exactly at the neutral or ionized boundary; only the diagnostic
raw field is clipped. Model schema 2 records this projection and field type;
schema 1 models keep their original behavior.

Seed 1008 has already been inspected. It is now a **regression test** for the
improved method, not evidence from a fresh unseen simulation. It remains
excluded from fitting and checkpoint selection; seed 1007 supplies validation.

An optional `--external-test-root /absolute/path/to/completed_teacher_run`
adds a separately audited archive to the same command. Its labels are evaluated
only after the final model is selected. No files or samples are merged into the
original training archive. Both seed identifiers and initial-condition hashes
must be new relative to all eight original seeds. Physical inputs, teacher
settings, implementation and native runtime must match. The launcher freezes
both archives' checksums and rejects later changes. This option evaluates
existing completed teachers; it does not generate new simulations.
For a future campaign, choose unused seeds such as 1009 and 1010 and preserve
the original campaign's recorded physical and solver settings. Its
`parent_run.txt` identifies the original native-source runtime parent required
by `research/pc_analytic/gadi_teacher_prepare.sh`; the teacher run itself is not
a replacement for that parent.

## Data split and fitting

| Role | Simulation seeds | Snapshots |
| --- | --- | ---: |
| Training | 1001–1006 | 42 |
| Validation | 1007 | 7 |
| Previously inspected regression test | 1008 | 7 |

Every redshift of an initial-condition seed stays in the same split. Input and
label arrays are checked against the native and teacher receipts, and the
teacher photon ledger is independently recomputed before fitting.

### Legacy objective

The legacy field model is a two-layer (32, 32) tanh MLP learning a bounded correction
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

### Improved objective

V2 starts from an exactly zero latent correction to native ES; it does not import
a fitted v1 checkpoint. Feature normalization uses training cells only. The
field optimizer processes complete snapshots, differentiates through the
photon-budget projection, and minimizes a front-weighted, stage-normalized
field error plus a large-scale Fourier brightness residual. The Fourier term
compares complex modes, including their phases; it is not a power-only fit.
Shells receive equal weight, and the default band stops at half the grid
Nyquist frequency. A normalized latent-field penalty assists optimization.
The relative Fourier denominator has a floor of 1e-4 times the fully neutral
density field's mean-square brightness, so almost vanished reference signals
do not dominate selection. Exactly zero teacher-power shells remain excluded.
Every cell participates in v2 fitting; `PC_SAMPLES_PER_SNAPSHOT` applies only
to the legacy field fit.

Defaults are 80 v2 epochs, front weight 1, Fourier weight 0.1, latent penalty
weight 0.05, variance floor 0.01 and Nyquist fraction 0.5. They are controlled by
`PC_V2_EPOCHS`, `PC_FRONT_WEIGHT`, `PC_SPECTRAL_WEIGHT`, `PC_RAW_WEIGHT`,
`PC_FIELD_VARIANCE_FLOOR`, `PC_K_FRACTION` and `PC_SPECTRAL_POWER_FLOOR`.
`PC_EMULATOR_EPOCHS` controls
the budget MLP and the retained legacy fits. The latent penalty does not select
checkpoints: the deployed, projected field-plus-Fourier objective does. Epoch
zero remains an eligible candidate.

For the projected candidates, validation first selects one absorption estimator
from eta=1, a physical-budget-weighted MLP, and a training-only table interpolating mean
absorption efficiency against photon load. That choice is independent of
the field architecture. The final comparison records all four combinations
of learned/native-ES field and selected/eta=1 absorption. This separates changes
in absorption budget from changes in spatial morphology. The original v1
training also runs: both its fitted candidate and its original selected
candidate are retained, producing six candidates per initialization. All six
are scored with the same v2 validation objective. The legacy candidates retain
their independently fitted legacy absorption models.
Validation selects the final deployed candidate, which may remain the legacy
model. Neither internal regression labels nor external test labels make this
choice. `--improved` runs this comparison and selection; its name is not an
accuracy result.

## Results and interpretation

Within the printed `analytic_train_...` directory:

| File under `results/` | Content |
| --- | --- |
| `summary.json` | Split, selection, evaluation roles, errors and maximum global budget residual |
| `model/field_model.npz` | Fitted field network and its training normalizer |
| `model/budget_model.npz` | Selected absorption estimator: network, table or reference |
| `model/model.json` | Inference configuration and teacher provenance |
| `model_selection.json` | Validation scores for all initializations and baseline candidates |
| `trials/seed*/history.json` | Per-epoch field history; v2 links the separate absorption selection |
| `dataset_manifest.json` | Audited input hashes, physical settings and seed split |
| `test_metrics.csv`, `test_metrics.json` | Raw, constrained and native ES errors against the teacher |
| `test/*/predictions.npz` | Internal test fields and predicted photon reservoirs |
| `test_slices.png` | Native ES, teacher, emulator and residual slices |
| `test_power_ratios.png` | 21-cm power relative to the teacher |
| `test_errors.png` | Internal test field and mass-ionized-fraction errors |
| `external_test/summary.json` | Separate external-test results, when requested |
| `external_test/test_metrics.json` | Per-snapshot external-test errors, without model selection |

21-cm brightness is reconstructed from the original cosmological parameters
using the explicit `BrightnessTemperatureBox.c` formula for `USE_TS_FLUCT=False`.
The reconstruction is checked against saved native and teacher brightness at
float32 rounding tolerance. Power ratios and cross-correlations with zero
reference power are undefined, not counted as perfect agreement.
The summary separately aggregates active snapshots with teacher mass-weighted
neutral fraction between 0.05 and 0.95. Per-snapshot reports also quantify the
raw-to-constrained field change and absorption/unabsorbed-count prediction errors.
Fixed phase summaries use the teacher's mass-weighted neutral fraction `x`:
early `x > 0.8`, middle `0.2 < x <= 0.8`, late `1e-10 < x <= 0.2`, and fully
ionized `x <= 1e-10`. Each phase records its snapshots and mean errors,
mean/maximum absolute ionization
bias, and worst snapshot MAE/RMSE. Absolute biases prevent cancellation between
overionized and underionized epochs; fully ionized fields do not dilute the late
phase. Empty phase metrics are undefined.

The learned scope is the teacher's scope: frozen coeval density, initially
neutral gas, cumulative native n_ion, no recombinations, thermal evolution,
photoheating feedback or independent helium chemistry. Successful fitting does
not add these processes or establish a continuum solution. A fresh external
test measures new initial conditions at the sampled grid and physical parameters.
Repeating evaluation on inspected seed 1008 measures regression behavior.
Neither tests new cosmologies, source prescriptions or resolutions.

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
