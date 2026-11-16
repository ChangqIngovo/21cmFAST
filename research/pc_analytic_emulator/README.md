# Emulator of the analytic coeval photon-conserving teacher

The [mixed-geometry workflow](refinement_20261117/reproduction/README.md) adds
optional feature-normalization safeguards, an explicit box-size descriptor and
geometry-aware dataset validation. It supports native generation or training
from supplied teacher archives, followed by immutable validation selection and
separate blind evaluation. Model weights and experiment outputs are supplied
separately from this source code.

The complete source map and execution sequence are in
[PHOTON_CONSERVATION.md](../PHOTON_CONSERVATION.md); independently generated
simulation results are in [the validation report](../pc_analytic/VALIDATION_20261116.md).
Controlled budget replacements, Fourier attribution, ranking limits and late
teacher refinement are in [the error-source report](ERROR_SOURCES_20261116.md).

The earlier [improvement report](COMPLETION_20261116.md) adds 164 training
snapshots, 28 validation snapshots and 44 frozen-model blind/transfer checks.
Its selected model uses 88 native features and a tabulated absorption budget.
Same-geometry active field MAE is 0.02282 and low-k power error is 9.89%; grid/box
transfer fails. Use those weights only within the documented L50/N16 physical
configuration. The [reproduction workflow](completion_20261116/reproduction/README.md)
and [Gadi launcher](completion_20261116/reproduction/GADI.md) run generation,
paired training, validation-only selection and independent evaluation.

This workflow trains on completed `pc_analytic.campaign` outputs. The labels are
our **analytic-column-characteristic** neutral fields, not SCRIPT or C²-Ray.
It reads the existing teacher archive and does not rerun the simulations.

## First training run on Gadi

From a clean checkout of `matched-photon-conservation`:

```bash
(
    set -euo pipefail
    cd "/g/data/${PROJECT}/${USER}/21cmFAST"
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation
    module load python3/3.12.1

    unset PC_TEACHER_RUN PC_EXTERNAL_TEST_ROOT PC_TRAINING_METHOD
    PC_RUN_ROOT="/g/data/${PROJECT}/${USER}/21cmfast_runs" \
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
    cd "/g/data/${PROJECT}/${USER}/21cmFAST"
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation
    module load python3/3.12.1

    unset PC_TEACHER_RUN PC_EXTERNAL_TEST_ROOT
    PC_RUN_ROOT="/g/data/${PROJECT}/${USER}/21cmfast_runs" \
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
excluded from fitting and checkpoint selection. V2 uses seed 1007 for validation;
the residual workflow below reserves seeds 1006 and 1007.

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

## Residual refinement with two validation realizations

`--residual` selects `warm_started_v3`. It first refits the legacy models using
the new training split, compares absorption estimators and projection methods
while keeping each fitted field fixed, and then learns a residual around the
chosen complete deployed prediction. The fixed base includes its own budget
and projection. A zero residual with the same budget therefore starts from the
base field, rather than from uncorrected native ES.

```bash
(
    set -euo pipefail
    cd "/g/data/${PROJECT}/${USER}/21cmFAST"
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation
    module load python3/3.12.1

    unset PC_TEACHER_RUN PC_EXTERNAL_TEST_ROOT PC_TRAINING_METHOD
    PC_RUN_ROOT="/g/data/${PROJECT}/${USER}/21cmfast_runs" \
    PC_TRAIN_SEEDS=1001,1002,1003,1004,1005 \
    PC_VALIDATION_SEEDS=1006,1007 PC_TEST_SEEDS=1008 \
    PC_MODEL_SEEDS=11,23,47,83 \
    PC_WORKERS=4 PC_THREADS=4 PC_NCPUS=16 \
    PC_EMULATOR_EPOCHS=200 PC_RESIDUAL_EPOCHS=80 \
    PC_RESIDUAL_LEARNING_RATE=0.0003 PC_SAMPLES_PER_SNAPSHOT=8192 \
    PC_MEM=64GB PC_WALLTIME=04:00:00 PC_QUEUE=normal \
    bash research/pc_analytic_emulator/gadi_prepare.sh \
        --latest-teacher --residual --submit
)
```

This reuses the same 56 teacher snapshots as 35 training, 14 validation and
7 regression snapshots. Every base network, budget predictor and normalization
is refitted on the new training split. Previous model checkpoints trained on
seed 1006 are not imported, because that seed now belongs to validation.
Explicit `PC_TRAIN_SEEDS`, `PC_VALIDATION_SEEDS` and `PC_TEST_SEEDS` settings can
select other disjoint groups; all redshifts of a seed remain together.
Use `PC_REGRESSION_TEST_SEEDS` to record previously inspected test seeds, or
`regression_test_seeds` in the JSON configuration. The frozen manifest reports
their intersection with the requested test split; labels outside that list
remain excluded from fitting and selection, without implying historical novelty.
Assess the change against the freshly refitted legacy candidates within this
same run. The earlier 42/7/7 run used different training and validation samples,
so its validation score is not directly comparable to the new 35/14/7 score.

`PC_RESIDUAL_EPOCHS` sets the residual epoch limit (80), and
`PC_RESIDUAL_LEARNING_RATE` its learning rate (0.0003). The original base and
budget fits still use `PC_EMULATOR_EPOCHS` (200). Unmodified base candidates
remain eligible during validation selection. The residual field uses the same
physical input features and projected field/Fourier objective as v2; it does
not introduce teacher fields as inference inputs. New predictions retain the
independent unabsorbed-photon reservoir and global conservation audit.

The budget comparison retains a constant estimator, a tabulated estimator,
the original residual MLP, a direct-sigmoid MLP, and local interpolation using
either three or five training neighbours. The direct-sigmoid candidate predicts
efficiency without the flat positive-residual branch about eta=1. Its
initialization and standardization use training labels and features only;
validation still selects the complete field/budget/projection combination.
The local estimators interpolate the absorption deficit `1 - eta`, where
`eta = absorbed photons / min(source photons, atom capacity)`, using photon load,
density/source scatter and native-ES statistics. Their tables and feature
normalization use training data only. They introduce no redshift switch and
do not force complete absorption whenever the photon supply exceeds the atom
count. Validation compares each budget with the fixed field and projection;
the scalar budget fit alone does not determine the deployed model.

The standalone `predict` command accepts native `source.npz` and `native.json`
without teacher labels, checks the declared physical parameters and input hashes,
and audits the saved field's global photon ledger. See the complete workflow
above for an executable example. Predictions on a different astrophysical
parameter set require a model trained for that set; native ES features alone
do not establish such generalization.

V3 expands validation within the existing campaign; it does not create a fresh
external test. Seed 1008 retains its regression label. An additional
`--external-test-root` remains available for independent completed teacher
realizations with matching physical settings. `--improved` continues to run v2,
and neither flag changes the legacy default.

## Data split and fitting

The legacy and v2 default split is:

| Role | Simulation seeds | Snapshots |
| --- | --- | ---: |
| Training | 1001–1006 | 42 |
| Validation | 1007 | 7 |
| Previously inspected regression test | 1008 | 7 |

The v3 default moves seed 1006 from training to validation, giving 35/14/7
snapshots. All models are refitted for that split.

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

### Frozen-base residual inference

In v3 the selected fixed base is evaluated on the native inputs before the
residual network. Its parameters are frozen throughout residual fitting.
The added network predicts a correction to the base's projected xHII field;
capacity-weighted projection then enforces the independently predicted budget.
The zero-residual checkpoint and unrefined candidates remain available for
validation selection. Training and inference use the same base prediction.

Schema 3 stores the complete base recursively in `model/base_model/`, alongside
the residual `model/field_model.npz` and its budget estimator. Inference therefore
does not need an earlier training directory or any teacher arrays. Schema 1 and
2 files remain readable with their existing predictions.

### Field features, normalization and geometry

`feature_set="legacy"` retains the original 24 descriptors.
`"multiscale_local"` adds 64 native-input spatial descriptors, and
`"multiscale_geometry"` additionally records `log1p(BOX_LEN)` in cMpc, for 89
features. The existing cell-width feature and the new box-size feature distinguish
changes of grid spacing from changes of periodic volume. Schema 4 records the
feature set; every frozen base retains its own feature set and normalization.
Earlier model schemas retain their original inference behavior.

Two optional controls apply to newly fitted field networks, including both the
legacy base and residual head:

| JSON setting | Gadi environment variable | Default | Effect |
| --- | --- | --- | --- |
| `feature_set` | `PC_FEATURE_SET` | `"legacy"` | Select the native-input descriptors |
| `field_feature_scale_floor` | `PC_FIELD_FEATURE_SCALE_FLOOR` | `0.0` | Lower bound on nonconstant training feature standard deviations |
| `zero_constant_features` | `PC_ZERO_CONSTANT_FEATURES` | `false` | Initialize exact training-constant first-layer rows to zero |
| `budget_epochs` | `PC_BUDGET_EPOCHS` | `null` | Separate budget fitting limit; null uses `epochs` |
| `frame_cache_MiB` | `PC_FRAME_CACHE_MIB` | `512` | Per-worker snapshot cache limit; zero disables caching |

For exact training-constant features, zeroing also stores the exact observed
value as the mean. Their standardized training inputs and weight gradients are
then zero. Varying features use their training mean and a standard deviation
bounded below by the configured floor. These options prevent unlearned constant
coordinates and almost-constant descriptors from dominating a new prediction.
They do not change the teacher physics, photon projection, budget-head
normalization or existing saved models. No inference clipping is added.

New models record the geometries represented in their training split. A
standalone prediction checks that support; a separately frozen benchmark can
explicitly measure unsupported transfer. Robust normalization alone does not
establish accuracy at a different grid or box size.

## Results and interpretation

Within the printed `analytic_train_...` directory:

| File under `results/` | Content |
| --- | --- |
| `summary.json` | Split, selection, evaluation roles, errors and maximum global budget residual |
| `model/field_model.npz` | Fitted field network and its training normalizer |
| `model/budget_model.npz` | Selected absorption estimator: network, table or reference |
| `model/model.json` | Inference configuration and teacher provenance |
| `model/base_model/` | Self-contained frozen base when a residual candidate wins |
| `model_selection.json` | Validation scores for all initializations and baseline candidates |
| `comparison_summary.json` | V3 compact candidate scores and validation-seed summaries |
| `trials/seed*/history.json` | Per-epoch field history; v2 links the separate absorption selection |
| `dataset_manifest.json` | Audited input hashes, physical settings and seed split |
| `test_metrics.csv`, `test_metrics.json` | Raw, constrained and native ES errors against the teacher |
| `test/*/predictions.npz` | Internal test fields and predicted photon reservoirs |
| `test_slices.png` | Native ES, teacher, emulator and residual slices |
| `test_power_ratios.png` | 21-cm power relative to the teacher |
| `test_errors.png` | Internal test field and mass-ionized-fraction errors |
| `external_test/summary.json` | Separate external-test results, when requested |
| `external_test/test_metrics.json` | Per-snapshot external-test errors, without model selection |

For v3 comparisons, inspect `summary.json` and `comparison_summary.json` first.
The latter reports candidate and validation-seed scores without repeating the
large per-shell arrays retained in `model_selection.json`.

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
in v3, an already completed legacy base is retained when its residual stage
restarts. This is stage-level recovery, not optimizer-state recovery within an epoch.
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
