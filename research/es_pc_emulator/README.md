# Native ES to C2-Ray field correction

See the [2026-10-27 residual validation report](RESIDUAL_VALIDATION.md) for
reproduced failures, controlled physical comparisons, held-out neural accuracy
and remaining limits. The labels are not yet convergence-certified.

This experiment pairs the native three-dimensional ES neutral-fraction field
and its largest crossing radius with C2-Ray hydrogen neutral-fraction labels.
Its purpose is to learn the error of an existing fast model, rather than to
learn a radiative-transfer map without an ES prior.

The new `STORE_ES_RADIUS` option is **off by default**. Enabling it records a
dedicated `es_crossing_radius` diagnostic at the first crossing in the native
largest-to-smallest radius scan. It does not reinterpret `mean_free_path`,
which is not allocated by the restricted no-recombination source workflow.
The ordinary native ionization, source and feedback equations are unchanged.
Dataset preparation rebuilds a fresh native runtime with this diagnostic;
the older runtime used by already running simulations is not modified.

## What is learned

The native ES inputs are genuine native outputs. However, those coevals use
the native cumulative `n_ion` proxy and evolving native gas density, whereas
the RT teacher integrates explicitly prescribed source rates and freezes the
density contrast at its reference redshift. Both come from the same initial
conditions and source prescription, and these differences are recorded. This
is **ES-to-C2-Ray emulation**, not an experiment attributing every difference
to photon non-conservation alone. C2-Ray also evolves H/He chemistry and
temperature; this first emulator predicts hydrogen fractions only.

Every sample provides `xhi_es`, `radius_es_cMpc`, reference-gas `density`,
integrated input-source `photons`, and target `xhi_c2ray`. Lengths are cMpc;
density and photons retain the history's compatible mean-nuclei-per-cell
normalization. RT labels are conservatively restricted to the original source
grid with nuclei weights. The code never silently interpolates label times.

There are three comparable input arms:

| Arm | Inference inputs |
| --- | --- |
| `es` | Native ES xHI, crossing-radius fields, redshift, cell width |
| `es-density` | Above plus reference-gas density |
| `es-density-source` | Above plus cumulative emitted photons and photons/atom |

All arms use directional periodic 3-by-3-by-3 patches and Gaussian context
at fixed comoving widths 6.25, 12.5, 25 and 50 cMpc. This is a portable NumPy
MLP baseline with two 32-node hidden layers, not a full 3D CNN. Physical widths
and cell width remain explicit when grid size changes. Targets, absorbed
photons and teacher-derived budget corrections are never inference features.

The default model predicts a bounded residual around native ES xHII, initialized
at zero correction. It can create or remove ionization, including in cells
where ES is exactly neutral or ionized. `--output-mode direct` is a separate
control. Fitting standardization uses training samples only. All ablation arms
sample identical cells. Independent neural initializations and input arms are
selected by weighted validation MSE; held-out test data never select a model.

Smoothed-feature standard deviations are floored by the training standard
deviation of the corresponding unsmoothed central-cell channel. This prevents
almost constant whole-box contexts from amplifying a change between seeds by
thousands of standard deviations. All statistics use training data only.
`--standardization independent` reproduces the original per-column scaling
for a controlled comparison; the default is `channel`. The fitted mean/scale
remain part of the saved model and are reused unchanged during inference.

## Generate a dataset on Gadi

For an end-to-end **exploratory accuracy evaluation**, prepare the dataset and
submit training with an `afterok` dependency in one command:

```bash
bash research/es_pc_emulator/gadi_validate.sh
```

This uses `validation_dataset.json`: 50 cMpc, 16-cubed source/RT/output grids,
z=20 to 5, 30 source intervals, 60 RT steps, and angular order 8 (384 rays
per source), with two training seeds and independent validation/test seeds.
It is a lower-cost accuracy experiment; `--config FILE` overrides these settings.
The larger `default_dataset.json` retains RT refinement 3/angular order 32.
It first generates and audits the ES/RT residuals, then fits all three input
arms. Training explicitly permits the still-uncertified teacher; this command
does not certify spatial/time/angular convergence. The training job is held
until the dataset job succeeds. Both job IDs and the exact run directory are
printed. Run directories start with `es_pc_dataset_`.

A new initial-state check compares the first native ES field with the uniform
RT initial xHII, weighted by the teacher gas. The default maximum MAE is 0.001.
Starting at z=8 when ES is already appreciably ionized fails **before RT runs**:
the RT solver otherwise starts nearly neutral with no earlier source history.
Use an earlier start for a usable dataset. `initial_state.allow_mismatch=true`
and training `--allow-initial-state-mismatch` are independent, explicit opt-ins
for reproducing the old diagnostic failure, not substitutes for an earlier start.

From a committed `Photon_conserving` checkout:

```bash
bash research/es_pc_emulator/gadi_prepare.sh --submit
```

Preparation makes an isolated native build, freezes the source commit, upstream
C2-Ray sources, runtime/compiler hashes and explicit configuration. The PBS
job generates the prescribed source histories, runs actual native C2-Ray smoke
and CPU-consistency checks, evolves the teacher, and exports audited pairs.
It uses the existing runtime modules/dependencies as a starting point and never
installs packages on a compute node. It requests 32 CPUs, 128 GB and 48 hours
in `normal`. These are requested resources, not a measured runtime forecast.

The default configuration uses 50 cMpc, a 16-cubed source/output mesh, native
IC DIM=48, redshift 20 to 5 with 30 source intervals, density reference z=8,
RT refinement 3, two RT substeps and angular order 32 (6144 rays per source).
The independent seed split is:

| Split | Initial-condition seeds |
| --- | --- |
| Train | 1234, 2345 |
| Validation | 3456 |
| Test | 4567 |

All times and geometries sharing a seed or IC fingerprint must stay in one
split. The existing four C2-Ray convergence jobs share one history and cannot
be reused as four independent realizations. Those old histories also lack the
new ES radius diagnostic, so the dataset generator creates new sidecars.

Use `--config /absolute/config.json` to override the teacher geometry and seed
lists. Unknown keys, duplicate seeds and missing split coverage fail before
science generation. The full defaults are in `default_dataset.json`.

After an interrupted dataset job has stopped, resubmit its same prepared PBS
file with the original resource request. It automatically uses `--resume`.
Completed seeds are rehashed and reaudited before reuse; the interrupted seed
starts a new attempt directory, preserving the partial output. Configuration,
source, runtime or completed-output changes reject continuation. An MPI/chemistry
interval is not checkpointed midway, so an incomplete seed is recomputed.
After a hard scheduler kill, a stale `RUN_DIR/.running` lock may remain. Confirm
that the preceding PBS job has stopped before removing only that lock and
resubmitting; never remove the lock of an active job.

Results include per-seed source histories, native ES sidecars, native C2-Ray
binary states and transport diagnostics, per-snapshot pairs, and
`results/dataset_manifest.json`. Completion requires independent integrity,
field bounds, primary-absorption/outflow accounting and source/IC matching.
`COMPLETED` certifies successful execution and audits, not spatial, angular or
time convergence. The four controlled convergence cases and a separate angular
refinement remain necessary before interpreting these labels as science-grade
truth. The emitted spectra's measured normalization is retained without hidden
rescaling. Primary absorption, secondary ionization and boundary outflow are
not conflated.

## Train after the dataset completes

The dataset preparation prints `Run directory`. Set that exact path below:

```bash
DATASET_RUN=/g/data/qp00/sz4640/21cmfast_runs/es_c2_dataset_REPLACE_ME
qsub -v DATASET_RUN="$DATASET_RUN" \
  "$DATASET_RUN/source/research/es_pc_emulator/gadi_train.pbs"
```

This trains all three input arms and four model seeds in the dataset's frozen
runtime. The separate training request is 8 CPUs, 32 GB and four hours on
`normal`. Results go to a fresh directory in `DATASET_RUN/training_runs`.
Check `model_selection.json`, independent-test field/front/island errors,
physical saturated-spin 21-cm power spectra, Fourier cross-correlations and
baseline ES metrics. Full test prediction cubes and fitted models are saved.
`accuracy_report.json` gives the original ES MAE, neural MAE, fractional
improvement and selected epoch for every arm. Epoch zero means validation
selected the unchanged ES baseline. The CLI prints these errors before the
full JSON report; a completed fit does not by itself mean accuracy improved.

The dataset job writes `audit_path.txt`, pointing to `summary.json` and
`samples.csv` for **all** paired redshifts. They include mean and worst-snapshot
errors, global ionized-stock bias, and the exact decomposition of MSE into
squared mean bias plus centred spatial variance. Source-history and density
diagnostics are also retained. This decomposition does not change any labels.
To inspect an existing paired dataset without fitting:

```bash
python -m research.es_pc_emulator.audit --dataset /dataset/dataset_manifest.json \
  --output /fresh/residual_audit --history /seed/native_history.npz \
  --es-sidecar /seed/native_es.npz
```

Repeat the history/sidecar options for each seed. These files are checked
against the pairing hashes. Without them, unavailable diagnostics remain unknown.

The trainer refuses uncertified teacher convergence by default. The current
generator deliberately marks its labels `convergence_certified=false`: primary
photon closure is not a spatial/temporal convergence certificate. To test the
learning pipeline before the independent convergence studies finish, an
explicit exploratory opt-in is available:

```bash
qsub -v DATASET_RUN="$DATASET_RUN",ALLOW_UNCONVERGED_TEACHER=1 \
  "$DATASET_RUN/source/research/es_pc_emulator/gadi_train.pbs"
```

Such models and their summaries retain the uncertified label status. This
opt-in is not a statement that convergence has passed.

For local or custom runs:

```bash
python -m research.es_pc_emulator.generate \
  --config research/es_pc_emulator/default_dataset.json \
  --output /absolute/fresh/dataset --upstream /absolute/pinned/C2-Ray --workers 32
python -m research.es_pc_emulator.train \
  --dataset /absolute/fresh/dataset/dataset_manifest.json \
  --output /absolute/fresh/training --feature-sets es es-density es-density-source
```

The model is a coeval field emulator, not an autoregressive radiation solver.
No global one-photon-per-HII projection is applied: it is physically invalid
for the H/He, recombination and secondary-ionization teacher. Ionized-stock
errors relative to C2-Ray are reported under that name. Strict local or
per-source photon conservation of neural predictions is not claimed.

## Validation

The causal history experiment is reproducible with the same native source
histories, including a reset that removes only the pre-final-interval RT state:

```bash
python -m research.es_pc_emulator.validate_history warm --output /fresh/check --upstream /pinned/C2-Ray --workers 8
python -m research.es_pc_emulator.validate_history reset --output /fresh/check --upstream /pinned/C2-Ray --workers 8
python -m research.es_pc_emulator.validate_history original-late --output /fresh/check --upstream /pinned/C2-Ray --workers 8
python -m research.es_pc_emulator.validate_history time2 --output /fresh/check --upstream /pinned/C2-Ray --workers 8
python -m research.es_pc_emulator.validate_history report --output /fresh/check --upstream /pinned/C2-Ray --workers 8
```

These are computational experiments; run them on a compute allocation. `warm`
uses three independent seeds, z=20 to 7.9, eight source intervals and 96 rays
per source. `original-late` reproduces the old z=8 to 7.9 test; `reset` retains
exactly the warm run's last source-rate segment and gas, starting RT afresh.
`time2` doubles RT substeps at fixed input history. Source hashes are recorded.

```bash
python -m pytest -q research/es_pc_emulator --confcutdir=research/es_pc_emulator
python -m pytest -q research/pc_evolution research/c2ray_teacher research/pc_field_emulator \
  --confcutdir=research
bash -n research/es_pc_emulator/gadi_prepare.sh
bash -n research/es_pc_emulator/gadi_dataset.pbs
bash -n research/es_pc_emulator/gadi_train.pbs
```

Tests cover source/clock/grid pairing, conservative projection, tampered
payloads, leakage between independent seeds, directional inference features,
input ablations, held-out learning, model serialization, workflow failures and
Gadi preparation contracts. The real native integration run is recorded
separately; synthetic tests alone are not a native transport validation.

## Small physical-control and posterior pilot

`physical_pilot.py` varies escaped photon amplitude, emission timing, source
concentration, gas-density contrast, blackbody temperature and recombination
clumping. It evolves actual native RT from z=20 to 7.9, using a 50 cMpc box,
8-cubed source/analysis grid and a baseline 24-cubed transport grid. Its
hydrogen-only budget-ES prior is a separate approximation, not native 21cmFAST.
The corner plots use a held-out noisy mock, a conditional fixed-IC likelihood
and a separately calibrated neural-summary error. They are diagnostics of
parameter degeneracy, not evidence of a universally applicable physical
posterior or a one-percent field model. Numerical refinement differences and
their heuristic posterior stress comparison are reported separately.

The optional [`physical_pilot.pbs`](physical_pilot.pbs) reproduces the reduced
80-case experiment: 13 physical controls, three numerical refinements, 32
training histories, 16 validation histories, eight fixed-IC test histories and
eight external-IC histories (four for each of seeds 222 and 333). The frozen
design contains 128 planned cases; `pilot_selection.json` records the exact
80 executed by this wrapper. This job does not submit production training.

Supply a clean committed checkout, the pinned upstream C2-Ray checkout, and an
existing Python >=3.12 environment with the native 21cmFAST extension, NumPy,
SciPy, scikit-learn and Matplotlib. `runtime.env` must define the `PC_MODULES`
array for that environment and provide GCC, FFTW, GSL and OpenMPI. The job
records package versions and freezes a detached source worktree; it does not
install packages or modify the supplied Python environment. Set the project
and storage permissions appropriate to your allocation:

```bash
export PILOT_REPO=/absolute/path/to/21cmFAST
export PILOT_PYTHON=/absolute/path/to/native/venv/bin/python
export PILOT_UPSTREAM=/absolute/path/to/C2-Ray3Dm1D_Helium
export PILOT_RUNTIME_ENV=/absolute/path/to/native/runtime.env
export PILOT_OUTPUT=/absolute/path/to/fresh/physical_pilot
export PILOT_STORAGE="gdata/${PROJECT:?Set your NCI project}+scratch/${PROJECT}"

qsub -P "$PROJECT" -l "storage=$PILOT_STORAGE" \
  -v PILOT_REPO,PILOT_PYTHON,PILOT_UPSTREAM,PILOT_OUTPUT,PILOT_RUNTIME_ENV \
  "$PILOT_REPO/research/es_pc_emulator/physical_pilot.pbs"
```

The preset requests eight CPUs, 32 GB and a 12-hour walltime limit; this is not
a completion-time prediction. RT cases run sequentially with all allocated
CPUs to avoid overlapping MPI core bindings. Afterwards, up to four field
ablation arms run concurrently with single-thread BLAS, followed by mock
inference. Inspect `sensitivity/sensitivity_summary.json`,
`field_training/summary.json`, and `inference/summary.json`; the latter contains
held-out errors, chain diagnostics, injected-truth recovery and posterior
predictive checks. `COMPLETED` means the workflow finished, not that numerical
convergence or the physical posterior passed validation.

For three additional direct RT checks in the posterior region, run the
following in a compute allocation after inference has finished, from the
frozen source checkout. The points are the power-only posterior median and
highest-likelihood retained draw, plus the highest-likelihood draw with the
global neutral history included. They do not enter training, calibration or
the original test set:

```bash
"$PILOT_PYTHON" -m research.es_pc_emulator.physical_pilot_predictive \
  --root "$PILOT_OUTPUT/native" --inference "$PILOT_OUTPUT/inference" \
  --workers "${PBS_NCPUS:-8}" --parallel 1
```

`inference/native_predictive_checks.json` compares saved neural predictions
with the new RT runs and records changes in the same approximate likelihood.
Its covariance is deliberately kept fixed to isolate forward-mean errors;
these three checks do not recalibrate the posterior or establish numerical
convergence of the teacher. When checking a different frozen inference, supply
a new `--case-prefix` (for example `predictive_corrected_`) to retain and
distinguish the earlier native checks; changed coordinates cannot reuse an
existing case identifier.

## Small history, chemistry, spatial-context and P21 experiment

`small_pilot_training.py` trains six CPU-sized models on the physical pilot:

| Arm | Change from the preceding field arm |
| --- | --- |
| `local` | ES residual, 3-cell local context, existing physical inputs plus Gaussian/global summaries |
| `history` | Three causal source intervals: emitted photons, endpoint emissivity, redshift and availability mask |
| `thermo` | Auxiliary supervision of log10 gas temperature and the HeI/HeII/HeIII simplex |
| `spatial` | Periodic 3-D convolutions with nominal 15-cell receptive field |
| `p21` | Add full-cube differentiable log-P21 loss to field and chemistry losses |
| `direct_p21` | Separate six-control-parameter MLP predicting P21; no ionization field output |

RT state is never an inference input. The field model retains the current ES
prior, density, cumulative photons, emissivity, emission-time moment, source
spectral temperature, clumping factor, redshift and cell size. Historical
intervals only use present/past sources. The direct model takes the six controls
in `physical_pilot.PARAMETER_NAMES`; it does not condition on the IC realization.
Changing the spatial network also changes parameter count, so this is a practical
progressive comparison rather than a capacity-matched causal experiment.

On Gadi, from a clean checkout of `Photon_conserving`:

```bash
bash research/es_pc_emulator/small_pilot_prepare.sh --submit
```

Defaults are 32 training, 8 validation, 8 test and 4 external-IC histories,
80 epochs with early stopping, one network seed, and 16 CPUs/64GB/8h. The
external cases cover both independent IC seeds. The native pilot remains
50 cMpc, 8^3 output cells and 24^3 RT cells; this run compares models, not
teacher resolution. It does not launch MCMC or convergence sweeps.

The preparation script uses the same existing native runtime default as
`gadi_prepare.sh`; override `SMALL_RUNTIME_PARENT` or `SMALL_NATIVE_PYTHON`
when necessary. It freezes a source worktree and creates a separate CPU
PyTorch environment on the login node. Native simulations and all training
run inside the PBS allocation. Existing environments/results are unchanged.

To reuse an already generated physical-pilot dataset and skip RT entirely:

```bash
bash research/es_pc_emulator/small_pilot_prepare.sh \
  --manifest /absolute/path/to/dataset_manifest.json --submit
```

To reduce the trial further, export `SMALL_TRAIN_CASES=16`,
`SMALL_VALIDATION_CASES=4`, `SMALL_TEST_CASES=4`, `SMALL_EXTERNAL_CASES=4`
and `SMALL_EPOCHS=40` before preparing. `--help` lists allocation overrides.
The equivalent training-only command, in an environment with CPU PyTorch,
NumPy, SciPy and Matplotlib, is:

```bash
python -m research.es_pc_emulator.small_pilot_training \
  --manifest /absolute/path/to/dataset_manifest.json \
  --output /absolute/fresh/training --threads 16 --epochs 80
```

Outputs are `training/summary.json`, `training/comparison.png`, individual
`model.pt` checkpoints, training histories and per-case fields/spectra.
`predict_checkpoint` reloads checkpoints with their saved normalization.
Test/external histories never enter fitting or early stopping. Primary reported
errors use the final three snapshots and 0.1 <= k <= 0.6 kNyquist; all spectra
are also retained. Spectra are in mK^2 cMpc^3 under saturated spin temperature
and no RSD. The additive 1e-8 power floor is only a loss/output-transform
parameter; reported relative errors use unmodified nonzero reference powers.
Teacher convergence remains explicitly uncertified in every checkpoint.

## Test which P21 components need supervision

The component experiment reuses a completed `small_nn_*` run, including its
exact cases, split, hashes and CPU training environment. From the updated
`Photon_conserving` checkout on Gadi:

```bash
bash research/es_pc_emulator/component_prepare.sh --submit
```

The script selects the most recently completed `small_nn_*` under
`/g/data/qp00/sz4640/21cmfast_runs` and prints its path. Select a specific run with
`--parent /absolute/path/to/small_nn_run`. No native simulations, package installs,
or changes to previous results are made. A new frozen checkout and output
directory are created. Defaults are two network seeds (11,23), inherited
epochs/patience (normally 80/15), and 16 CPUs/64GB/8h.

All seven arms retain the same small network, history inputs and thermal/helium
auxiliary supervision. Only the spectral objectives change:

| Arm | Added objective |
| --- | --- |
| `thermo` | None; field and thermochemical baseline |
| `thermo_p21` | Total P21 |
| `thermo_pxx` | Neutral-fraction auto contribution |
| `thermo_cross` | Signed density-neutral cross contribution |
| `thermo_nonlinear` | Product auto and its two cross contributions |
| `thermo_components` | All six signed contributions |
| `thermo_components_p21` | All six contributions and total P21 |

For `T21=A rho xHI`, define centered fields `D=A*xbar*(rho-rhobar)`,
`X=A*rhobar*(xHI-xbar)`, and `N=A*centered((rho-rhobar)*(xHI-xbar))`.
The exact six terms are `PDD`, `PXX`, `2PDX`, `PNN`, `2PDN`, `2PXN`.
These are weighted contributions to brightness power, not six independent
physical control parameters. In particular the cross terms retain their signs.

Component losses compare `asinh(term/scale)` with fixed per-term/per-mode RMS
scales derived only from the primary training snapshots. Total P21 retains the
log-power loss. New spectral objectives use the final three snapshots and
0.1 <= k <= 0.6 kNyquist, as does the common validation epoch selection.
The default spectral/auxiliary weights are 0.01. Set `COMPONENT_WEIGHT`,
`COMPONENT_POWER_WEIGHT`, `COMPONENT_EPOCHS`, `COMPONENT_PATIENCE` or
`COMPONENT_SEEDS` before preparing to change them; seeds are comma-separated.

`training/component_comparison.png` compares total and individual component
errors. `training/component_report.json` contains paired differences from
`thermo` for each entire simulation and each seed, plus reconstruction closure.
The component errors are reported as `(predicted_term-true_term)/true_P21`,
so their signed sum reconstructs the total fractional power error even when
individual cross spectra are negative or cross zero. Original summary,
checkpoints, fields and per-snapshot spectra are also retained. These runs test
learning objectives on the existing uncertified teacher; they do not certify
physical photon conservation or numerical convergence.

## Small loss-weight sweep and learning curves

After the component experiment, run:

```bash
git switch Photon_conserving
git fetch origin Photon_conserving
git merge --ff-only FETCH_HEAD
bash research/es_pc_emulator/loss_sweep_prepare.sh --submit
```

This reuses the latest completed `component_nn_*` run (fallback `small_nn_*`),
including its exact case list, split, checksums and Python environment. Select
another completed run with `--parent /absolute/run/path`. Preparation prints
and records the parent and creates a frozen checkout in a new `loss_sweep_*`
directory. Training uses existing labels; no new RT simulations run.

The seven settings retain the same small thermo network and input features:

| Settings | Component weight | Total P21 weight |
| --- | --- | --- |
| Thermo baseline | 0 | 0 |
| Six-component scan (3) | 0.003, 0.01, 0.03 | 0 |
| Joint scan (3) | 0.01 | 0.003, 0.01, 0.03 |

These are one-factor scans, not a full two-dimensional grid. The objective is
`field_MSE + auxiliary_weight*chemistry_MSE + component_weight*component_loss
+ power_weight*log_P21_MSE`. The auxiliary weight and other network/training
parameters are inherited. Defaults: seeds 11 and 23, at most 120 epochs per fit,
early-stopping patience 30, diagnostics every 5 epochs, 16 CPUs, 64GB and 12h.
There are 14 fits, run sequentially inside one PBS job.

Every candidate retains its best epoch using the same validation log-P21 MSE.
The setting is selected by averaging that validation score across both network
seeds. Only after this choice is saved are its models and the paired thermo
baseline evaluated on test/external cases. Seed averages are averages of
individual-model errors, not ensemble predictions. These test sets were used
in previous exploratory comparisons; this is not a new independent test set.

Diagnostics measure **the same primary snapshots and k bins** on training and
validation data: log-P21 MSE, absolute relative P21 error, field MAE/MSE,
chemistry MSE and unweighted six-component loss. They are recorded at epoch 1,
every diagnostic interval, the final/stopping epoch and the restored best
checkpoint. This makes train/validation curves comparable even when different
weighted objectives drive optimization. Undefined relative-power bins are
counted and excluded. Teacher convergence remains uncertified.

Important outputs under the new run's `training/` directory:

- `sweep_report.json`: weights, all seed/epoch diagnostics, validation selection,
  selected held-out results and whole-simulation paired comparisons.
- `learning_curves.png`: common-metric train/validation histories.
- `loss_sweep.png`: weight comparison using train/validation errors.
- `selected_summary.json`: selected models and baseline test/external errors,
  including six spectral contributions.
- `selection.json`: setting selected before held-out evaluation.
- `trials/<setting>/seed<seed>/attempt_*/`: checkpoint, full training history,
  train/validation spectra and fields.
- `progress.json`: current fit; `sweep_progress.json`: completed settings.

Set `LOSS_SWEEP_EPOCHS`, `LOSS_SWEEP_PATIENCE`, `LOSS_SWEEP_DIAGNOSTIC_INTERVAL`,
`LOSS_SWEEP_SEEDS` (comma-separated), `LOSS_SWEEP_COMPONENT_WEIGHTS`,
`LOSS_SWEEP_POWER_WEIGHTS` and `LOSS_SWEEP_ANCHOR_COMPONENT_WEIGHT` before
preparation to change the scan. The anchor must appear in component weights.
Allocation overrides are `LOSS_SWEEP_WORKERS`, `LOSS_SWEEP_MEM` and
`LOSS_SWEEP_WALLTIME`; `--help` lists parent/root/project overrides.

After a walltime interruption, submit the prepared run again with
`bash /absolute/loss_sweep_run/resubmit.sh`. Completed fits are reused; only
an interrupted fit restarts, with its old files retained in a separate attempt.
Configuration, manifest and source hashes must match for resumption. An
exclusive PBS lock prevents duplicate jobs writing to the same run.

## Train from the initial state through the end of reionization

From the updated branch on Gadi:

```bash
bash research/es_pc_emulator/full_history_prepare.sh --submit
```

The parent is the latest completed loss-weight sweep, component experiment or
small pilot, in that order. `--parent /absolute/run` selects one explicitly.
The script reuses both Python environments and the exact previous parameter
points, case IDs and splits (normally 32 train / 8 validation / 8 test / 4 external).
It freezes a new worktree and archives the same pinned C2-Ray upstream source.

This job **generates new native histories**: by default 40 intervals from z=20 to z=4,
with 41 saved snapshots including the declared initial state. It retains the
50 cMpc box, 8^3 source/label grid, 24^3 RT grid, angular order 2, source prescription,
and frozen z=8 density contrast of the physical pilot. The requested final
redshift is a domain choice; completion is checked from the resulting fields.
The history-tilt transform is normalized over the full new time interval, so
all endpoints are regenerated rather than appended to old labels.

Snapshot 0 records the RT input's exact initial H/He fractions and temperature,
zero integrated photons, zero mean emission time, initial endpoint emissivity,
and a neutral ES prior with zero first-crossing radius. All saved arrays and
spectra have the same redshift/time axes. This initial label is a prescribed
condition, not an RT-evolved output.

Before training, `coverage.json` checks every selected case:

- Every requested endpoint is present, with time starting at zero.
- Initial H/He/T match the declared RT initial state, and the initial neutral
  fraction is at least 0.99.
- At least one saved snapshot has 0.1 <= mean xHI <= 0.9.
- Both final volume- and mass-weighted xHI are <= 0.01.

A failure retains per-case reasons and stops before training. It does not drop
slow-reionizing cases or reinterpret z=4 as a completed history. Temporal coverage
is separate from the teacher's still-uncertified numerical convergence.

Training uses `--snapshot-scope all` with the existing thermo residual network,
three-interval causal source features, six-component weight 0.01 and total-P21
weight 0.01. All snapshots enter field, chemistry and spectral supervision,
train-only component normalization, validation checkpoint selection and reported
errors. Redshifts have equal snapshot weighting; the k range remains
0.1 <= k <= 0.6 kNyquist. This is snapshot correction over a full trajectory, not a
recurrent model feeding predictions into the next snapshot.

Both seeds 11 and 23 run for at most 120 epochs with patience 30; other optimization
settings are inherited. Errors are reported for each redshift and in neutral
fraction bins [0, .01, .1, .5, .9, .99, 1], alongside the aggregate. These stage bins use
teacher xHI only for evaluation. Undefined relative-power bins are counted,
and absolute power errors are retained, including after the signal vanishes.

Default allocation: 32 CPUs, 128 GB, 24 h; two native cases run concurrently with 16
threads each. Training runs after collection and successful coverage validation.
No new packages are installed. Set `FULL_Z_END`, `FULL_INTERVALS`, `FULL_EPOCHS`,
`FULL_SEEDS`, `FULL_WORKERS`, `FULL_NATIVE_PARALLEL`, `FULL_MEM` or `FULL_WALLTIME`
before preparation to change the run; `--help` lists other overrides.

Outputs in the printed `full_history_*` run directory:

- `coverage.json`: actual start/end fractions and stage coverage for every case.
- `native/dataset_manifest.json`: regenerated complete-history dataset.
- `training/full_history_report.json`: training diagnostics, global and stage
  errors, coverage and provenance in one uploadable report.
- `training/evolution_errors.png`: P21 errors throughout redshift evolution.
- `training/learning_curves.png`: common-metric train/validation curves.
- `training/summary.json`: full six-component distributions by split, snapshot
  and neutral-fraction stage; individual checkpoints and predictions remain.

If interrupted, `bash /absolute/run/resubmit.sh` reuses completed native cases.
An incomplete training directory is archived and training restarts; completed
training is reused to regenerate reports. Previous runs remain unchanged.
