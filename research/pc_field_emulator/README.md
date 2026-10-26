# Train a residual field emulator on existing PC outputs

This experiment learns coeval ionization fields from completed fixed-gas PC
checkpoints. It reads the existing cumulative emitted photons and atom capacity;
it does not generate initial conditions, rerun transport, or change simulation
outputs. The training target is the checkpoint's `ionized / capacity` field on
its original output grid. Fine absorber subcells are not expanded into labels.

## Model

The default Gadi workflow trains a two-hidden-layer NumPy network (32, 32) around
the local source estimate `b = clip(photons / capacity, 0, 1)`. For a network
amplitude `a = tanh(logit)`, the bounded correction is

```text
xHII = b + (1-b)*a    if a >= 0
xHII = b + b*a        if a < 0
```

The output layer starts at zero, so the initial prediction is exactly `b`.
That initial model remains a validation candidate if training never improves
the declared loss. Corrections are disabled only for a completely photon-free
box. A source below its coarse-cell capacity can still illuminate neighboring
cells in the refined teacher; coarse overflow is not used as a transport gate.

The 19 input features contain local density/source ratios and their periodic
Gaussian averages at **6.25, 12.5, 25 and 50 cMpc**, plus the global emitted
photons per atom, redshift, cell width and a zero-capacity indicator. Physical
scales have the same meaning in both box/grid combinations. Target ionization,
absorbed photons and retained photons are not inference features.

Training samples up to 8192 uniform cells per snapshot, up to 24 snapshots per
seed/geometry history. The weighted MSE has unit base weight, adds three for
cells touching a six-neighbor `xHII=0.5` boundary, and adds one for neutral cells
when the target mass-ionized fraction is at least 0.5. A cell in both categories
therefore has weight five. Targets determine these training/validation loss
weights, never inference features. Four initializations train for up to 200
epochs; weighted validation loss alone selects the model. Unweighted MSE is
also reported.

Raw predictions and an optional global count projection are evaluated
separately. The projection enforces
`sum(capacity*xHII) = min(sum(INPUT source_emitted), sum(capacity))`.
It imposes a box-wide count, without per-source, directional or reachability
constraints. Independent coeval predictions do not advance an evolving
radiation field or encode the complete emission ordering.

## Existing data and splits

The default inputs are:

| Existing run | Box length | Output grid | Initial-condition seeds |
| --- | --- | --- | --- |
| `evolution_pc_30e8497de2_20261024T122119_qC2V` | 200 cMpc | 32³ | 1234, 2345, 3456, 4567 |
| `evolution_production_ecd4bbdb88_20261024T144219_8iMG` | 400 cMpc | 128³ | 1234 |

Both are under `/g/data/qp00/sz4640/21cmfast_runs`. The selected teacher arm is
four absorber subcells per axis, 768 directions and stride four. The 128³
training target is the original large-run output grid; it is not a 512³
reconstruction of the teacher's absorber state.

All geometries sharing an IC seed remain in one split. With these existing
inputs, seeds 1234/2345 train, 3456 validates, and 4567 tests. Consequently, the
large box contributes to training but currently has no independent large-box
test seed. Three training snapshots per geometry are reconstructed and saved
under `training_reconstruction/`; these are explicitly in-sample diagnostics.
`geometry_coverage` in the summary reports the actual coverage of each split.

The loader compares astrophysics, cosmology, gas reference redshift, count
conventions and source clocks before mixing runs. Geometry and thread settings
may differ. Checkpoint hashes, raw source/grid photon ledgers and fraction
bounds are verified before fitting. Only audited checkpoints with
`(escaped + retained) / emitted > 1e-7` while neutral gas remains are excluded;
other integrity failures stop the job. Every exclusion and its redshift/count
impact is recorded in `preflight.json`. An empty split or entirely missing
requested geometry stops training instead of silently reducing its scope.

Preparation freezes the list of durable checkpoints and their hashes in
`sample_manifest.json`. A production simulation may still be running; later
checkpoints will not enter an already prepared training job. Only recorded
committed outputs are eligible, never a partially written transport step.

## Run on Gadi

From the committed `Photon_conserving` checkout:

```bash
bash research/pc_field_emulator/gadi_prepare.sh --submit
```

This selects both existing runs above. It requests **32 CPUs, 64 GB and two
hours in the normal queue**, with four independent fits and eight BLAS threads
per fit. The walltime is an allocation. Input reading, feature construction
and output diagnostics also contribute to the job's runtime.

To choose explicit input runs, supply one or more paths:

```bash
bash research/pc_field_emulator/gadi_prepare.sh \
  /path/to/completed-small-run \
  /path/to/existing-large-run --submit
```

An explicit single path or `PC_EMULATOR_INPUT` overrides the default pair.
`PC_RUNTIME_PARENT` and `PC_PYTHON` select another existing compatible runtime;
by default the first input supplies the Python/module configuration.
`PC_EMULATOR_EPOCHS`, `PC_MEM`, `PC_WALLTIME` and `PC_QUEUE` override fitting or
allocation settings. Source, interpreter, package versions and the frozen
inventory are pinned in a fresh `field_emulator_*` run. A failed job should be
prepared in a new run directory.

Direct invocation supports the same mixed inputs:

```bash
python research/pc_field_emulator/train.py \
  --input /path/to/small-run --input /path/to/large-run \
  --output /path/to/fresh/results \
  --workers 4 --model-seeds 11 23 47 83 \
  --epochs 200 --samples-per-snapshot 8192 \
  --model-mode residual --radii-cmpc 6.25 12.5 25 50 \
  --front-weight 4 --neutral-weight 2
```

For direct invocation, set BLAS thread limits before starting Python and keep
workers times threads within the allocation. `--sample-manifest` optionally
loads a prepared frozen inventory. The Python API retains its legacy sigmoid
defaults, and schema-1 models remain loadable; the CLI/Gadi defaults select the
new residual mode.

## Inspect results

- `dataset_manifest.json`, `preflight.json`: exact input identities, physical
  compatibility, split roles, geometry coverage and excluded epochs.
- `model.npz`, `features.json`: selected model and ordered feature names;
  the saved model metadata carries the physical smoothing configuration.
- `trials/seed*/history.json`, `model_selection.json`: weighted validation
  selection, unweighted losses and the baseline epoch-zero candidate.
- `summary.json`, `test_metrics.csv`, `test/`: independent held-out metrics,
  complete prediction arrays and comparison figures.
- `training_reconstruction_metrics.csv`, `training_reconstruction/`: explicitly
  in-sample full-grid reconstructions, including the existing large box.

Metrics include mass/volume ionization, cell and front errors, neutral-island
IoU/MAE, physical 21-cm power and Fourier cross-correlation. Brightness uses the
fixed-gas, saturated-spin-temperature, no-RSD convention of the teacher.
Every spectrum uses that snapshot's own physical box length and Nyquist limit;
different grids are not silently rebinned into a common spectrum.

Inference timing includes features, network and global projection. It excludes
input/output and diagnostics. No equivalent teacher solve is timed by this
training workflow, so its timing is not a full-simulation speedup measurement.
