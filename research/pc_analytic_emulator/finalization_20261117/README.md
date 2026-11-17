# Validation-gated coeval photon-conserving inference

The analytic teacher and its native 21cmFAST photon/atom inputs are unchanged.
The added emulator controls address mixed-geometry budget prediction, unequal
training exposure and rare neutral islands. They are opt-in; saved older models
and default training configurations retain their behavior.

## Training controls

- `include_geometry_tabulated_budget: true`: add a training-only absorption
  table for each cell width/box length. This avoids averaging different grid
  resolutions at a common redshift. The original pooled table is the explicit
  fallback for unseen geometries. This candidate must still win full-field
  validation; better scalar budgets alone do not establish better fields.
- `geometry_balanced_sampling: true`: expose each training geometry equally
  to full-snapshot residual updates. Validation remains snapshot-weighted.
- `phase_balance_weight: 1.0`: give rare neutral/ionized phases more supervised
  weight, using continuous fractions and a fixed 0.01 phase-volume floor.
  Zero recovers the original loss. Teacher fields are never inference inputs.

Use the existing `research.pc_analytic_emulator.train` configuration workflow,
with these optional keys, or `refine.py` to add a residual to an already frozen
model using an audited fit cache. Select candidates on validation only. Preserve
all seeds of a realization within one split.

## Accuracy contract

`reliability.accuracy_metrics` measures xHI absolute MAE, mass-ionization bias,
and equal-shell 21-cm brightness power error below half the grid Nyquist.
The common saturated-spin brightness coefficient cancels between fields.

| Reference stage | Field MAE | Absolute mass-ionization bias | Brightness condition |
|---|---:|---:|---|
| xHI_mass > 0.05 | <=0.03 | <=0.005 | Mean shell relative power error <=0.10 |
| xHI_mass <= 0.05 | <=0.01 | <=0.002 | Low-k complex residual RMS <=0.01 of fully-neutral RMS |

Near zero reference power, fractional power error is ill-conditioned. The
late condition is an **absolute brightness** criterion, not a 1% relative
power claim. Unfloored relative errors are still recorded wherever defined;
zero-power modes remain included in the absolute residual. Uniform spurious
neutrality is additionally constrained by the mass-ionization bias.

A route is an exact `(DIM, HII_DIM, BOX_LEN, redshift)` tuple. All independent
validation seeds must pass; the study driver uses a 0.75 safety margin and at
least two seeds. Among eligible candidates it picks the lowest validation MAE.
A new realization need not obey the empirical bound, so a fresh holdout is
required after selection. An untested geometry/redshift or a failed route uses
a **fresh analytic teacher solve**, not an unchecked surrogate prediction.
Learned-only errors, route fractions and actual fallback timing must be
reported separately. This hybrid is not a uniformly accurate neural emulator.

## Freeze and evaluate

Prepare a JSON mapping safe candidate IDs to self-contained model directories.
`--settings` is the original teacher campaign configuration. Its solver controls,
physical implementation and native runtime must match model training provenance.

```bash
python -m research.pc_analytic_emulator.finalization_20261117.study validate \
  --cache /path/to/training/fit_cache \
  --candidates /path/to/candidates.json \
  --settings /path/to/teacher_config.json \
  --output /path/to/new_validation

python -m research.pc_analytic_emulator.finalization_20261117.study evaluate \
  --deployment /path/to/new_validation/deployment \
  --teachers /path/to/new_teacher_campaign \
  --output /path/to/new_holdout_results
```

The deployment packages weights, policy, physics and implementation hashes.
Do not edit its receipts or overwrite a completed experiment. Code/model drift
requires refreezing before a fresh evaluation. The holdout cannot overlap any
recorded training/validation seed.

## Native input inference

Run `python -m research.pc_analytic_emulator.hybrid --help` for the native-snapshot
interface. Input is the original receipt-backed native snapshot directory,
including `source.npz`, `native.json` and `native.h5`. No teacher fields are
required. Returned `report.json` identifies `emulator` or `teacher`, the selected
candidate, measured computation time and an independent saved-field photon
ledger. Analytic output stores radiation, escaped and unemitted inventories
separately; emulator output stores its predicted global unabsorbed reservoir.

This workflow retains frozen coeval density, initially neutral gas, cumulative
native n_ion, the recorded effective atom convention and characteristic optical
absorption. It does not add recombinations, thermal evolution, photoheating
feedback, independent helium chemistry or cosmic light-travel evolution.
Agreement with this teacher does not certify its spatial/angular convergence.
