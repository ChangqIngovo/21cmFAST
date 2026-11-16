# Photon-conserving teacher and emulator

The current implementation on `matched-photon-conservation` consists of the
analytic coeval teacher in `pc_analytic` and its supervised emulator in
`pc_analytic_emulator`. Earlier transport experiments remain in other research
directories; they are not the teacher used by this training workflow.

The [prediction-error investigation](pc_analytic_emulator/ERROR_SOURCES_20261116.md)
separates absorbed-count errors, projection-induced saturation and spatial
ranking limitations using fixed-model controls.

## Complete source entry points

| Component | Source |
| --- | --- |
| Finite-exposure analytic absorption | [pc_analytic/absorption.py](pc_analytic/absorption.py) |
| Directional characteristic transport and photon inventories | [pc_analytic/transport_kernel.cpp](pc_analytic/transport_kernel.cpp) |
| Validated Python solver interface | [pc_analytic/solver.py](pc_analytic/solver.py) |
| Native 21cmFAST simulations and parallel teacher generation | [pc_analytic/campaign.py](pc_analytic/campaign.py) |
| Native source/atom extraction | [pc_reconsidered/native_driver.py](pc_reconsidered/native_driver.py) |
| SCRIPT, angular/dose, boundary, grid and MFP diagnostics | [pc_analytic/diagnostics.py](pc_analytic/diagnostics.py) |
| Audited teacher data and seed-separated splits | [pc_analytic_emulator/data.py](pc_analytic_emulator/data.py) |
| Complete parallel training workflow | [pc_analytic_emulator/train.py](pc_analytic_emulator/train.py) |
| Fixed-field comparisons and frozen-base residual training | [pc_analytic_emulator/warmstart.py](pc_analytic_emulator/warmstart.py) |
| Field objective, gradients and bounded photon projection | [pc_analytic_emulator/fit.py](pc_analytic_emulator/fit.py), [objectives.py](pc_analytic_emulator/objectives.py), [projection.py](pc_analytic_emulator/projection.py) |
| Global absorption estimator and deployed model | [pc_analytic_emulator/budget.py](pc_analytic_emulator/budget.py), [model.py](pc_analytic_emulator/model.py) |
| Standalone inference on native arrays | [pc_analytic_emulator/predict.py](pc_analytic_emulator/predict.py) |
| Diagnostic replacement of predicted photon budgets | [pc_analytic_emulator/diagnose_errors.py](pc_analytic_emulator/diagnose_errors.py) |

## What the teacher solves

Each redshift uses its original native cumulative `n_ion` photon field and
`1 + density` atom field. There is no additional density multiplier or emissivity
time integral. Sources emit dose batches along a finite angular quadrature;
characteristics cross voxel faces and apply the finite-exposure absorption law
to the evolving neutral column. Directions share the cell's atom inventory.

For the initially neutral campaign, the independent photon ledger is

```text
source = field-ionized atoms + remaining radiation + escaped photons + unemitted source.
```

The reservoirs are stored independently of the final neutral field. Successful
completion also requires bounded finite fields and the solver's stopping
criteria. A closed ledger alone does not mark an unfinished solve as complete.

The physical calculation is a frozen coeval, monochromatic model without
recombination, thermal/spin-temperature evolution, photoheating feedback,
separate helium chemistry or a finite-light-speed time history. It retains the
native effective H+He-nuclei count normalization, assigning each effective atom
hydrogen opacity. These switches are checked when reading native inputs.

Native ES uses `R_BUBBLE_MAX` as its maximum filtering radius. The analytic
teacher instead uses resolved neutral opacity and its declared boundary;
it does not apply that ES radius as an extra absorption sink or ray cutoff.
`physical_mapping` records this difference in native-input, teacher and training
reports. The native density/source prescription is shared, while the
ionization/propagation prescription is replaced.

The default teacher boundary, `source_periodic_cube`, follows one source-centred
periodic density domain and records photons leaving it. `periodic` recycles
photons indefinitely; SCRIPT uses a closed periodic photon box. Diagnostics
compare both analytic boundaries to distinguish this effect from allocation
differences. Some coarse angular quadratures can leave neutral cells outside
the sampled photon paths; such incomplete periodic solves retain their remaining
inventories and are reported as incomplete.

## What the emulator learns

The field network predicts a correction from native source, density and ES
fields and their smoothed features. A separate estimator predicts absorption
efficiency `eta`. Before projecting the field, it sets

```text
absorbed_target = eta * min(total_source, total_atoms)
predicted_unabsorbed = total_source - absorbed_target.
```

The final bounded field is projected onto `sum(atoms*xHII) = absorbed_target`.
This conserves the predicted global budget; agreement of that budget and the
spatial field with the teacher is measured separately. The scalar reservoir is
not a source-resolved prediction of measured escape.

V3 refits its base models using only the training seeds. It holds each fitted
field fixed while comparing budget estimators and projection methods, selects
the base on validation data, and learns a zero-initialized residual around that
complete deployed prediction. The original base remains eligible. Field and
Fourier residuals enter validation selection; held-out labels do not.

## Run the full pipeline

Use a Python environment with the matching native 21cmFAST extension, NumPy,
SciPy and Matplotlib. Teacher generation needs the native GSL/FFTW dependencies
and a C++/OpenMP compiler. SCRIPT comparisons additionally need a Fortran
compiler. Emulator training and inference do not import the native extension.

1. Generate genuinely independent native simulations and teacher labels:

   ```bash
   python -m research.pc_analytic.campaign \
       --config /absolute/teacher_config.json \
       --output /absolute/teacher_results
   ```

   `teacher_config.json` supplies a full native `InputParameters.asdict()`
   template through `native_template_json`, plus geometry, seeds and resources.
   See [teacher instructions](pc_analytic/README.md) and the completed diagnostic
   campaign configuration linked from [the validation report](pc_analytic/VALIDATION_20261116.md).

2. Train with disjoint simulation seeds:

   ```bash
   python -m research.pc_analytic_emulator.train \
       --config /absolute/training_config.json \
       --output /absolute/emulator_results
   ```

   Set `input_root` to the completed teacher results and `training_method` to
   `warm_started_v3`. Set explicit `train_seeds`, `validation_seeds`, `test_seeds`
   and `model_seeds`. `summary.json` gives final evaluation;
   `comparison_summary.json` separates the controlled budget/projection
   comparisons and residual gains. The self-contained deployed model is in
   `model/`, including its frozen base when selected.

3. Predict a new native field without teacher labels:

   ```bash
   python -m research.pc_analytic_emulator.predict \
       --model /absolute/emulator_results/model \
       --input /absolute/native_snapshot/source.npz \
       --native-inputs /absolute/native_snapshot/native.json \
       --redshift 7.5 --box-len 50 \
       --output /absolute/new_prediction
   ```

   The command checks the native parameters against the trained model, verifies
   array hashes when archived native metadata provides them, and independently
   audits the saved field and predicted unabsorbed reservoir. It writes
   `fields.npz`, `report.json`, then `COMPLETED`; existing outputs are not replaced.

4. Diagnose a native snapshot against original SCRIPT:

   ```bash
   python -m research.pc_analytic.diagnostics \
       --snapshot /absolute/native_snapshot \
       --output /absolute/new_diagnostics \
       --threads 2 --coarse-dimension 8
   ```

   This runs adjacent angular/dose refinements, both boundary conventions,
   independent ledgers, power spectra and geometric MFP. Grid restriction is
   explicitly an input-and-transport sensitivity test, not a native source
   resolution convergence certificate. Undefined spectra and empty phases
   remain undefined.

For Gadi, use the frozen-source launchers in
[pc_analytic/README.md](pc_analytic/README.md) and
[pc_analytic_emulator/README.md](pc_analytic_emulator/README.md).
Use a fresh run directory after a code or configuration change. Unchanged runs
verify receipts before reusing completed work.
