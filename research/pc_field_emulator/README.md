# Train a field emulator from existing PC outputs

This pilot learns the coarse ionized-fraction field of the existing fixed-gas,
prescribed-source PC calculation. Inputs are gas capacity, cumulative emitted
photons, redshift and multiscale density/source features. The target is the
checkpoint's ionized count divided by its gas capacity. It does not require new
21cmFAST simulations or install a deep-learning framework.

The first model is a small NumPy neural network. It predicts a coeval field,
rather than advancing radiation or predicting angular boundary fluxes. Existing
outputs can therefore test whether these features reproduce PC morphology,
without treating the emulator as a new coupled photon-conserving solver.
The inputs summarize the cumulative source field; they do not encode the full
emission ordering. Histories with identical cumulative sources can have different
fronts, so this pilot tests the existing source-history family. It predicts the
coarse output grid, without reconstructing the teacher's fine subcells.

## Data and validation

- One numerical arm is selected: by default four absorber subcells per axis,
  768 directions and stride four. Different refinement arms are not combined.
- The completed four-seed run is the default training source. Two seeds train
  the network, one selects the model and one supplies independent test fields.
  A single-seed input uses a chronological split, which measures temporal
  extrapolation within that realization.
- Samples come from recorded durable checkpoints. The source outputs remain
  read-only. Target fields are not features, normalization inputs or photon
  budgets.
- Each test field is evaluated before and after a global photon-budget
  projection. The latter uses input emitted counts and gas capacity; it does
  not enforce source-by-source or directional conservation.
- Metrics include field errors, volume/mass ionized fractions, 21-cm spectra,
  Fourier cross-correlation and field-generation time. The brightness model is
  saturated spin temperature without RSD, matching these training outputs.

## Run on Gadi

From the committed `Photon_conserving` checkout:

```bash
bash research/pc_field_emulator/gadi_prepare.sh --submit
```

The default input is:

```text
/g/data/qp00/sz4640/21cmfast_runs/evolution_pc_30e8497de2_20261024T122119_qC2V
```

To use committed checkpoints of the larger single-seed run instead:

```bash
bash research/pc_field_emulator/gadi_prepare.sh \
  /g/data/qp00/sz4640/21cmfast_runs/evolution_production_ecd4bbdb88_20261024T144219_8iMG \
  --submit
```

The preparer archives the selected commit into a fresh `field_emulator_*` run,
reuses the input run's Python/module environment, and records versions and
checksums. `PC_RUNTIME_PARENT` can select another existing runtime and
`PC_PYTHON` can select its interpreter. Only NumPy, SciPy and Matplotlib are
required; the native 21cmFAST extension is not imported.

The default request is **32 CPUs, 64 GB, two hours, normal queue**. Four model
initializations fit independently, with eight BLAS threads available per fit.
The validation set selects one model before the held-out test set is evaluated.
The requested walltime is an allocation, not a measured training prediction.
Override resources with `PC_MEM`, `PC_WALLTIME`, `PC_QUEUE`; set
`PC_EMULATOR_EPOCHS` for the maximum training epochs. A failed fit requires a
fresh prepared run; source simulation checkpoints are never overwritten.

The script prints the new run directory. Results are under its `results/`
directory and the log is `run_<PBS_JOBID>.log`. Keep the generated model,
selection history, data manifest, prediction fields, metrics and comparison
figures together so accuracy is tied to the actual training inputs.

The main outputs are:

- `dataset_manifest.json`: exact checkpoint hashes and train/validation/test seeds.
- `model.npz` and `features.json`: selected network, standardizer and input schema.
- `trials/seed*/history.json`: training and validation losses for every epoch.
- `model_selection.json`: validation-only choice across the four initializations.
- `summary.json` and `test_metrics.csv`: independent test errors and timings.
- `test/seed*_node*.png`: teacher/prediction/residual slices, 21-cm power and cross-correlation.
- `test/seed*_node*.npz`: teacher, raw prediction, budget-projected prediction,
  simple projected local-source baseline, capacity and photons.

Training uses uniformly sampled cells; each selected test snapshot is evaluated
on its full grid. The default epoch window is 0.01 <= mass ionized fraction <= 0.99,
avoiding trivial fully neutral/ionized examples. Inference time includes feature
construction, the network and projection, while input/output and diagnostic time
are reported separately through the full experiment duration. A matched teacher
solve is not timed, so no end-to-end simulation speedup is inferred from this job.

## Direct execution

```bash
python research/pc_field_emulator/train.py \
  --input /path/to/existing/run \
  --output /path/to/fresh/emulator_results \
  --workers 4 --model-seeds 11 23 47 83 \
  --subcells 4 --angles 768 --stride 4
```

For direct execution, set BLAS thread limits before starting Python so that
worker count times BLAS threads does not exceed the allocated CPUs. Evaluate
the raw network alongside its projected result: the projection determines the
global ionized mass, while the field and spectral errors measure morphology.
