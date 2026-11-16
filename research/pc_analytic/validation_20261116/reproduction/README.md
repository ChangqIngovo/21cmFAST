# Reproduce the 2026-11-16 diagnostic campaign

Run from the repository root, with a Python environment containing the matching
native 21cmFAST extension and its GSL/FFTW/compiler dependencies:

```bash
python research/pc_analytic/validation_20261116/reproduction/reproduce.py \
    --source "$PWD" --output /absolute/new_run \
    --python /absolute/native_runtime/bin/python
```

The runner creates a fresh output directory, generates 56 fitting/regression
teachers and 28 independent external teachers, then trains and evaluates the
emulator. All original parameters are in `native_template.json` and the three
configuration files. It uses two workers with two threads each. The separate
`training_before_diagnosis_config.json` records the initial experiment; the
current source includes the additional sigmoid candidate and therefore does
not recreate the earlier candidate set merely by switching configuration.

For an independent evaluation of the final saved model, use the same repository
Python environment; this step does not require importing the native extension:

```bash
python research/pc_analytic/validation_20261116/reproduction/external_audit.py \
    --source "$PWD" \
    --teacher /absolute/new_run/external_teacher_results \
    --training /absolute/new_run/emulator_results \
    --output /absolute/new_external_audit \
    --seeds 2109 2110 2111 2112 --rays 32768

python research/pc_analytic/validation_20261116/reproduction/selfcheck.py \
    --source "$PWD" --output /absolute/metric_selfcheck.json
```

The audit independently recomputes photon ledgers, full-complex-FFT spectra,
cross-correlations and voxel-traced geometric MFP. It evaluates the selected
model and the validation-best legacy model without ranking on external labels.
The metric self-check verifies analytic slab distances, empty/full phases,
full versus real FFT normalization, and agreement with the separate C++ MFP
implementation. `summarize_external.py` produces the compact JSON committed
beside the validation report. `recompute.py` and `compare_legacy.py` reproduce
the initial seed-2108 regression audit.

`execution_provenance.json` records the actual runtime, source parent and
implementation hashes used in this execution. Its absolute paths are historical
provenance; the runner resolves the supplied paths for a new execution.
`SHA256SUMS` covers the files in this reproduction directory.
