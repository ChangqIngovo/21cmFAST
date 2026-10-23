# Source-consistent, adaptive angular photon conservation

This directory contains a new research implementation and comparisons built on
native 21cmFAST source fields. It does **not** replace the existing
`global-budget` production setting. The leading candidate is in `hybrid.py` and
`hybrid.cpp`; `native_driver.py` demonstrates a restricted, real 21cmFAST
integration including native brightness-temperature calculation.

Read `REPORT_zh.md` for the decision and limitations, `comparison_report.md` for
measured comparisons, `REVIEW.md` for the independent review, and `literature.md`
for the surveyed model families and primary references.

## Physical contract

Inputs to the spatial kernels are photons emitted during a unit-duration
constant-rate interval, total atom counts, and optional initially ionized atom
counts. Gas is stationary; propagation has infinite speed; a cell with remaining
capacity is opaque. Recombination, frequency-dependent opacity, gas advection,
thermal evolution and UV feedback are outside these kernels.

Every directional source has its own photon budget. Rays retain their direction
through transparent cells. Rates arriving at a cell are added before determining
when that cell becomes transparent. Source-specific angular resolution reduces
cost relative to assigning the largest quadrature to every source. This is an
approximation with explicit angular and spatial convergence tests, not a new
scalar excursion-set threshold or a proof of radiative-transfer accuracy.

`source_budget.py` separates source normalization from transport and provides
explicit cumulative-history and instantaneous-rate adapters. Do not clip
negative changes of cumulative Eulerian source maps into positive emission.

## Run the tests and spatial comparisons

From the repository root, with an installed development build of 21cmFAST:

```bash
python -m pytest research/pc_reconsidered -q --confcutdir=research/pc_reconsidered
python research/pc_reconsidered/compare_models.py \
  --methods es-control es-global es-front radial-shell voronoi adaptive-event \
  --reference-angles 16384 --dense-reference-angles 768 --refinement 4 \
  --output /path/to/fresh/results.json
```

Compilation is excluded from spatial-kernel timings; setup and radius prediction
are included. These timings are not complete simulation timings. The provided
JSON files preserve the exact scope, source hashes and reference-refinement
error. At small bubble radii, spatial error exceeds angular error.

## Run actual native sources and native 21-cm outputs

```bash
python research/pc_reconsidered/native_driver.py \
  --output /path/to/fresh/native32 \
  --dimension 32 --box-len 200 --redshift 8 --threads 4
```

This is explicitly **not exact Qin20**. It starts with Qin20 astrophysical
parameters but selects `L-INTEGRAL`, ACG only, no recombinations, no spin/X-ray
evolution, and no photoheating feedback. Input differences are saved. It compares
native ES, an additive-source ES control, and the angular PC candidate. Each
coeval is a cumulative-budget frozen-gas construction, not a time-integrated
radiation history. Native caches never receive research PC outputs under an ES
cache identity. The native 21-cm calculation uses the saturated-spin assumption
implied by the disabled spin module.

To study the exact Qin20 **source fixture**, separately prepare and freeze native
Qin20 outputs with `../pc_alternatives/native_benchmark.py`, then pass its
`source_capacity.npz` to the comparison harness. The E-INTEGRAL source adapter is
a declared new spatial closure, and the transport comparison freezes X-rays and
omits recombination evolution. It must not be described as a full Qin20 PC run.

## Gadi

`gadi_compare.pbs` runs the tests, spatial comparisons and restricted native
integration with a supplied installed Python environment. Submit from an
isolated checkout of the desired commit:

```bash
qsub -v PC_REV="$(git rev-parse HEAD)",PC_PYTHON=/absolute/path/to/venv/bin/python \
  research/pc_reconsidered/gadi_compare.pbs
```

The job records its commit, copies the research code with `git archive`, checks
the installed package location and tracked source files, records and copies the
native binary into the snapshot, and creates a fresh result directory. The supplied environment must
already contain a build of that checkout; these checks cannot reconstruct a
prebuilt binary's provenance. Native 21cmFAST uses allocated OpenMP threads; the shared
angular event clock is currently serial. Requesting more CPUs does not make
that transport kernel parallel.

## Files

- `fast_candidates.py`: additive ES control, global calibration, front-ranked
  allocation, source-anchored shell allocation and capacitated nearest-source
  allocation. Exact total counts do not establish directional correctness.
- `hybrid.py`, `hybrid.cpp`: adaptive directional event kernel.
- `merge_sources.py`: optional source-position approximation deep in transparent
  regions; separate from the default candidate and not certified by its ledger.
- `source_budget.py`: species-preserving source adapters and time-budget checks.
- `native_driver.py`: guarded native integration and saved native observables.
- `compare_models.py`: analytic/reference tests, count audits, morphology,
  cross-correlation, spectral comparisons and timings.
- `test_*.py`: regression and independent contract tests.

The old production mode and all original Qin20 outputs remain available for
comparison. Full Qin20 integration needs a consistent update of UV rates,
recombination, temperatures and feedback before it can be promoted.
