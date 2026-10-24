# Angular validation of the refined front

Spatial front refinement does not by itself establish angular convergence.
This workflow tests directional sampling independently of the source and gas
meshes, while checking both Fourier statistics and resolved gas allocation.

This experiment multiplies each stored source-wise direction count by exactly
1, 2, 4, 8 and 16. It does not rerun the angular predictor, cap the multiplied counts
at the previous ceiling, or move any source. Each spatial subdivision consumes
the same multiplied bank. The source positions, photon counts, piecewise gas
density, propagation horizon and output measurement grid remain fixed within
each angular comparison. Directions at different counts need not be nested;
the test checks convergence of the existing quadrature rule, and changes need
not decrease monotonically at each step.

The default matrix has 160 static cases: four seeds, original grids 25^3/50^3,
Q=0.8/0.9, eight/64 gas subcells, and five angular multipliers. Multiplier 1
repeats the completed parent's refined implementation, whose f=2 result
already reproduced the old adaptive model. The repeat must reproduce the
saved parent fields before the new comparisons can be interpreted.

## Run on Gadi

After pulling the committed branch in the fork checkout:

```bash
bash research/pc_refined_front/gadi_angular_prepare.sh --submit
```

The script selects the latest completed `refined_pc_*` run. To select a
specific parent:

```bash
bash research/pc_refined_front/gadi_angular_prepare.sh \
  /g/data/qp00/sz4640/21cmfast_runs/refined_pc_COMMIT_TIMESTAMP_SUFFIX --submit
```

It reuses the parent runtime and its original native fixtures. No native
initial conditions are regenerated. The source is archived from the selected
Git commit, and the inputs, prepared angle banks, compiler and Python
environment are checked for changes. The original solver files must match
the completed parent. A fresh `angular_pc_*` directory receives the results.

The allocation defaults to 32 CPUs, 320 GB, `hugemem`, and 12 hours. The time
limit is an allocation, not a runtime prediction. Up to 32 independent cases
run concurrently; each event-transport kernel is serial.

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `PC_ANGULAR_PARENT_RUN` | Latest completed refined run | Explicit parent run |
| `PC_PARENT_ROOT` | `/g/data/qp00/sz4640/21cmfast_runs` | Parent search directory |
| `PC_ANGULAR_RUN_ROOT` | Parent's directory | New run directory root |
| `PC_GRIDS` | `25 50` | Original source/density grids |
| `PC_REFINEMENTS` | `2 4` | Subcells per original-cell axis |
| `PC_DOSES` | `0.8 0.9` | Emitted photons / atom count |
| `PC_ANGULAR_MULTIPLIERS` | `1 2 4 8 16` | Exact per-source direction-count multipliers |
| `PC_ANGULAR_POWER_TOLERANCE` | `0.01` | Target-band fractional power-change diagnostic |
| `PC_ANGULAR_FIELD_TOLERANCE` | `0.01` | Count-field L1 / emitted-photon diagnostic |
| `PC_CONTIGUOUS_RAY_GB` | `4` | Contiguous ray-storage selection threshold per case |
| `PC_THREADS` | `32` | Independent workers, from 1 to 32 |
| `PC_MEM` | `320GB` | PBS memory allocation |
| `PC_WALLTIME` | `12:00:00` | PBS walltime allocation |
| `PC_QUEUE` | `hugemem` | PBS queue |

The ray threshold selects a storage layout; it is not a total memory limit or
an RSS measurement. Preflight reports the potential ray reservation. Both
larger banks and sparse cell states contribute to actual memory use.

## Interpretation

`results/summary.json` separates parent reproduction, photon accounting,
angular changes at fixed gas resolution, spatial changes at each angular
resolution, and original-grid differences. Power ratios, resolved-subcell and
original-grid count-field L1, and Fourier field correlations must be considered
together: matching power alone does not establish matching morphology.
The subcell comparison uses the sparse cleared-cell masks and partial-cell
counts directly, without materializing dense fine grids. Source ledgers and
the mass-weighted ionized fraction check that these differences arise from
allocation rather than a changing photon budget.

`results/experiment.json` records the matrix, source and angle hashes, and
angle-count distributions. `results/execution.json` records observed run time
before collection, worker allocation and failures. Per-case `result.json`
files retain solver timing and allocation counters; counters do not substitute
for measured peak RSS. `fields.npz` retains original-grid and common-grid
absorbed counts and the sparse fine state for spatial comparisons.

The two finest adjacent pairs (by default x4 to x8 and x8 to x16) must each
change target-band power, original-grid field L1 / emitted photons, and
resolved-subcell L1 / emitted photons by less than 1%, together with the
recorded mass-fraction and retained-photon limits.
At least three angular levels are needed for this diagnostic. These are
explicit convergence diagnostics for
the tested sequence, not a continuum error bound; x16 is not assumed converged.
Results near the common
measurement-grid Nyquist remain resolution-sensitive and are reported with
their relative wave number.

A local native-field control passed the target-band power-change criterion
from x4 to x8 but failed it from x8 to x16. This is why a single apparently
stable angular step is insufficient and both finest adjacent pairs are
required.

`COMPLETED` means that the calculations, numerical checks and parent
reproduction succeeded. Inspect the separate convergence results even when
this marker exists. This is a frozen-source, frozen-density angular study;
it does not evolve a coupled Qin20 reionization history.
