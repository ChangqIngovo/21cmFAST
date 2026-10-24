# Fixed-source front refinement

This solver addresses the front-discretization effect isolated in
`../pc_scale_origin`: reducing the absorber cell size at fixed sources changes
the ionization--density correlation, even when photon counts close. The
original source positions, gas density information and photon budgets are
retained. The new spatial control divides each original cell into 2, 4 or 8
cells per axis, corresponding to 8, 64 or 512 gas subcells.

## Allocation rule

An original cell with atom count C has subcell capacities C/f^3. Its point
source remains at the original centre. Exactly eight subcells touch that
centre for every supported even f, and the octant-balanced quadrature supplies
each of these eight cells with S/8. The source is not spread over all f^3
subcells. All other gas cells receive radiation by geometric propagation.

An isolated uniform parent illustrates why this matters. In the eight-cell
model, all eight children touch the source and have equal capacity and rate,
so no source ray leaves the parent until it has absorbed C photons. A
continuum spherical front touches the nearest parent face after only
(4*pi/3)*(dx/2)^3*n_H = (pi/6)*C photons. Increasing the gas resolution lets
inner gas clear and radiation reach the parent boundary while other regions
of that parent remain neutral.

Each opaque subcell combines the rates of the rays that currently reach it.
Its exhaustion time is remaining_capacity / incident_rate after the current
event time. Rays then continue from their original source along the same
directions. The common event clock handles simultaneous sources. This retains
the static-density, constant-emission, infinite-light-speed approximation.

Untouched neutral gas is implicit. Cleared cells are represented by bit masks,
and only illuminated, partly neutral cells need full event nodes. Cleared
nodes are recycled. An entirely cleared original parent can be skipped during
ray traversal. This saves state memory; spatial refinement still increases
the number of physically represented absorption events.

## Exact isolated-parent shortcut

Before a foreign ray enters an initially neutral parent, its uniform gas and
central source follow an isolated solution depending only on f, the angular
bank, and normalized emitted dose u=S*t/C. The solver tabulates that small
problem once per angular bank, through the first ray leaving the parent.

The parent can remain implicit until the earlier of a foreign-ray arrival and
its own first escape. At that time the solver restores the exact tabulated
fine-cell ionization, accounts for the S*t photons already absorbed, and
launches the remaining angular rates from the original source centre. Rays
then follow the ordinary shared event clock. Parents that never interact are
evaluated at the interval endpoint without materializing their internal rays.
This shortcut changes the representation of the same discrete evolution.

Short horizons that can truncate a ray inside its source parent use the
ordinary solver. Parents with resolved carried states likewise use the direct path. The
option `use_local_templates=False` disables the shortcut for independent
field and source-ledger comparisons. `clear_template_cache()` resets the
lookup cache for cold-cost measurements. Reported first solves include cache
construction; warmed repeated timings are reported separately.

The returned state preserves the resolved gas geometry for another interval
with the same capacity grid and spatial refinement. Every subsequent call
receives newly emitted photons. Reported escaped/retained photon counts are
not automatically re-emitted.

## Independent spatial and angular controls

```python
from refined import solve

result = solve(source, capacity, subcells_per_axis=4, angles=1536,
               maximum_distance=8 * source.shape[0], return_dense=False)
next_result = solve(next_source, capacity, subcells_per_axis=4, angles=1536,
                    maximum_distance=8 * source.shape[0],
                    initial_state=result["state"])
```

`maximum_distance` is in original coarse-cell units. `angle_counts` optionally
supplies a direction count for every original source cell and overrides
`angles`. It allows identical angular banks to be used across different gas
resolutions. Increasing f does not silently increase the photon budget, move
the sources, create additional density information or alter the angular bank.

The native restricted 21cmFAST adapter accepts `--solver refined-front`,
selecting f=4 with the existing coarse-source angular predictor. It constructs
native ionized and brightness-temperature outputs from the refined field.
The adapter uses frozen L-INTEGRAL sources with no recombinations, spin
temperature evolution or photoheating feedback. It remains a comparison of
independent coevals, not the full coupled Qin20 history.

## Verification and benchmarks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  research/pc_refined_front -q --confcutdir=research/pc_refined_front
python research/pc_refined_front/benchmark.py --help
```

The independent dense reference uses explicit point-source coordinates on the
same refined gas mesh. Comparisons include f=2/4/8, mixed per-source angular
counts, open and periodic boundaries, short transport horizons, very weak
budgets, unchanged and changed-source continuations, and source symmetry.
An integrated spherical solution tests spatial accuracy separately from this
implementation equivalence. Native fixture spectra are all measured on one
common output grid.

Here "sparse" and "lazy" describe state allocation and ray materialization.
The selected f is uniform across the physical gas mesh; the solver does not
yet select spatial resolution from a local error tolerance.

## Measured spatial improvement

The isolated sphere uses an independently integrated continuum volume in each
original measurement cell. At 49,152 directions, the L1 ionization-count error
divided by emitted photons is:

| Gas subdivision | 8 cells | 64 cells | 512 cells |
|---|---:|---:|---:|
| Integrated-sphere L1 / photons | 5.720% | 2.680% | 1.298% |

The 64-cell model removes 53.1% of the eight-cell error. The sphere integration
itself changes by only 0.0016% of the photon budget when its quadrature doubles.

One native32 fixture was conservatively reduced to source/density grids 8^3
and 16^3. All spectra were measured on the same 8^3 grid, using the same
physical band with 18 full Fourier modes. At Q=0.9:

| Directions per source | Eight-cell grid difference | 64-cell grid difference |
|---|---:|---:|
| 1,536 | -11.39% | -4.05% |
| 3,072 | -11.57% | -2.72% |

The grid difference is b_HI^2(source16)/b_HI^2(source8)-1. Subdivision reduces
its magnitude by 64--76% in this fixture. These input-grid comparisons include
the effects of moving and aggregating sources as well as density coarsening;
the front-only arms within each source grid hold those inputs fixed.
Doubling angles still changes the source8/f4 power by 1.58%, so the remaining
grid difference is not an angular-converged limit. The fixed-source f4-to-f8
comparison also retains a spatial difference. The large-volume follow-up
tests whether the improvement persists across seeds and larger grids.

`sphere_results.json`, `native_results.json`, `native_pair8.json`,
`native_pair16.json` and `native_convergence.json` retain the source hashes,
ledgers, cold and warmed timings. `benchmark_summary.json` collects the
comparisons; `refinement_results.png` plots them. The timing covers the solver
call, including preparation and returned state, rather than the full
21cmFAST initial-condition and galaxy-source calculation.

For source16 at Q=0.9 and K=3072, the existing eight-cell solver takes 12.663 s
and the refined 64-cell solver takes 33.058 s: a factor of 2.61. Thus the
spatial improvement has a measured computational cost. At K=1536 the refined
model processes 179,264 exhaustion events versus 22,778 in the old solver,
and 162.2 million geometry queries versus 74.9 million. These are additional
resolved propagation events, not just storage overhead.

## Ray storage

Source rates are stored once per source. Compact ray records retain
long-double travel distance and arrival time in 48 bytes; coordinates or
angular banks exceeding the compact integer range use 64-byte records.
The geometry and arithmetic are identical between these layouts.

If the potential ray reservation fits `contiguous_ray_bytes` (default 1 GiB),
the solver reserves one contiguous array. Larger reservations use stable
4,096-ray blocks. Setting the threshold to zero forces blocks. The threshold
selects storage layout; it is not a limit on total ray storage or process RSS.
`packing_benchmark.py` compares this layout with the frozen
`reference_prepack.cpp`, which implements the same model with 80-byte vector
records. This reference is retained for reproducible storage-cost and
bitwise-equivalence checks.

The compact contiguous layout reduces the same 64-cell solver's ray
reservation by approximately 55% in the measured native16 cases. All compared
fields, source ledgers and carried states are bit-identical. Wall-time changes
are not consistently favourable, so this is a memory improvement rather than
a demonstrated speedup. `packing_results.json` retains the paired samples.

## Large-volume comparison on Gadi

After pulling this branch in the existing fork checkout:

```bash
bash research/pc_refined_front/gadi_prepare.sh /path/to/completed/large_pc_run
```

This prepares an immutable comparison against the existing native fixtures.
Adding `--submit` submits the prepared run. The default allocation is 32 CPUs,
320 GB and 12 hours. The event kernel is serial; independent cases run in
parallel. No native initial conditions or density fields need to be regenerated.

The default matrix compares old adaptive f=2, new refined f=2 and new refined
f=4 for the four parent seeds, original grids 25/50 and photon doses 0.8/0.9:
48 cases. All arms share stored per-source angular counts. Fields are projected
onto the same 25^3 output grid. `PC_GRIDS`, `PC_REFINEMENTS`, `PC_DOSES`,
`PC_THREADS`, `PC_MEM` and `PC_WALLTIME` override the corresponding settings.
Grid 100 and f=8 are explicit options.

The Gadi driver defaults to a 4 GiB contiguous-ray reservation threshold per
case, adjustable with `PC_CONTIGUOUS_RAY_GB`. Preflight reports potential ray
capacity and the expected storage layout. A case falling back to blocks may
still use more memory than that threshold if it materializes enough rays.

`results/summary.json` reports the independent ledgers, old/new f=2 equivalence,
paired-seed spectra and runtime ratios. `results/experiment.json` records the
selected inputs, angle banks and code hashes. The prepare and run stages reject
changed inputs. The requested walltime is an allocation, not a measured runtime
prediction.

The principal runtime ratio includes one angular prediction on the same worker
in both arms. Raw solve time and preparation-subtracted transport plus output
time are separate fields. Compilation is excluded; first-use template
construction is included. Native allocation counters are partial bookkeeping,
not measured process peak RSS.
