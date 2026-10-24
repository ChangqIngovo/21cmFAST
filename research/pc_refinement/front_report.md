# Subcell-front refinement: a measured accuracy/cost tradeoff

The coarse opaque-cell model has a spatial error even at converged angular
resolution. Its simultaneous cell-exhaustion rule holds every beam in a voxel
until the entire voxel clears. Subdividing its gas capacity allows different
parts of the voxel to clear separately and tests this approximation directly.

`front_subcells.py/.cpp` implements a factor-two subdivision along every axis.
Each coarse cell of capacity C becomes eight capacities C/8. An original
cell-centred source remains at its original physical point, which is the common
vertex of eight refined cells. Its angular directions are partitioned into
octants, each with Q/8 photons; no source is moved to a refined-cell centre.
The same total number of rays is used at either spatial resolution. All eight
groups share the continuous multi-source exhaustion clock. Their absorption,
escape and retained-photon ledgers are summed back to the original source.

The matched angular rule has eight sign images and three cyclic permutations
of each positive-octant node. It has zero dipole, isotropic second moment, and
exact octant quotas. It is not claimed to be optimal or fully cube-rotation
invariant. The experiment compares this rule on both grids to separate angular
and spatial changes, and also retains the original Fibonacci baseline.

## Measurements

On a uniform sphere of radius four coarse cells:

| Resolution / angular rule | Continuum sphere L1/Q |
|---|---:|
| Original coarse, default Fibonacci K=1536 | 11.42% |
| Coarse, matched balanced K=1536 | 12.70% |
| Eight subcells, matched balanced K=1536 | 7.39% |
| Coarse, matched balanced K=6144 | 11.56% |
| Eight subcells, matched balanced K=6144 | 5.23% |

The continuum oracle integrates the sphere over original coarse cells. Thus
these improvements reduce a physical spatial-discretization error; they are
not merely improvements against the old opaque-cell solver. At K=1536 the
refined angular quadrature is under-resolved relative to its smaller gas cells.
The high-K comparison separates that angular limitation more clearly.

At default angular counts on the frozen native32 source field, full subdivision
costs roughly 1.4–1.5 seconds versus about 0.55 seconds for the original solver.
Ray allocation barely changes (0.569 million to 0.594 million); the extra work
comes from about 8.5 times more exhaustion events and initially twice as many
voxel queries. The final normalized map difference is about 1.52%, but there is
no independent continuum oracle for this native field, so this change is not
labelled a measured 1.52% improvement.

Exact dynamic skipping of a parent whose eight cells have all cleared reduces
queries from 15.06 million to 12.45 million. It uses the original ray and exact
DDA face-time comparisons, including midpoint comparisons on non-crossed axes.
Skip-on/off absorption maps agree in tests. The measured wall-time gain is
small and host-dependent; it does not remove the additional front-event cost.
`front_results.json` records reproducible timed samples rather than a claimed
universal speedup.

## Tests and limits

Ten tests cover local all-partial absorption, exact source placement, angular
moments, periodic translation, short propagation ranges, an opaque plane,
initial-ionization accounting, input rejection, skip equivalence, and time
partition with retained fine-cell state. The last qualification matters:
restarting from only a coarse ionized fraction would redistribute its partial
ionization uniformly among eight subcells and erase the resolved front. The
wrapper therefore returns `subcell_absorbed` and accepts
`initial_subcell_ionized` for consistent fixed-gas continuation.

All source, capacity and global ledgers pass. Chemistry, source evolution,
thermal coupling and density advection are outside this experiment.

**Decision:** retain this as a higher-spatial-accuracy experimental option and
reference. It does not meet a strict 1.5–2 times cost budget in the measured
native case, and is not recommended as the default low-cost update. A truly
adaptive gas mesh, which refines only an active front and coalesces transparent
interiors, remains a distinct implementation task; this uniform prototype does
not demonstrate its eventual accuracy or speed.

Reproduce from the repository root:

```bash
python -m pytest research/pc_refinement/test_front_subcells.py -q --confcutdir=research/pc_refinement
python research/pc_refinement/front_benchmark.py --native /path/to/source_capacity.npz
```
