# Event-front resource validation and parallelization audit

The corrected event solver completed all 14 bounded scaling cases. No run exceeded
the 1 GiB address-space limit, 30 CPU-second hard limit, or 40-second wall timeout.
Every case used one solver thread. The largest measured dense case, 48³ cells and
48 directions, took **3.511 seconds and 348.9 MiB peak process RSS**. These are
synthetic fixed-state tests; they do not establish the cost of a Qin20 evolution,
128³ grids, or a multithreaded solver.

## Reproducibility and scope

`validation_scaling.py` generates the inputs, runs each case sequentially in a
fresh process, and saves audits and input hashes in `validation_scaling.json`.
The current core was read-only throughout:

- `event_front.cpp`: `d97be4a8bd20132d8ed6da7314e1bef1d148c0346435d91d49fb2a36aa628fd8`
- `event_front.py`: `1fea8965df8f480d06857a01b15b08e8ad03741420552daa7eda5dac6e374688`
- `characteristics.py`: `15b87ad85050658fe5dda0ebc3fc2615c4bf430af4b0320320c96dd7c3c8dc26`

The run command is `OPENBLAS_NUM_THREADS=1 python validation_scaling.py` with the
research environment. `--resume` skips completed cases with matching core hashes;
it regenerates and verifies inputs when deriving the saved audit summary.
Compilation and Python startup are excluded from solve time, but peak RSS includes
the interpreter, libraries, generated inputs, and solver. The resource cap also
covers preprocessing. Timings are single measurements on a shared host, so small
ratios should not be interpreted as precise scaling exponents.

Cell size is fixed at one; the box grows from 32³ to 64³ cells. Capacity is
lognormal with mean one and log-standard-deviation 0.5. Exactly approximately 1%
of cells contain sources, with a fixed lognormal source-rate distribution of mean
10 and log-standard-deviation 0.7. Source and capacity realizations change with
box size. Within each box, sparse early/late cases use identical source rates.
Duration is one, the source-centred maximum travel distance is 16 cells, and the
box is periodic.

“Early” means all cells initially have positive capacity. “Late” means cells in
the union of radius-3.5-cell spheres around the source centres start transparent;
this is 83.7–85.3% of the box. These labels describe initial geometry, not a tested
cosmological redshift or evolving history. Dense cases use that same transparent
geometry, emit from every cell, and normalize their total photon budget to half
the remaining available capacity. They expose the important difference between
1%-source toys and native E-INTEGRAL fields with a positive source in every cell.

## Measurements

| Sources | Grid | Directions | Early time (s) | Late time (s) | Early peak RSS (MiB) | Late peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 1% of cells | 32³ | 48 | 0.0129 | 0.0233 | 65.2 | 65.6 |
| 1% of cells | 32³ | 192 | 0.0384 | 0.0586 | 68.1 | 67.2 |
| 1% of cells | 48³ | 48 | 0.0353 | 0.1201 | 74.8 | 75.4 |
| 1% of cells | 48³ | 192 | 0.0870 | 0.3200 | 85.8 | 83.5 |
| 1% of cells | 64³ | 48 | 0.0611 | 0.1671 | 94.6 | 93.4 |
| 1% of cells | 64³ | 192 | 0.1642 | 0.4986 | 117.9 | 117.5 |
| Every cell | 32³ | 48 | — | 0.9628 | — | 147.3 |
| Every cell | 48³ | 48 | — | 3.5105 | — | 348.9 |

All early sparse cases absorb their full budget. The late sparse cases retain
33.4–36.6%, and dense cases retain 22.7–23.8%, because a ray may reach the imposed
16-cell range without finding another opaque cell. Retained photons are explicitly
accounted for and are **not** counted as ionization. Consequently these timings
describe the declared range-limited problem; they do not measure transporting all
photons to arbitrarily distant absorption sites. They also do not establish a
physical mean-free-path absorption model or persistence of retained radiation.

## Numerical and memory checks

All 14 cases pass the independent output/source/angle ledger and capacity checks:

- Maximum relative global budget residual: `2.0063e-16`.
- Maximum conservative bound on any source's relative budget residual:
  `6.5652e-13`. This divides the maximum absolute source residual by the minimum
  positive source budget, so it bounds every source's own relative error; it is
  not the observed maximum relative error.
- Maximum relative direction-budget residual: `9.8560e-15`.
- No negative absorption or capacity excess; independent absorbed-ledger errors
  also pass the `1e-11` relative check.
- The indexed event heap never contains more than one node per grid cell.
- Ray allocation obeys `64 R <= allocated_bytes < 64 R + 262144`, where `R` is
  the number of materialized ray records. Core hashes were unchanged.

These are ledger and implementation checks, not angular-convergence tests. The
dense 48³ case materialized 4,556,976 ray records. Chunked storage avoids vector
capacity doubling, but total ray memory remains proportional to the number of
exposed source cells times the angular count. The heap bound does **not** make the
whole algorithm source-independent or O(number of cells) in memory. Dense 64³ at
192 directions and 128³ runs were deliberately not attempted under this budget.

## What can be parallelized without changing the model

Independent realizations and validation cases can run in parallel immediately.
That is useful throughput, but each event solve remains serial.

We also implemented an actual two-process experiment for two source groups whose
reachable voxel sets are provably disjoint. A ray travels at most `R`; the centre
of any cell it enters is at most `R + sqrt(3) dx / 2` from its source. Thus periodic
source separation strictly greater than `2 R + sqrt(3) dx` excludes any shared
entered voxel. Connected groups under the complementary conservative overlap
criterion can be solved independently and their maps added, provided the same
fixed capacity and range rules apply.

The fixture uses a 48³ periodic grid, two sources of rates 600 and 900, 192
directions, range 6, and periodic separation 23, exceeding the threshold 13.732.
The two-process combined map is **bitwise identical** to the monolithic map and
to the sum of sequential group solutions, with zero shared absorbing cells.

| Execution of this fixture | Wall time (s) |
|---|---:|
| Monolithic serial solver | 0.01649 |
| Two independent groups, sequential | 0.01633 |
| Two independent groups, two processes including pool and transfer | 0.04946 |

This verifies exact separability, not acceleration: process overhead dominates
this small case. With a positive source in every cell, neighbouring influence
regions overlap and the conservative source graph is connected. Therefore this
partition is not a general multicore solution for native dense Qin20 fields.

For interacting groups, independent per-source solves are invalid: sources share
target capacities, and incoming flux changes subsequent exhaustion times. The
global event order and same-time release of rays must be respected. A plausible
exact shared-memory implementation would mark all cells exhausted at an identical
event time, parallelize read-only ray tracing from that released batch, reduce
target-rate and source-ledger increments deterministically, and drain any same-time
cascade before advancing time. The current code does not implement or benchmark
this. Concurrent mutation of capacity inside ray tracing is not a safe substitute.

Equal-time event batching may have limited parallel work for generic unequal
rates, although one exhausted cell can release many rays. Processing future-event
windows without rollback or a proven independence bound would change the
competition model; infinite propagation speed gives no positive distance-based
lookahead. MPI domains synchronized at every global event may be dominated by
communication. These are development routes, not measured speedups.

## Remaining evidence needed

The corrected kernel still needs native dense-source cost tests at the intended
grid, angular resolution and range, with angular/spatial accuracy considered
together. A full evolving Qin20 run also needs a declared source closure,
recombination/state updates, and treatment of retained photons. The geometric
independence probe does not remove those requirements. Earlier timings from the
pre-correction `aca74136...` core are historical and should not be relabelled as
measurements of this version.
