# Independent event-front review and bounded-heap optimization

Review scope: constant point-source rates, infinite propagation speed, fixed
equal-weight angular characteristics, shared opaque-voxel capacities, no
recombinations. This is a review of that explicitly discrete model, not proof
of full radiative-transfer accuracy or a completed native Qin20 integration.

## Semantic checks

`python -m unittest -v test_event_front.py` passes **nine tests**:

1. Zero source; transparent box with nonzero sources and open escape.
2. Analytic continuous competition between an upstream source and a downstream
   source occupying a still-opaque shared cell.
3. Complete capacity saturation with open escape or a periodic retained-photon
   reservoir.
4. A ray crossing a periodic boundary, and an explicit shorter allocation
   range retaining the photons instead of absorbing or discarding them.
5. Splitting a multi-source run into durations 0.17, 0.29 and 0.54 gives the
   same final absorption as one duration-1 run, to roundoff.
6. Twelve fixed-seed random fixtures preserve pointwise absorption monotonicity
   after adding a source rate.
7. A rotated oblique beam follows the same foreground/background allocation.
8. A cell exhausting exactly at the requested end time absorbs its capacity
   without unnecessarily materializing future rays.
9. Nonfinite/nonpositive time and distance, malformed direction arrays, empty
   direction sets and non-unit directions are rejected by the public API.

Each applicable test checks cell bounds, total counts, every-source and
aggregate-every-angle ledgers, and independent agreement between target and
source absorption. The test also asserts that the diagnostic peak heap size
does not exceed the number of cells. These are semantic fixtures, not tests
that reproduce the implementation's update equations as their expected answer.

The analytic competition fixture has a +x beam, upstream source rate 2 in a
cell with capacity 1, a second capacity-1 cell, downstream source rate 3 in a
capacity-10 cell, and then capacity 4. The downstream source cell exhausts at
time 2.4 and the final cell at 3.2. At time 4, source absorption is 6.4 and 9.6,
and escaped photons are 1.6 and 2.4. The implementation matches these values.
At time 2, the three affected capacities contain exactly 1, 1 and 8 photons.
This exercises simultaneous continuous illumination and lazy local-source
emission, not just isolated spherical growth.

## Reviewed invariants

An opaque cell's stored capacity is updated from its last-update time before
its incident rate changes. Since a cell never becomes opaque again in this
model, a ray need only move when its current absorber exhausts. Once a ray
escapes or reaches the permitted range, its remaining constant-rate emission
over the interval can be accounted immediately: a later absorber cannot appear
on the already transparent part of that path. Recombination would invalidate
this argument and is explicitly absent.

Own-cell source emission is absorbed locally until that cell exhausts. Its
local contribution is `source_rate * exhaustion_time`, while incoming rays
pay their own elapsed residence intervals. Only the remaining source emission
then requires angular rays. If exhaustion lies beyond the requested end time,
no ray state is necessary. This is why source cells with tiny early emission
can be cheap without dropping photon budgets.

Initial transparent-distance bounds remain conservative as more cells become
transparent. The open-box distance padding also prevents an oversized jump
across a periodic seam; wrap handling then resumes normal traversal. The tests
verify an actual wrapped absorption and the corresponding short-range retained
case. A finite range is a declared cut with a separate ledger, not an implicit
sink or a recombination prescription.

## Bugs and resource issues addressed

The original `std::priority_queue` inserted a replacement event for every ray
arrival and discarded obsolete entries later by version number. It was correct
on the reviewed fixtures but its memory was not bounded by active cells.
Stopping at the first event beyond the requested duration can leave many
obsolete future entries in the queue, so the reported number of discarded stale
entries was not a bound on retained queue storage.

The replacement is an indexed binary min-heap with exactly one scheduled
exhaustion per receiving cell. An array maps cell indices to heap positions.
Ordering remains `(time, cell)`, including deterministic ties. Updates allow
both upward and downward changes: rates mathematically only increase while a
cell stays opaque, but lazy floating-point subtraction can move a recomputed
time upward by an ulp. The old latest-version semantics are therefore retained.
No floating-point expression controlling rates, capacity, positions or source
ledgers was changed.

The heap stores at most Ncell nodes; its vector capacity is explicitly capped
at Ncell and this invariant is asserted. The obsolete per-cell version array
is replaced by the position array. Each node occupies 16 bytes on this build,
with four bytes per cell for its position. Peak node count and update count are
reported through diagnostic getters without changing the original six-double
statistics-buffer ABI.

Ray storage now uses fixed blocks of 4096 unchanged-precision `Ray` objects,
rather than doubling a vector and temporarily retaining both old and new
allocations. A Ray is 64 bytes on this build. Unused capacity is less than one
256-KiB block, plus a small vector of block pointers. Ray indices and all
floating-point fields are unchanged. A guard rejects further materialization
before exceeding `INT_MAX` ray indices; grid cell-count overflow is also
checked. These checks fail explicitly rather than silently wrapping indices.
An out-of-memory allocation is not yet a recoverable production fallback.

The Python API review additionally found acceptance of nonfinite times/ranges,
and direction arrays with a wrong second dimension. The root agent fixed those
checks and the tests now cover them. The shared fixed-name temporary `.so`
could be overwritten while another process had it mapped; compilation now
uses a unique temporary directory.

## Bitwise regression and preliminary performance

The pre-optimization C++ SHA256 was
`4b616f324cc553e4e95a2ea5481a01386ba41aeb155238a9ec00bea1095d91f8`.
The indexed-heap plus blocked-ray version reviewed here has SHA256
`aca74136b8383b7202903a4570cc78473feb4ae4f487a832b758cb2db9426c19`.

Using the saved native-derived 32³ source/capacity fixture and the same periodic
duration-1 run, every absorption cell is **bitwise identical** to the saved
pre-optimization map for both K=48 and K=192, in two repeated runs each.
Event, ray, geometry-query and assignment counts are unchanged. Source and
angle ledgers reproduce the original numerical values. The original maps and
benchmark report were preserved.

| Angles | Original seconds | Optimized seconds, two runs | Original peak RSS | Optimized peak RSS |
| --- | --- | --- | --- | --- |
| 48 | 1.804 | 0.813, 0.864 | 162.22 MiB | 115.19 MiB |
| 192 | 8.316 | 3.646, 4.159 | 392.07 MiB | 262.56 MiB |

Both optimized runs peak at 32768 heap nodes. K=192 has 10,834,450 ray-to-cell
assignments but no stale heap nodes. Allocated ray blocks total 49.25 MiB for
K=48 and 196.75 MiB for K=192. The final native agent's serialized measurements
should supersede these preliminary shared-host timing numbers; bitwise
regression does not depend on timing noise.

Raw diagnostic data are in `event_heap_only_K48.json`,
`event_heap_only_K192.json`, `event_indexed_blocks_K48.json` and
`event_indexed_blocks_K192.json`. The first pair isolates the heap improvement;
the second adds bounded-waste ray storage. They record source hashes and map
equality as well as timing and accounting.

## Honest scaling and remaining limitations

Let R be the number of materialized rays, A the number of ray assignments to
opaque cells, E the number of exhausted cells, and G geometry queries.
E is at most Ncell. R is `K * number_of_sources_exposed_before_the_end_time`,
including sources initially in transparent cells. Work is

`O(Ncell + R + G + (initial_active_cells + A + E) log Ncell)`.

Heap storage is O(Ncell), but ray storage remains O(R). There is no source-count
independent worst-case cost claim. Dense source fields at large angular
resolution can still require many gigabytes, and the current solver is serial.
The per-advance traversal safety limit is one million geometry queries; it is
an explicit failure bound, not proof that arbitrarily long periodic paths are
handled. No recombinations, finite light speed, time-varying emissivity,
subcell chemistry or resumed retained-radiation state has been added.

An end-time restart with the same constant source rates and remaining
capacities is validated for this no-recombination model. It should not be
generalized to arbitrary radiation histories or sinks. The `1e-10` geometric
boundary displacement is a numerical tie convention; its error near exact
edges is separate from angular and voxel convergence. The native-derived
fixture treats cumulative native photons as rates over an arbitrary unit
interval, so these speed and map checks establish implementation preservation,
not physical redshift evolution or unchanged native source statistics.
