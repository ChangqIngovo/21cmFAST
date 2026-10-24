# Exact queue and ray-layout experiments

These experiments change storage and event-queue organization only. All events
retain their actual floating-point timestamps and the `(time, node_id)` priority.
The physical model, eight-child gas capacities, source positions and quadrature
banks are identical to the initial lazy kernel. Later variants also change DDA
evaluation order without changing any face-crossing timestamp.

The initial implementation is preserved verbatim as `lazy.cpp`; benchmark
variant `original` compiles that file explicitly. The selected local-DDA core
is copied into `adaptive.cpp` as the default. Historical JSON records that name
`adaptive.cpp` with SHA-256
`b020180351593307f788e7f89aa616fe6f01e9cb90a9b286069d6e58adbc0f41`
refer to the content now preserved in `lazy.cpp`.

## Compact rays and heap arity

`optimized.cpp` supports `PC_HEAP_ARITY=2/4/8` and `PC_COMPACT_RAYS=0/1`.
Compact rays retain the long-double travelled distance and all double fields;
only direction and unwrapped voxel indices narrow. The record is 48 instead of
64 bytes. The whole finite support is checked before selecting compact records;
large horizons or angular banks automatically use the original wide layout.

Three alternating warmed timing repetitions produced:

| Variant | Native32 seconds | Native64 seconds |
|---|---:|---:|
| Original lazy kernel | 1.4873 | 1.7523 |
| Heap 2, wide | 1.5330 | 1.7566 |
| Heap 4, wide | 1.4024 | 1.7462 |
| Heap 8, wide | 1.5158 | 1.7189 |
| Heap 2, compact | 1.3106 | 1.7073 |
| Heap 4, compact | **1.3017** | **1.7036** |
| Heap 8, compact | 1.3594 | 1.7167 |

Allocated ray storage changes from 38,273,024 to 28,704,768 bytes in native32
and from 80,478,208 to 60,358,656 bytes in native64. This is exactly 25% less
ray storage, not a claim of 25% lower process RSS.

All compared absorption maps, source ledgers and sparse carried states are
bitwise identical. A later shared-host timing batch measured heap-4 compact at
1.428/1.712 seconds versus original 1.473/1.719 seconds. Thus the storage benefit
is deterministic, while the small native64 wall-time benefit is not robust.
The heap-4 compact version is a modest exact optimization, not the solution to
the extra spatial-refinement cost.

`PC_ALIGN_NODES=1` optionally aligns each existing 64-byte node to a cache line
without changing its size. It passes equality tests but has no timing result in
this report and is not recommended on performance grounds.

## Calendar queue: correct but not a useful default

`calendar.cpp` partitions the interval into 1024, 4096 or 16384 fixed time
buckets. It does not quantize event timestamps. Each small bucket stores an
unsorted indexed vector and retrieves its exact minimum. A bucket exceeding
32 entries converts to an indexed binary heap, avoiding quadratic behavior when
many uniform source cells exhaust at the same time. Membership and removal are
indexed; new earlier entries can move the current bucket cursor backward.

| Variant | Native32 seconds | Native64 seconds |
|---|---:|---:|
| Original lazy kernel | 1.4732 | 1.7190 |
| Heap 4, compact | 1.4276 | 1.7123 |
| Calendar 1024, compact | 1.5048 | 1.7849 |
| Calendar 4096, compact | 1.4510 | 1.7615 |
| Calendar 16384, compact | 1.4463 | 1.6943 |

The best difference is comparable to timing variation, and sparse cases regress.
Keep this as a negative experiment. Reducing global-heap work does not remove
the dominant cost of many ray/cell operations in these source fields.

## Checks and reproduction

Thirty-seven tests pass across `test_optimized_core.py` and `test_local_dda.py`.
They cover exact maps and source
ledgers for all layouts, sparse-state equality, compact-width fallbacks, and an
independent `std::set` queue oracle. Each bucket-count oracle runs 300,000 random
update/erase/pop operations, large equal-time bursts, adjacent floating-point
values at bucket boundaries, subnormal timestamps, duration rescaling and
dynamic node reservation. Recorded native benchmark comparisons additionally
check exact equality after every repetition.

```bash
python -m pytest research/pc_adaptive_front/test_optimized_core.py -q
python research/pc_adaptive_front/optimize_core.py \
  --native32 /path/to/native32/source_capacity.npz \
  --native64 /path/to/native64/native_fields.npz --repeats 3 \
  --variants original heap2-wide heap4-wide heap8-wide heap2-compact heap4-compact heap8-compact \
  --output /path/to/optimized_results.json
```

Replace the variants with `original heap4-compact calendar1024-compact
calendar4096-compact calendar16384-compact` to repeat the calendar experiment.
`optimized_results.json` and `calendar_results.json` retain all measured timings,
input hashes, source hashes, event counts and memory diagnostics. Timings exclude
compilation and were obtained sequentially on a shared host.

## Final exact DDA comparison

`local_dda.cpp` combines the four-way heap and compact rays with a cursor inside
each partially transparent parent. It reuses the three absolute face-crossing
times until a crossed axis needs a new face. Each new face still uses the
original long-double division; no reciprocal increments accumulate. At most
three internal midplanes can be crossed before the ray leaves the parent.
Fully clear parents retain the original exact parent jump.

`face_table.cpp` adds a bounded cache of integer-offset/absolute-direction
long-double quotients. Source centers and faces are integer coordinates in the
fine grid, so lookup reproduces the same quotient and restores its sign.
At least 16 active sources and at most 64 MiB of cache are required; otherwise
the original division is used. Offsets remain unwrapped through periodic
boundaries. Both native cases allocated 1,249,920 cache bytes.

Final three alternating repetitions, measured in one batch:

| Variant | Native32 seconds | Native64 seconds |
|---|---:|---:|
| Initial lazy kernel | 1.3614 | 1.6562 |
| Heap 4, compact | 1.3050 | 1.6207 |
| Heap 4, compact, local DDA | 1.2884 | **1.5700** |
| Local DDA plus exact face table | **1.2424** | 1.6426 |

The local-DDA version gives about 5% lower median time than the initial lazy
kernel in both native cases. The face table helps native32 but is 4.6% slower
than local DDA at native64. We therefore recommend local DDA as the default
and retain the face table as a workload-dependent experiment. Timings have
visible shared-host scatter; these are modest improvements, not a new scaling
claim. The sparse two-source case is around 3 ms for all variants and local
DDA is slightly slower there.

Every timed run has bitwise-identical coarse absorption, source ledgers and
sparse carried ionization state relative to the initial lazy kernel. Additional
tests compare dense fine maps and per-angle ledgers directly at exact and
near-tied internal corners (down to a direction offset of 2^-52), nextafter
values around the distance horizon, open boundaries, periodic travel beyond
one box, and the cache-memory fallback. `final_optimized_results.json` retains
all samples and input/source hashes.

```bash
python -m pytest research/pc_adaptive_front/test_optimized_core.py \
  research/pc_adaptive_front/test_local_dda.py -q
python research/pc_adaptive_front/optimize_core.py \
  --native32 /path/to/native32/source_capacity.npz \
  --native64 /path/to/native64/native_fields.npz --repeats 3 \
  --variants original heap4-compact local-dda-compact face-table-compact \
  --output /path/to/final_optimized_results.json
```
