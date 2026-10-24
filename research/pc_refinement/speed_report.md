# Exact traversal and packet-storage experiments

These variants preserve the original photons, quadrature, opacity model,
event clock and floating-point arithmetic used to consume atoms. They are
performance experiments, not changes to physical accuracy.

## Rejected traversal arithmetic changes

`index` avoids integer modulo while the cursor remains in the fundamental box.
`filtered` also selects clearly separated face crossings with double estimates,
then computes the winning crossing with the original long-double division.
Near ties and non-normal intermediates fall back to the original three-axis
calculation. Half-integer source positions and int32 cursor faces are exactly
representable in double; two roundings in reciprocal-and-multiply have relative
error below roughly three unit roundoffs. The selection margin uses 64 double
epsilons, with the original division retained for the stored path parameter.

Neither variant improved the tested wall time. Five alternating repetitions:

| Case | Original median/s | Index/s | Filtered/s |
|---|---:|---:|---:|
| Native 32 cubed | 0.5339 | 0.5381 | 0.5873 |
| Native 64 cubed | 0.6231 | 0.6238 | 0.6642 |

The original long-double traversal should be retained. Measurements are in
`speed_results.json`; the filtered code remains solely as a reproducible
negative experiment.

## Useful compact-packet variant

`compact` retains the original traversal arithmetic. It instead stores the
unwrapped voxel cursor in signed 16-bit integers and the quadrature index in an
unsigned 16-bit integer, while keeping the source/linked-list indices as int32
and all floating-point fields unchanged. Reordering the fields gives a 48-byte
packet instead of 64 bytes on this GCC/x86-64 build.

The compact kernel is used only when the complete cursor support satisfies
`n + maximum_distance + 2 <= 32767`, all direction components have absolute
value at most one, and the concatenated quadrature has at most 65536 directions.
Otherwise dispatch falls back to the 64-byte wide representation. This limits
the representation, not the physical propagation range or angular resolution.
An exact face crossing and certified transparent jump cannot exceed this
support bound. Long-double ray parameters remain unchanged.

Seven alternating repetitions give:

| Case | Original median/s | Compact median/s | Original/compact |
|---|---:|---:|---:|
| Isolated R=8 cells | 0.00848 | 0.00889 | 0.955 |
| Native 32 cubed | 0.58899 | 0.54578 | 1.079 |
| Native 64 cubed | 0.61755 | 0.55202 | 1.119 |

Allocated ray storage drops exactly 25% in these tests. This is not a 25%
reduction in total process memory. Native-case timings suggest a modest
8–12% kernel speedup; the small isolated case is about 5% slower. The shared
host introduces timing noise, so the memory reduction is a stronger result
than the precise speedup. Outputs, event counts and individual source ledgers
are bitwise identical to the original kernel.

`test_speed_core.py` contains 29 passing tests, including positive-length thin
opaque voxels near a corner, exact corner ties, horizons immediately around a
face, periodic/open boundaries, random initial cavities, and both large-horizon
and large-quadrature wide-storage fallbacks.

Reproduce with:

```bash
python -m pytest research/pc_refinement/test_speed_core.py -q --confcutdir=research/pc_refinement
python research/pc_refinement/speed_benchmark.py \
  --native32 /path/to/source_capacity.npz \
  --native64 /path/to/native_fields.npz \
  --variants original compact --repeats 7 \
  --output research/pc_refinement/speed_compact_results.json
```

The `compact` variant is suitable for adoption as an exact storage optimization.
It provides a little computational headroom for improved angular or spatial
sampling; it does not itself reduce the model's approximation errors.
