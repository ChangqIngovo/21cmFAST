# Accuracy refinements for the 21cmFAST research photon-conserving adapter

Use `upgraded.solve(source, capacity, initial_ionized, profile=...)` with counts
per cell, or select a profile in the native restricted adapter:

```bash
python research/pc_reconsidered/native_driver.py \
  --solver balanced --dimension 32 --box-len 200 --redshift 8 \
  --threads 4 --output /path/to/new/results
```

| Profile | Intended use | Change |
|---|---|---|
| `balanced` | Fresh coeval source field | Remaining-neutral radius, 1.25× angular samples, compact rays |
| `incremental` | New photons in existing cavities | 48 geometry probes, RMS radius, surface factor 16, dominant-source guard, compact rays |
| `compact` | Exact old-model comparison | Original angular allocation, smaller ray records only |
| `original` | Native CLI baseline | Previous adaptive event solver |

The event kernel remains serial. Native 21cmFAST fields use the requested thread
count. `warmup(profile)` excludes one-time C++ compilation from benchmark timings.

This is a static, infinite-light-speed, opaque-voxel approximation. All photon
counts are retained in absorbed, escaped or range-limited ledgers; conservation
does not certify continuum spatial accuracy. The native adapter rejects physics
it does not couple, including full Qin20 heating/recombination/feedback.

See [REPORT_zh.md](REPORT_zh.md) for measured accuracy/cost, negative results and
remaining work; [README_REPRODUCE.md](README_REPRODUCE.md) for reproducible runs.
Raw grid fixtures are external inputs and are not committed to git.
