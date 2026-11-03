# Same-input coeval PC / SCRIPT comparison

This module replaces the **comparison experiment**, not the causal RT solver.
It retains the archived 21cmFAST source, density and brightness physics, while
explicitly replacing finite-speed propagation through an evolving universe with
instantaneous redistribution in a frozen coeval box. It is not a full RT model.

## What changed

- New independent C++/OpenMP allocator: all active source cells request atoms in
  increasing periodic distance shells. Competing requests are accepted
  simultaneously without exceeding a cell's remaining atom capacity. Rejected
  photons remain assigned to their original emitter for the next round.
- Equal-distance targets share according to available capacity. No angular
  sampling, CFL step or cosmic-time cutoff enters this coeval model. There is
  no global rescaling to fit SCRIPT or a target reionization history.
- Each source remembers the last shell it reached. Fully claimed inner shells
  have no atoms left, so later rounds skip them exactly, without dropping weak
  sources or changing the allocation rule.
- The new allocator and the author's unmodified, pinned SCRIPT Fortran kernel
  receive the **same raw native cumulative `n_ion` and `1+density` arrays**.
  There is no extra source-density factor, rate integral, bounded envelope,
  re-generated density realization, or retuned astrophysical parameter.
- Native ES remains a control, with `PHOTON_CONS_TYPE=no-photoncons`: the Park
  redshift correction is disabled and rejected by the input adapter if enabled.
- Each snapshot is validated against its native H5 and receipt. The runner
  streams individual snapshots and never reads the enormous `history.npz`.
- Every completed snapshot is saved with checksums and can be reused after a
  failed job. PBS logs and temporary compiler products stay outside frozen code.

## What this comparison can establish

For both completed PC allocators, in this no-recombination scope,

`mass_mean_xHI = max(0, 1 - total_source_photons / total_atom_capacity)`.

Their mass-mean histories must therefore agree within arithmetic/allocation
tolerances. This is a budget test, **not a test of morphology or physical truth**.
Volume means can differ even when mass means agree. SCRIPT is an independent
allocation reference, not an exact solution. ES need not always finish later:
its net photon discrepancy depends on the input and the stage of reionization.

The report separately saves field MAE, mass-weighted MAE, ionized/neutral overlap,
field correlation, native 21-cm auto/cross power, mode counts, and zero-power
validity masks. A half-Nyquist diagnostic mask is provided; it is not an accuracy
guarantee. Neither one grid nor agreement with SCRIPT certifies spatial
convergence. Both models still need comparisons on successively refined grids.

The native adapter currently requires L-INTEGRAL, ACG only, no MCGs, no
recombinations, no thermal/spin-temperature evolution and no photoheating
feedback, matching the existing experiment. It reuses the original native
brightness routine and archived binary. Native early-time ES may include
RECFAST residual ions outside the stellar photon budget.

## Completion and cost

The allocator preserves `photons = ions + remaining_source_photons`, and checks
atom and source bounds independently. Once all atoms are ionized, photons left
over are recorded as excess. Otherwise, remaining photons are unresolved and
must fall below the requested relative photon tolerance. A work/iteration limit
produces an explicit incomplete result, never a completed comparison receipt.

Memory is proportional to cell count, not to all source-target pairs. Runtime
can still grow steeply near percolation when many sources search for distant
neutral capacity. The default 1,000,000,000,000 interaction cap protects the
large comparison; inspect `incomplete_pc.json` before increasing it. These
limits are operational bounds, not changes to physical parameters.

## Gadi usage

From a clean, committed checkout containing this module:

```bash
PC_THREADS=8 PC_MEM=32GB PC_WALLTIME=04:00:00 \
bash research/pc_coeval/gadi_prepare.sh --input-run /absolute/original/input_run --submit
```

The input is the original native-source run, containing `results/histories`,
`native_build`, `native_runtime`, `runtime.env` and `python_path.txt`; it is not
the later `causal_budget_*` result folder. No native simulation is regenerated.

Defaults select z=10,9,8,7.5,7.25,7,6.75,6.5,6.375,6. Every requested redshift must
exist exactly in the archive. Override using `PC_REDSHIFTS=8,7.5,7,6.5,6` if needed.
Other overrides: `PC_RUN_ROOT`, `PC_QUEUE`, `PC_SCRIPT_TIMEOUT`, `PC_TOLERANCE`,
`PC_MAX_ITERATIONS`, `PC_MAX_INTERACTIONS`. Change settings by preparing a new run.

The preparation command prints the new run directory. Resubmit that same run:

```bash
bash /absolute/new_run/resubmit.sh
```

Results include `results/coeval_report.json`, `results/coeval_comparison.png`,
per-snapshot `fields.npz`, `metrics.json`, and `receipt.json`. `COMPLETED` appears
only after every requested snapshot, final report and frozen-source check pass.
Preserve previous causal RT results as a distinct finite-speed model.

## Local verification

```bash
python -m pytest research/pc_coeval/test_solver.py research/pc_coeval/test_prepare.py
python -m unittest research.pc_coeval.test_run
```

An end-to-end run additionally needs the input archive's original native Python
environment and C++/Fortran compilers. Original SCRIPT retains its MIT license
and pinned upstream bytes in `research/pc_script_reference/vendor`.
