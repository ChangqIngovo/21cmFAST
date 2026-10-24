# Large-box prescribed-source production run

This workflow scales the validated **fixed-gas, carried-ionization** model to
a complete prescribed source history. It is not the full coupled Qin20
simulation: the gas capacity is fixed at z=8, only ACG sources are used, and
recombinations, spin-temperature evolution and photoheating feedback remain
disabled. Source rates are generated from native 21cmFAST at every endpoint.
The refined front carries its fine ionization geometry between increments.
The ES control receives the same integrated source budgets and gas field.

The source and absorption rules are unchanged from the temporal validation.
This production workflow adds a larger configuration, separate native IC
resolution, checkpoint recovery, and one PC/ES pair instead of a convergence
matrix. It does not select the earlier `global-budget` algorithm.

## Default run

| Quantity | Value |
| --- | --- |
| Box length | 400 cMpc |
| Original source/output grid | 128 cubed |
| Native initial-condition grid | 512 cubed |
| Absorber subdivision | 4 per axis, 64 per original cell |
| Original / absorber cell width | 3.125 / 0.78125 cMpc |
| Directions per source | 768, fixed through the history |
| Seed | 1234 |
| Redshift interval | 25 to 5 |
| Native source endpoints | 321 |
| Transport increments | 80 (stride 4 over 320 source intervals) |
| Gas-density reference redshift | 8 |
| Periodic transport horizon | 8 box lengths |
| PBS request | 32 CPUs, 320 GB, hugemem, 48 hours |

The native source generator uses the allocated OpenMP threads. The refined
event kernel is serial. PC and matched ES are run sequentially; allocating
32 CPUs does not make one transport solve 32 times faster. The walltime is an
allocation, not a measured runtime prediction. The small validation results
do not establish convergence at this new volume and resolution.

The default worst-case number of rays, 128^3 times 768, is 1,610,612,736,
below the kernel's signed 32-bit ray-index limit. Compact ray storage alone
can reach 72 GiB. Other solver state, native grids and Python arrays require
additional memory. Invalid ray counts are rejected before submission.

## Prepare and submit on Gadi

Use a checkout pinned to the desired commit on `Photon_conserving`, then:

```bash
export PC_THREADS=32 PC_MEM=320GB PC_WALLTIME=48:00:00
bash research/pc_evolution/gadi_production_prepare.sh --submit
```

The preparer finds a completed compatible prior run and reuses its native
runtime, verifies that installed native sources match this checkout, archives
the tracked commit, and records runtime/compiler/source identities. It does
not generate the large initial conditions on the login node. A specific
runtime can be selected with `PC_RUNTIME_PARENT=/path/to/completed/run`.
See the script's `--help` for numerical overrides. Physics scope is not changed
by these overrides.

The scripts print the new `RUN_DIR`, which contains `production_config.json`,
`run_manifest.json`, `commit.txt`, and `submit.pbs`. Review the generated
configuration with `cat "$RUN_DIR/production_config.json"`.

## Recovery and outputs

After an interrupted job has stopped, resubmit the **same** prepared directory:

```bash
qsub -v RUN_DIR="$RUN_DIR" "$RUN_DIR/submit.pbs"
```

Completed native fixtures are reused. PC resumes from its latest durable
checkpoint with the sparse fine state and per-source photon ledgers. The
source budget, fixed capacity, numerical settings, code, runtime, and saved
checkpoint hashes are checked before recovery. Work after the last committed
checkpoint may be repeated. A stale `.running` lock from a hard scheduler kill
must only be removed after confirming that the preceding job is no longer
running. Do not edit a prepared source tree or configuration to resume it.

Results are written below `RUN_DIR/results`, including the native source
fixture, PC and ES checkpoint data, `summary.json`, `history.csv`,
`power_spectra.csv`, and the evolution comparison figure. Preserve the
checkpoint files until the run and its diagnostics finish. Native source
generation interrupted before its fixture is complete must restart that stage.

`COMPLETED` records successful execution, source-budget agreement and photon
accounting. It is not a spatial-convergence certificate. Retained photons are
reported separately and are never relabeled as absorbed or emitted again.
The very small transport-horizon and time-discretization differences seen in
the prior four-seed run are accepted for this production comparison; their
diagnostic values remain visible.

The 21-cm spectra use saturated spin temperature and no RSD. Same-redshift
PC/ES power differences include their different mean ionization histories.
The matched ES control reconstructs a sharp-k centre-cell map from the
integrated source counts; contextual native ES outputs use a different
cumulative source proxy and must not replace that control.

## Validation of this workflow

The evolution/refined-front suite passes 196 tests, including interrupted
continuation, exact restored fine-state/ledger equivalence, changed-input
rejection, stage recovery and completed-run idempotence. A real native run
with HII_DIM=8, DIM=32, 40 cMpc, 33 source endpoints from z=25 to 5, f=4 and
96 directions completed both PC and ES, the spectra and the figures. Its
maximum relative photon-ledger and matched-source residuals were below
6e-17. Resubmission preserved every completed NPZ file without emitting
additional photons. This checks the production plumbing; it is not a timing
or accuracy measurement of the default 128-cubed run.

The Gadi wrapper was exercised with the real native runtime and mocked PBS
submission/execution, including failure, retry, configuration tampering and
CPU-allocation mismatch. An actual Gadi job ID is required to establish that
a remote run has been submitted.
