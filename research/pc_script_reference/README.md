# Original SCRIPT reference on existing 21cmFAST archives

This runs the author's original, pinned SCRIPT Fortran allocation and periodic
shell-sorting routines on the native source/density snapshots already saved by
`research/pc_causal_rt`. It does not launch a new native source history or the
expensive causal RT convergence matrix. The original MIT license and source
hashes are in `vendor/`; the allocator itself is not rewritten.

**Project requirement:** 21cmFAST remains the physical model. SCRIPT is an
independent comparison, not a replacement for its source, chemistry, thermal,
recombination or feedback prescriptions. This module does not modify the
production 21cmFAST solver, parameters or any existing input archive. Its scope
guards describe the already-simplified archived experiment; they are not an
instruction to disable physics in the target model to accommodate SCRIPT.

## What is compared

Each snapshot supplies its original extensive `EmissivityFields.n_ion` and
`1 + density` in the same native effective count units. There is no additional
density multiplier, efficiency recalibration, or time integration. The native
ES field uses this coeval source convention too. SCRIPT's mean ionized *mass*
fraction should therefore agree with `min(sum(source)/sum(capacity), 1)` within
the author's single-precision numerical accuracy. Excess photons after full
ionization are recorded as a global surplus, not a resolved radiation field.

The previous causal calculation instead integrated `n_ion * H(z) / t_STAR`.
Its integrated total appears in the new report as context, never as the input
to this coeval SCRIPT comparison. The two photon budgets generally differ.

The adapter reloads the archived native HDF5 and checks it against the saved
source/density/xHI/brightness arrays. It uses that run's exact native binary to
calculate brightness for SCRIPT. No native initial conditions, galaxy parameters,
source fields or density fields are regenerated or changed. Original single-
precision SCRIPT inputs and output are audited against the double-precision
archived counts; the report records rounding and conservation residuals.

This is a coeval reference for the archived restricted L-INTEGRAL ACG baseline:
no active recombinations, thermal evolution, MCGs or photoheating feedback.
It is **not a fully coupled SCRIPT option in the production 21cmFAST driver**.
SCRIPT uses its original full periodic redistribution; native ES keeps its
own filter and `R_BUBBLE_MAX`. The resulting field/power differences are model
comparisons, not errors against an exact RT solution. There is no convergence
certificate or claim that all optional 21cmFAST physics has been integrated.

At very low emissivity, native 21cmFAST has an early-exit branch that writes
RECFAST residual ionization. In this no-spin-temperature configuration its
regular ES branch does not carry that residual forward. Raw SCRIPT here counts
only the archived stellar photons. Consequently the early native
`source - ions` diagnostic is not an isolated stellar photon-loss measurement;
residual electrons are not added to SCRIPT as artificial stellar photons.

## Gadi

After updating this branch on Gadi:

```bash
PC_INPUT_RUN=/g/data/qp00/sz4640/21cmfast_runs/causal_rt_5c1b6d8719_20261029T121626_GyH1 \
PC_MEM=4GB PC_WALLTIME=01:00:00 \
bash research/pc_script_reference/gadi_prepare.sh --submit
```

The launcher freezes the current commit, reuses the input run's Python and
native build, and adds `gcc/12.2.0` for the original Fortran routines. It requires
no package download on compute nodes. CPU allocation matches the archived
native thread count (eight for the run above). The one-hour request is a limit,
not a measured Gadi duration. `SCRIPT_FC` can select an available compiler;
`PC_GFORTRAN_MODULE` selects another module if necessary.

Output goes to a new `script_reference_*` directory. `resubmit.sh` continues
that exact frozen run, reusing checksum-verified completed snapshot results.
The input run remains read-only. Defaults include every original checkpoint
and the last redshift; they do not request angular/spatial refinement ladders.

Read `results/script_report.json` and `results/script_comparison.png`. Per-node
`fields.npz` contains original inputs/ES outputs and the new SCRIPT xHI and
native brightness. JSON includes mass/volume histories, a photon ledger,
neutral-field MAE and shell auto/cross power spectra with actual mode counts.
Power ratios are `P_native_ES / P_SCRIPT`; undefined reference power is null.
`COMPLETED` means execution and declared single-precision ledger checks passed,
not convergence or agreement with exact radiative transfer.

## Local execution and checks

With the original native runtime selected and `gfortran` available:

```bash
python -m research.pc_script_reference.run --input-run /path/to/existing/run \
    --output /path/to/new/reference/results
python -m pytest research/pc_script_reference -q \
    --confcutdir=research/pc_script_reference --import-mode=importlib
```

`--input-run` also accepts a direct existing results directory for local checks.
Native source hashes, native binary identity and saved result receipts must
match before a snapshot is reused. A changed source or implementation requires
a fresh output directory. A stalled upstream kernel is terminated by a per-
snapshot timeout instead of blocking the job indefinitely.

Local validation on two archived 8³ histories passed 42 snapshot ledgers, with
maximum relative photon residual 7.17e-9. The audit/allocation/native-brightness
loop took 2.70 seconds on the local machine; this excludes Python startup and
plotting, and is not a Gadi timing. A complete frozen PBS shell run also passed
locally on three checkpoints. See `local_validation.json` for exact measurements.
