# Analytic absorption on shared native 21cmFAST inputs

This experiment compares a directional analytic-absorption model, the author's
unmodified SCRIPT allocation kernel, and the native CPU short-characteristics
kernel from pyC2Ray. It preserves the archived native cumulative `n_ion` and
`1 + density` arrays. It does not replace 21cmFAST's production ionization driver.

## Parallel teacher simulations

`campaign.py` generates new native 21cmFAST realizations and analytic-PC teacher
fields. It does not invoke the reference solvers or train a neural network.
Each independent seed runs in a spawned process with a private native cache;
redshifts for that seed reuse its initial conditions. Each analytic solve also
uses OpenMP. `workers * threads` must fit the allocated CPUs on one node.
Fixed-size reduction blocks, photon injection and radiation retirement now run
in parallel, with a deterministic block-merge order independent of thread count.
The absorption law, ray geometry, cross-section and numerical tolerances are
unchanged.

The campaign clones the parent's complete native input template, changing only
the requested seed, box/grid dimensions and thread count. Cosmology, source
physics, astrophysical parameters and all physics switches are checked for
exact equality. The default first campaign contains 8 independent seeds and
7 redshifts (56 labels), with native low-resolution N=32, high-resolution N=256,
and L=256 cMpc. The 1-cMpc high-resolution density grid retains the existing
2LPT density evolution; only the teacher output grid is small. These are new
simulations, not coarsened copies of seed 222.
Each redshift is an independent cumulative-budget coeval solution; it is not a
time-evolution step. The physical scope documented below remains in force.

```bash
(
    set -euo pipefail
    cd /g/data/qp00/sz4640/21cmFAST
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation

    PC_WORKERS=4 PC_THREADS=4 PC_NCPUS=16 \
    PC_DIMENSION=32 PC_HIGHRES_DIMENSION=256 PC_BOX_LEN=256 \
    PC_SEEDS=1001,1002,1003,1004,1005,1006,1007,1008 \
    PC_REDSHIFTS=10,9,8,7.5,7,6.5,6 \
    PC_DIRECTIONS=96 PC_EXPOSURE_STEPS=32 \
    PC_MEM=64GB PC_MEMORY_GB=48 PC_WALLTIME=12:00:00 \
    bash research/pc_analytic/gadi_teacher_prepare.sh \
      --parent /g/data/qp00/sz4640/21cmfast_runs/pc_compare_873f8b6152_20261031T110335_IuGe/causal_rt_873f8b6152_20261031T110336_JIAc \
      --submit
)
```

The launcher prints the new `analytic_teacher_...` directory and job ID. It
freezes committed source, the original native runtime and the full physical
template. `bash /absolute/new/run/resubmit.sh` resubmits the unchanged campaign:
completed native snapshots and teacher labels are checksum-verified and reused.
Changing code or configuration requires a new run directory. Failed samples
never receive passing receipts. No network or reference-backend build is needed
on the compute node.

The output `results/dataset_manifest.json` indexes the exact seed/redshift set.
Each `results/seeds/seedNNNN/teacher/z.../fields.npz` stores source and atom
arrays, `teacher_xHI`, ionized atoms, remaining photons, escaped photons by
direction, unemitted photons, native ES and both native/teacher brightness
fields. `remaining_photons` is summed over direction; these coeval labels are
not radiation restart states. Original native HDF5 snapshots remain alongside
them. Every label must finish, remain finite and bounded, preserve its source
inputs, and pass an independent field-derived photon ledger at relative
residual <=1e-10. Completion is recorded only when every requested sample passes.

The first campaign varies realization and redshift at fixed astrophysical
parameters. It supplies an initial teacher dataset; later parameter coverage
must be explicit. When training starts, split by seed to keep redshifts from
the same initial conditions out of both training and held-out sets.

For a local campaign, provide a full native `InputParameters.asdict()` JSON
template in `native_template_json`, plus the settings in `campaign.DEFAULTS`:

```bash
python -m research.pc_analytic.campaign \
  --config /absolute/teacher_config.json --output /absolute/new/teacher_results
```

The selected Python environment must contain the matching native 21cmFAST
runtime and the compiler/shared-library dependencies. Native input serialization
and extension identity are checked before generating labels.

## Absorption and model scope

For a constant-area ray tube with initial neutral column `C`, incident photon
fluence `F`, and absorption cross-section `sigma`, the exact no-recombination
single-ray response is

```text
F_out = log[1 + exp(-sigma*C) * (exp(sigma*F) - 1)] / sigma
absorbed = F - F_out
C_new = C - absorbed
```

This includes the decreasing opacity as atoms become ionized. It reduces to
Beer-Lambert absorption for a weak exposure and to `min(photons, atoms)` in the
opaque limit. It is a known saturable-absorption response, not a claim of a new
fundamental absorption law. Extending it to a coarse cell with several incoming
directions requires a shared neutral-state closure. The characteristic transport
mode follows fixed ray directions to actual voxel faces, using each crossed path
for its optical depth. This avoids spreading a
partly transmitted beam across neighbouring cells by trilinear grid translation.
The solver records the actual angular quadrature, transport discretization and
exposure settings. A local analytic update does not make the three-dimensional
field analytically exact.

The comparison freezes the density field and starts fully neutral. There is no
recombination, collisional ionization, thermal/spin-temperature evolution,
photoheating feedback, frequency evolution or separate helium chemistry.
`PHOTON_CONS_TYPE=no-photoncons` is required for the native ES control. The
native source-model and switches are validated from each archived receipt.

The existing native effective-atom convention is retained: the count unit is
the mean number of **H+He nuclei** in an original cell, and every effective atom
has hydrogen opacity. Both photon and atom arrays receive the same conversion
factor. Density is computed by the existing `pc_matched.workflow.opacity_parameters`
convention, including its constants, rather than changing to the smaller true-H
number density. This is not a separate H/He chemistry calculation.

The cross-section is an explicit shared input for analytic absorption and
C²-Ray. Cumulative `n_ion` alone does not provide a spectrum. SCRIPT consumes
only photon and atom budgets; it has no cross-section or optical-depth equation.
Thus these runs test the consequences of different propagation/allocation
rules on common physical inputs.

## Reference identity and boundary conditions

SCRIPT retains the pinned original Fortran bytes and REAL32 arithmetic. Its
executed-input and original-input ledgers are both reported.

C²-Ray uses the actual pinned pyC2Ray CPU kernel. Its adapter records precisely
which source and physics configuration is used; it never substitutes a second
implementation of our analytic formula. Recombination/collisional coefficients
are zero and monochromatic transport is explicitly selected in the build
provenance. See `c2ray_reference.py` and its generated build manifest.

The native C²-Ray geometry traces one source-centred periodic density domain
and does **not** recycle radiation leaving that tracing domain. Use analytic
`boundary: "source_periodic_cube"` to match this intended domain. The analytic
`"periodic"` mode instead recycles photons; SCRIPT always closes a periodic
photon box. The report states which pair shares the intended boundary rather
than declaring all three identical.

C²-Ray's native subbox optimization stops a source trace when its surface-loss
counter falls below `loss_fraction` (default `1e-10`). This is a numerical
truncation control. Its raw counter includes both inner-subbox termination and
outer-domain outflow and is not an independent escaped-photon measurement.
The report retains this counter and field-derived ionization separately;
missing independent escape counters remain `null`. It never declares that
`emitted - ionized` is measured escape, rescales C²-Ray to the SCRIPT mean, or
treats a transport difference as a known truth error. A negligible outflow
counter by itself does not establish a correct photon budget.

## Input and execution

Use the original native runtime of the selected archive, including its shared
library environment. `brightness: "native"` reopens `native.h5`, re-extracts the
source/atom fields and checks the native extension checksum before generating
21-cm brightness with the original native routine.

```bash
python -m research.pc_analytic.compare \
  --config /absolute/experiment_config.json \
  --output /absolute/new/results
```

C²-Ray is required by default. `--allow-missing-c2ray` is an explicit two-method
diagnostic fallback; the report then records `three_methods_completed=false`.

Configuration can select an archive using `input_run`, `redshifts`, and `seeds`,
or provide `snapshot_directories` containing the original `source.npz`,
`native.h5`, `native.json`, and `receipt.json`. Every snapshot is checksum
verified. `example_config.json` illustrates an explicit coarse pilot of the
existing Gadi archive. The C²-Ray upstream checkout and compilers must be
available locally; compute jobs need not access the network.

An unchanged run can reuse completed snapshot receipts after verifying their
identity and saved field/metric checksums. A changed implementation or setting
requires a fresh comparison output directory. Incomplete native attempts are
not accepted as finished results.

For repeated analytic resolution tests, set `c2ray_cache_directory` to a shared
absolute cache path. C²-Ray results are reused only if the source/capacity bytes,
all effective backend parameters (including defaults), pinned source/build
identity and backend Python code match. Cached arrays, metadata and the original
native binary output are checksum-verified. Failed attempts remain separate and
never acquire a completion receipt. Cache hits report zero native execution time
for that call and retain the original solve time separately; they are not counted
as a faster C²-Ray calculation.

`brightness: "none"` skips 21-cm power. The optional `"saturated_proxy"` uses
`27 sqrt((1+z)/10) (1+delta) xHI mK` for all methods, clearly labelled as an
approximation. Neither mode is silently selected when the native runtime fails.

For an explicitly requested `coarsen_to`, source and capacity are block means
in **coarse-cell count units**. The larger cell's physical conversion factor
restores the exact photon and atom totals. Native ES is atom-weighted when
restricted. For native 21-cm brightness, coarse model neutral fractions are
prolonged over the original gas, passed through the original native brightness
routine, and volume-averaged back to the tested grid. The native ES brightness
is volume-averaged directly. This is a comparison on filtered inputs, not a new
native simulation or evidence of convergence on the original fine grid.

## Outputs and interpretation

Each snapshot has `fields.npz`, `metrics.json`, and a checksum receipt. The
run writes `identity.json`, `comparison_report.json`, and `comparison.png`.
Source and capacity hashes, native input parameters, count-unit conversion,
solver settings and reference provenance remain attached to the result.

The report includes:

- Independently field-derived ionized atoms, remaining radiation and escape
  where available; surplus inferred by SCRIPT is explicitly identified.
- Volume/mass neutral fractions, same-snapshot field MAE, cross-correlation and
  ionized/neutral IoU. Mean differences are reported, not removed by field
  normalization. No matched-mean interpolation is performed.
- Neutral-field and common-definition 21-cm auto/cross power with shell mode
  counts and undefined-reference-power masks. Half-Nyquist is a diagnostic
  label, not an accuracy certificate.
- Geometric first-exit MFP distributions for both phases, with common random
  draws, empty-phase handling and distance-cap censoring. This morphology
  statistic is distinct from the optical absorption mean free path.
- Backend timing with its scope; compilation and executable IO can enter the
  measured call, so small-grid times are not pure-kernel speedups.

`COMPLETED` means the requested backend calculations finished. It does not
certify angular, exposure, spatial or morphology convergence. Exposure-dose,
angular-resolution and grid-refinement comparisons must use the same source
and density realization.

## Targeted checks

```bash
uv run pytest -q research/pc_analytic/test_absorption.py \
  research/pc_analytic/test_solver.py research/pc_analytic/test_compare.py
uv run ruff check research/pc_analytic
```

## Gadi pilot

The launcher creates a frozen source archive, verifies the input runtime and
receipts, and retains all three backends' results. It downloads the pinned C²-Ray
source on the login host when necessary; the compute job needs no network.

```bash
cd /g/data/qp00/sz4640/21cmFAST
git switch matched-photon-conservation
git pull --ff-only origin matched-photon-conservation

PC_COARSEN_TO=16 PC_SEEDS=222 PC_REDSHIFTS=8,7.5 PC_THREADS=8 \
PC_DIRECTIONS=96 PC_EXPOSURE_STEPS=32 PC_C2_EXPOSURE_STEPS=32 \
PC_MEM=32GB PC_WALLTIME=04:00:00 \
bash research/pc_analytic/gadi_prepare.sh \
  --input-run /g/data/qp00/sz4640/21cmfast_runs/pc_compare_873f8b6152_20261031T110335_IuGe/causal_rt_873f8b6152_20261031T110336_JIAc \
  --submit
```

This explicitly compares all methods on common **16³ filtered arrays** while
retaining the archive's box size. It is a small reference pilot, not a
high-resolution 256-cMpc validation. Omitting `PC_COARSEN_TO` uses an existing
native N≤16 archive directly; larger grids require an explicit target such as
`PC_COARSEN_TO=32`, `64`, or the native dimension. Local validation results and
their scope are documented in `VALIDATION.md`.

## Fixed-input grid verification on Gadi

The completed 256-cMpc N16 comparison at z=7.5 has a maximum analytic/C²-Ray
21-cm power difference of 9.69% in its three shells below half Nyquist. To
measure grid sensitivity, rerun **all three methods** at N32 using the same
original N128 source/atom archive. Keep the cross-section, source exposure,
angular quadrature and boundary settings fixed. The existing N16 calculation
is reused without editing its frozen source or results.

```bash
(
    set -euo pipefail
    cd /g/data/qp00/sz4640/21cmFAST
    git switch matched-photon-conservation
    git pull --ff-only origin matched-photon-conservation

    PC_GRID_REFERENCE_RUN=/g/data/qp00/sz4640/21cmfast_runs/analytic_pc_01aa674c8b_20261104T035222_bZTV \
    PC_COARSEN_TO=32 PC_SEEDS=222 PC_REDSHIFTS=7.5 PC_THREADS=8 \
    PC_DIRECTIONS=96 PC_EXPOSURE_STEPS=32 PC_C2_EXPOSURE_STEPS=32 \
    PC_BOUNDARY=source_periodic_cube PC_SIGMA_CM2=2.7173135802469135e-18 \
    PC_C2_TOLERANCE=1e-6 PC_C2_MAX_ITERATIONS=2000 \
    PC_C2_SUBBOX_SIZE=2 PC_C2_LOSS_FRACTION=1e-10 \
    PC_C2_TIMEOUT=72000 PC_SCRIPT_TIMEOUT=900 \
    PC_MEM=32GB PC_WALLTIME=24:00:00 PC_QUEUE=normal \
    bash research/pc_analytic/gadi_prepare.sh \
      --input-run /g/data/qp00/sz4640/21cmfast_runs/pc_compare_873f8b6152_20261031T110335_IuGe/causal_rt_873f8b6152_20261031T110336_JIAc \
      --submit
)
```

`PC_C2_TIMEOUT` is a per-snapshot backend limit in seconds, separate from PBS
walltime. Its ordinary default remains 1800 seconds. The 72000-second limit
allows this explicitly requested larger reference calculation to run for up to
20 hours. Actual C²-Ray CPU execution remains serial in this backend.

The job automatically writes `grid_comparison/grid_report.json` and
`grid_comparison/grid_comparison.png` beneath the new run directory after all
three fine-grid methods complete. The reference report identity is frozen at
preparation, and the analysis verifies both runs' field receipts, original
input hashes and physical/numerical controls before comparing them.

The primary comparison maps N32 results onto N16 using atom-weighted neutral
fractions and volume-weighted brightness temperatures. It uses identical FFT
shells and mode counts, with no adjustment of the mean neutral fraction.
It reports each method's own grid change and the analytic/reference difference
at both resolutions. The first diagnostic band is
`k < pi / (2 * 16 cMpc) = 0.09817 cMpc^-1`; higher-k shells are retained and
labelled separately. This is a fixed-input grid-sensitivity test: refinement
also restores source and density substructure. Two grids alone do not establish
convergence or isolate propagation error.

Existing completed runs can also be analysed without rerunning any solver:

```bash
python -m research.pc_analytic.grid_compare \
  --coarse-run /absolute/coarse_run \
  --fine-run /absolute/fine_run \
  --redshifts 7.5 --output /absolute/new/grid_comparison
```
