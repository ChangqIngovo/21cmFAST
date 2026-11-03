# Analytic absorption on shared native 21cmFAST inputs

This experiment compares a directional analytic-absorption model, the author's
unmodified SCRIPT allocation kernel, and the native CPU short-characteristics
kernel from pyC2Ray. It preserves the archived native cumulative `n_ion` and
`1 + density` arrays. It does not replace 21cmFAST's production ionization driver.

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
native N≤16 archive directly; larger native grids are rejected by this pilot
launcher rather than silently downsampled. Local validation results and their
scope are documented in `VALIDATION.md`.
