# Direct radiation transfer with native 21cmFAST sources

This branch tests a direct radiation-transfer teacher while retaining a shared
21cmFAST source prescription. The teacher does **not** call excursion-set
flagging, the previous opaque-front solver, `global-budget`, or C2-Ray. ES is
computed separately as a comparison.

## Equations and physical scope

`rt_kernel.cpp` evolves angular photon fields on a periodic Cartesian grid with
positive, conservative finite-volume fluxes. Photons travel at the physical
speed of light. In comoving coordinates, the solved monochromatic equations are

\[
\partial_t q_m + (c/a)\,\mathbf n_m\cdot\nabla q_m
=w_m\dot S-c\sigma n_{0}a^{-3}(C-I)q_m,
\qquad
\partial_t I=c\sigma n_{0}a^{-3}(C-I)\sum_m q_m.
\]

Here `C=1+delta` is the fixed native effective atom capacity, `I` is its ionized
count, and `q_m` is the photon count in one direction, in the same mean-cell
units. Local absorption is integrated with the analytic coupled photon/neutral
solution. Transport uses a first-order upwind flux and explicit CFL substeps;
the source history is linearly interpolated and integrated exactly between the
supplied endpoints. Expansion changes propagation speed in comoving coordinates
and the proper absorber density. Photon frequency redshifting is not included.

The first experiment deliberately uses the existing restricted native
L-INTEGRAL, atomic-cooling-galaxy source model: recombinations, helium chemistry,
X-ray/spin evolution and photoheating feedback are disabled. Transport density
contrast is frozen at z=8 in both controlled arms; source fields evolve. This is
not the complete Qin20 configuration or a multifrequency thermochemical solver.
The native effective atom convention is kept unchanged in source and capacity
arrays; the native present-day H+He nuclei number density sets the effective
single-absorber opacity. The approximation and constants are recorded in every
RT case. No extra hydrogen fraction or escape fraction multiplies the sources.

The global numerical identity is

\[
N_{\gamma,\rm emitted}+N_{\gamma,\rm initial}
=N_{\rm ionized}-N_{\rm ionized,initial}+N_{\gamma,\rm in\ flight}.
\]

Periodic boundaries have no escaping flux. Unabsorbed photons remain in their
angular radiation fields and propagate in the next interval. They are never
converted to ions by a global rescaling. This radiation inventory is physical
within the model, not a finite-distance remainder to be discarded.

## Controlled comparisons

1. **Direct RT:** the equations above with actual native source rates.
2. **Matched ES:** the same integrated source counts and frozen atom field,
   with the existing sharp-k centre-cell threshold. This isolates differences
   in the ionization/transport treatment, including finite propagation time.
3. **Native ES context:** original native coeval outputs. Their own evolving
   density is exported in a hashed sidecar and used for their brightness field.
   They use the native cumulative source proxy, so this third comparison also
   contains source-history and density differences.

All brightness spectra assume saturated spin temperature and omit RSD. The
report retains the physical k grid, raw per-bin errors, undefined zero-power
bins, and field errors. Errors between matched ES and RT are not NN errors.

## Small run

Defaults are two independent initial-condition seeds (111, 222), 50 cMpc, an
8-cubed source grid, and 41 snapshots from z=20 through z=4. There are five RT
histories per seed:

| Directions | Grid refinement | CFL | Purpose |
| --- | --- | --- | --- |
| 6 | 1 | 0.4 | Angular comparison |
| 26 | 1 | 0.4 | Common baseline |
| 26 | 2 | 0.4 | Spatial comparison |
| 26 | 1 | 0.2 | Time-step comparison |
| 26 | 1 | 0.1 | Second time-step comparison |

Subdivided cells inherit the same piecewise-constant gas and source density.
Fine-grid counts are conservatively restricted before field comparisons.
Six and 26 directions are positive symmetric quadratures, not an angular
convergence proof. The small box/grid is a numerical pilot, not a certified
1% cosmological teacher. `convergence_certified=false` is explicit in exports;
passing photon ledgers does not override this flag. No NN is trained here.

On Gadi, from a clean committed checkout:

```bash
bash research/pc_matched/gadi_prepare.sh --submit
```

The script discovers a completed native runtime from earlier runs. To choose
one explicitly, set `PC_RUNTIME_PARENT`, or supply `PC_PYTHON` and, when needed,
`PC_RUNTIME_ENV`. Discovery uses a standalone Python 3.6-compatible standard
library entry point, so the login node does not need the scientific environment
loaded first. After loading the recorded modules, preparation checks that the
selected native Python is at least 3.12. The preparation freezes the branch
commit and the existing dependency environment. If the installed native sources
differ from the frozen checkout, the PBS job builds an isolated CFFI extension
from a copy of that checkout before generating any histories. It reuses Python
dependencies, requires no network access, and leaves the existing installation
and frozen source archive unchanged. The imported scientific package must then
match every archived native source file; its compiled extension is hashed and
verified before and after the calculation. Default PBS
resources are 16 CPUs, 64 GB, 12 hours on `normal`, project `qp00`. The walltime is
a request limit, not a runtime estimate. Native generation uses the allocated
threads; the RT kernel uses the same allocated thread count through OpenMP.
Cases execute sequentially so their memory allocations do not overlap.

## Successive refinement run

Use the opt-in convergence profile for the next numerical study:

```bash
bash research/pc_matched/gadi_prepare.sh --convergence --submit
```

It retains the same two seeds, 50 cMpc box, native 8-cubed source grid, fixed
density contrast and restricted physical model. Each seed generates one finest
native source history with 160 intervals from z=20 to 4. The nine RT histories
per seed isolate four numerical axes:

| Axis | Levels | Other settings held fixed |
| --- | --- | --- |
| Angular | 24, 96, 384 directions | 32-cubed RT grid, CFL 0.4, 160 source intervals |
| Spatial | 8-, 16-, 32-cubed RT grids | 384 directions, CFL 0.4, 160 source intervals |
| Transport time step | CFL 0.4, 0.2, 0.1 | 32-cubed grid, 384 directions, 160 source intervals |
| Source history | 40, 80, 160 emissivity intervals | 32-cubed grid, 384 directions, CFL 0.1 |

Shared configurations are computed once, giving 18 histories in total. The
jointly finest reference uses 32-cubed transport, 384 directions, CFL 0.1 and
all native source samples. Angular weights are positive exact solid angles of
cube-face cells, with directions through their centres; 24/96/384 form a common
refinement family. Legacy 6/26 quadratures remain available for the pilot.

Source-history arms all evolve on the identical 161-node time/redshift grid.
Only the sampled emissivities change: every fourth/second/all native endpoint
is retained and interpolated in physical time onto the common timeline. This
keeps background evolution fixed and isolates source sampling from transport
time steps. Spatial subdivision repeats the same source and gas densities;
it does not add unresolved cosmological density or source modes.

The convergence profile requests 8 CPUs, 32 GB and 24 hours by default. Explicit
`PC_*` values override the selected profile. Preparation prints and freezes the
case count and an RT-array memory estimate; the latter excludes the native
generator, libraries and compression buffers. A timing estimate is not a
convergence result or a guaranteed Gadi runtime.

Full directional radiation and ionization state are checkpointed after the
first interval, every eight intervals and at the final interval. The frozen
run's `resubmit.sh` resumes the last verified checkpoint and reuses completed
cases. Hash mismatches fail rather than silently mixing calculations.

`matched_summary.json` and `self_convergence.png` show adjacent-level RT
comparisons, with raw errors in every primary power bin. Passing these sampled
checks does not automatically certify production training labels: source-grid,
box-size and remaining discretization convergence must still be established.
The report also compares ES and RT at mean xHI targets 0.8, 0.6, 0.4 and 0.2 by
interpolating bracketed neutral-fraction fields before computing spectra. Both
fields use the same 27 mK brightness normalization. Unreached targets are
reported as unavailable, without extrapolation. These interpolated fields are
diagnostics, not RT solutions or training labels.

For a local actual-native run with a compatible installed environment:

```bash
python -m research.pc_matched.workflow run --root /fresh/output --threads 2
```

Existing native histories can be supplied with `--fixtures`, one per configured
seed. Their grid, timeline, source model, density reference and hashes are
checked. Repeating an identical run verifies and reuses completed cases;
incomplete cases are preserved before restart. Changed inputs require a fresh
run. A failed ledger blocks completion. Missing active-reionization samples do
not count as successful accuracy checks.

## Outputs

- `results/matched_report.json`: independent ledger checks, field histories,
  ES/RT spectra and numerical sensitivity tests.
- `results/matched_summary.json`: compact self-refinement and matched-mean-xHI
  diagnostics for uploading or inspecting without the large checkpoint report.
- `results/matched_comparison.png`: histories and physical comparisons.
- `results/self_convergence.png`: separate spatial, angular, transport-time and
  source-history refinement comparisons.
- `results/cases/*/fields.npz`: ionized atoms, radiation inventory and emitted
  photons at every stored snapshot.
- `results/pairs/`: matched ES/direct-RT fields and an uncertified manifest.
- `results/fixtures/`: native rates and separate native ES density/radius data.
- `native_build/`: isolated native build and compilation log, when required.
- `resubmit.sh`: repeat the frozen job safely after an interruption.

## Tests

```bash
uv run --no-project --python /path/to/python python -m pytest \
  research/pc_matched -q --confcutdir=research/pc_matched
```

The tests cover analytic local absorption, angular symmetry, transparent
periodic transport, positivity, photon/ion balance, independent report audits,
native sidecar identity, and restart/runtime integrity. Real-data measured
results are recorded separately after execution.
