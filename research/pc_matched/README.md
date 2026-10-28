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
threads; the RT kernel is serial and cases execute sequentially in this pilot.

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
- `results/matched_comparison.png`: histories and physical comparisons.
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
