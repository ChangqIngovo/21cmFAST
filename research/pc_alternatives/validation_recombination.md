# Recombination and UVB coupling audit

The event solver remains unchanged. These are independent analytical, graph-path,
and homogeneous-cell ODE probes, **not a Qin20 cosmological validation**. The
result is mixed: a useful local chemistry correction is inexpensive and verified,
but simply adding recombinations to event capacity does not provide a complete
UVB closure. Its time truncation error and its opacity approximation must be
tested separately.

Reproduce from the repository root with:

```bash
python research/pc_alternatives/validation_recombination.py
python -m pytest -q research/pc_alternatives/test_validation_recombination.py
```

`validation_recombination.json` records all inputs, outputs, ledgers, source-file
hashes and the microbenchmark. The seven tests passed. The mathematical event
reference uses prescribed ordered paths and an exhaustive event scan from
`event_front_audit.py`; it does not share the candidate's C++ heap or 3D DDA.

## What Qin20 actually couples

The repository's `src/py21cmfast/templates/Qin20.toml` selects inhomogeneous
recombinations, `CELL_RECOMB=false`, temperature fluctuations, and reionization
photoheating feedback. Keeping those options is essential to a Qin20 comparison.

The native ES path in `src/py21cmfast/src/IonisationBox.c` does the following:

1. `find_ionised_regions` reads the previous cumulative recombination field,
   filters it when appropriate, divides by the local filtered density factor,
   and uses it in the ionization barrier.
2. At the first/largest successful barrier radius, it estimates
   `ionisation_rate_G12` from radius times filtered emissivity and a redshift,
   spectrum and units prefactor. This is an ES UVB closure, not an attenuation
   calculation along each event ray.
3. `set_recombination_rates` evaluates the current UVB-dependent table at an
   effective redshift encoding the cell density, multiplies by the time interval
   and ionized fraction, and adds this to the cumulative recombinations.
4. Previous `Gamma12` and `z_reion` enter the subsequent source photoheating
   threshold through `compute_mturns` / `reionization_feedback` in
   `thermochem.c`. Thus changing Gamma changes future sources as well as sinks.

`recombinations.c` constructs a Miralda-Escudé density-PDF integral, with
photoionization-equilibrium hydrogen fractions and a self-shielding fit. The
table uses case-B recombination at **10,000 K**, even though Qin20 also evolves
large-scale temperature fields. Below its minimum Gamma, including Gamma=0,
the table returns zero. That is an equilibrium subgrid sink closure; it is not
the non-equilibrium decay law of a recently ionized parcel after its source
switches off. The source-off test below does **not** claim that the entire native
ES algorithm evolves according to a zero-sink parcel ODE.

The native source adapter currently establishes an effective native photon/atom
normalization. A first-principles hydrogen/helium conversion, including the
normalization of the cumulative recombination field, remains a separate
acceptance item. These toy tests use explicitly defined pure-H atom counts and
do not resolve that issue.

## Test 1: Strömgren growth separates time error from geometry

For a constant source in static uniform gas with a fully ionized interior,

\[
\dot N=Q-N/t_{\rm rec},\qquad
N(t)=Qt_{\rm rec}(1-e^{-t/t_{\rm rec}}).
\]

Here N is ionized **volume times hydrogen density**. It is not a homogeneous
parcel fraction: a parcel recombines with an x² sink, whereas a fully ionized
volume has a sink proportional to its volume. We test the volume equation with
Q=1, t_rec=1, t=2. The exact answer is N=0.8646647167633873.

Each step first adds `N_old*dt/t_rec` to capacity, properly subtracts that amount
from the existing ionized count, then supplies Q*dt through the independent event
solver. It is explicit Euler for this problem.

| dt / t_rec | Relative volume error | Relative radius error |
|---:|---:|---:|
| 0.5 | +8.4235% | +2.7325% |
| 0.125 | +1.9971% | +0.6613% |
| 0.03125 | +0.4917% | +0.1636% |
| 0.015625 | +0.2452% | +0.0817% |

The measured final order is 1.0037; the atom ledger closes to 4.5e-16. Thus
adding R*dt can converge in this restricted sharp-front problem. A large step
does not ensure positivity: with Q=0, N0=1 and dt=2*t_rec the explicit result is
N1=-1. Clipping would conceal a time-integration error and change the ledger.

For constant Q and t_rec, the exact aggregate update is inexpensive:

\[
N_{n+1}=N_n e^{-h/t_{\rm rec}}+Qt_{\rm rec}(1-e^{-h/t_{\rm rec}}).
\]

It agrees to 1.1e-16 in this test. It does not determine which source supplies a
shared sink in overlapping bubbles, nor the surviving directional flux.

## Test 2: a source that turns off must actually recombine

A fixed, homogeneous, pure-H parcel obeys

\[
\dot x=-a x^2,\quad a=\alpha_B n_H,\qquad
x(t)=\frac{x_0}{1+a x_0t}.
\]

With x0=a=1 and t=2, x=1/3. A correctly debited explicit capacity update converges
from x=0.258270 at four steps to x=0.331418 at 128 steps. The exact local update
`x /= 1 + a*x*dt` recovers 1/3 to 3.4e-16 for every tested partition and is
bounded for arbitrarily large positive time steps.

If adding capacity only changes a demand counter while the old ionized field
is retained, no photons are emitted and x incorrectly stays 1. Likewise,
directly substituting the native Gamma=0 equilibrium table as this parcel's
recombination RHS leaves x=1. Both fail by 2/3 regardless of step size. A source
switch-off treatment needs either explicit nonequilibrium chemistry or a stated,
validated alternative; it must not silently reuse the equilibrium sink table.

## Test 3: the remaining opacity error need not shrink with dt

For a homogeneous finite-volume cell, assume monochromatic radiation with
instantaneous transfer, fixed incoming rate F, N hydrogen atoms, and all-neutral
optical depth tau0. A photon-conserving local reference is

\[
 A=F[1-e^{-\tau_0(1-x)}],\quad
 F_{\rm out}=F-A,\quad N\dot x=A-Na x^2.
\]

This reference is a well-mixed-cell closure, not a spatially resolved continuum
ionization front. It exposes the extra approximation made if that same
homogeneous fraction x is treated as completely opaque until its remaining
capacity is exhausted. The sharp-front candidate is allowed its stated
approximation; this test rules out claiming finite-opacity/UVB accuracy merely
from event timing and conservation.

Start with x=1 and set N=F=a=1, t=2. In the naive debt-event scheme, each step
removes dt ions and uses exactly dt photons to replace them. Therefore x stays
1, two photons are absorbed, and none escape for **all** tested step counts
(4, 16, 64, 256). Both its photon and atom ledgers close exactly.

| tau0 | Independent ODE final x | ODE escaped photons / 2 | Debt-event final x | Absolute x error |
|---:|---:|---:|---:|---:|
| 1 | 0.59189327 | 73.0521% | 1 | 0.40810673 |
| 10 | 0.86328043 | 30.0627% | 1 | 0.13671957 |
| 1000 | 0.99532523 | 1.1513% | 1 | 0.00467477 |

These differences are **opacity-closure errors, not time-convergence failures**.
They decrease in this optically thick limit; they do not establish the expected
size of an error in a cosmological Qin20 run. Even an apparently small residual
neutral-fraction error can strongly affect the transmitted background when the
incoming flux is near that required to balance recombinations.

The two-source probe uses beams with rates 1 and 2. They both cross a preionized
screen, then illuminate different targets. The ODE reference gives
x=[0.77650057, 0.36669451, 0.64348893]. At 256 steps, the debt-event method gives
x=[1, 0.31285960, 0.59035468]. Its absorption-map L1 difference is
**25.7868% of emitted photons**, unchanged between 8 and 256 steps. The ledgers
still close to 3e-15. This is a concrete shared-sink failure for the finite-depth
reference closure, beyond a global photon count test.

## A low-cost piece that is now verified

For one cell with a prescribed incoming flux, both subflows are analytic:

\[
 x_{\rm rec}(h)=\frac{x}{1+axh},\qquad
 y_{\rm photo}(h)=\frac{1}{\tau_0}
 \ln\!\left[1+(e^{\tau_0 y}-1)e^{-\tau_0 Fh/N}\right],\quad y=1-x.
\]

`local_chemistry_step` combines recombination half-step, photoionization full
step, and recombination half-step. The implementation evaluates the exponentials
in a stable log form. Each substep records its actual absorbed photons or
recombinations; the remainder of F*h is transmitted. It is bounded and maintains
both ledgers, including zero opacity, zero source and very large tau0/time step
tests. The toy neglects helium, collisional ionization, expansion, temperature
evolution and diffuse case-A recombination radiation.

Against independent Radau and DOP853 reference integration (agreement 6.7e-14),
the x errors at 16/32/64/128 steps are 6.4418e-4 / 1.6127e-4 / 4.0330e-5 /
1.0083e-5. The final observed order is **1.99988**. Ledgers close to 4.5e-16.

The vectorized NumPy kernel takes roughly 0.0035 s for 32³ independent cells and
0.039 s for 64³ on this host in the recorded microbenchmark (minimum 0.023 s at
64³; these small timings depend on concurrent host load). This only establishes
that the **local chemistry operation** is inexpensive. It excludes finding and
updating each cell's incident flux, UVB/sink iterations, source feedback and
cosmological integration. It is not an event-front or Qin20 runtime estimate.

For multiple cells, integrating the transmitted *number* from one cell and
passing its average to the next loses its time history in general. Therefore
this verified local kernel is a component for a future coupled radiation solve,
not a validated multi-source replacement by itself. Time-averaged opacity and
chemistry iteration, as explored in C2-Ray, provide a concrete related strategy;
the iteration count and its cost here must still be measured.

## Recombination invalidates the monotone-transparent shortcut

The current event solver is appropriate for monotonically exhausted capacities
within its no-recombination step. A fresh call reconstructs its geometry/cache.
Recombination introduces a new event type: a previously transparent cell becomes
opaque again. A future optimization which retains old ray cursors or empty-space
clearance across that change must invalidate them.

The regression test deliberately makes this mistake: a transparent foreground
screen recombines to x=1/2 while the source is off. On restart, 0.2 photons must
all be absorbed in its 0.5 neutral capacity. A rebuilt solve does that; an old
cursor sends all 0.2 photons to the downstream cell. Photon conservation alone
does not distinguish them. This is a negative control for future stateful reuse,
**not a bug in the current fresh-call C++ API**.

Furthermore, `capacity==0` is not a certificate of negligible absorption after
introducing residual H I or subgrid recombination sinks. A transparent-block jump
must either integrate its attenuation/sink or certify a bound on the photons
neglected there. That certificate must be invalidated when sources, Gamma,
temperature, density or the ionization state change sufficiently.

## Remaining work expressed as acceptance tests

The numerical tolerances below are proposed engineering gates for developing the
coupling, not measured cosmological accuracies or universal physical thresholds.

| Work | Required evidence before a production claim | Status |
|---|---|---|
| Local nonequilibrium H chemistry | Source-off analytic law; positive fractions; photon and ion ledgers; second-order convergence against two independent ODE integrators | Passed for the prescribed-flux pure-H toy |
| Sharp-front recombination time integration | Strömgren radius/volume converge; no negative state; report chosen dt relative to local recombination and source-variation times | First-order naive update measured; production coupling absent |
| Interior UVB and partial transmission | Shared-screen/overlap ODE fixtures: x error below 1e-3 and integrated absorption/escape error below 0.1% of emitted photons after refinement; Γ/flux must converge independently of x | Naive debt-events fail; coupled solver not implemented |
| Reionization after source shutoff | Off/on screen; behind-screen leakage bounded by the chosen transport tolerance; results identical with full cache rebuilding and selective invalidation | Independent rebuilt-path control passes; optimized cache policy absent |
| Qin20 subgrid sinks | Preserve or explicitly replace the MHR/self-shielding closure; verify recombination-to-atom normalization; do not double-count diffuse recombinations and PDF sinks | Unresolved |
| Gamma feedback | Iterate/predict-correct sources, intensity and sinks; show convergence in Γ, cumulative recombinations, source emissivity and M_turn, rather than only global xHI | Unresolved |
| Cumulative versus incremental state | Each interval closes emitted=absorbed+escaped+retained; Δionized=absorbed−recombined in consistent units, with X-ray/helium terms explicitly accounted for | Toy ledgers pass; physical/native normalization integration unresolved |
| Controlled approximation | Separate angle, spatial, time, subgrid and UVB iteration refinement; report error-versus-cost; reject a run if unresolved recombination debt is mislabeled absorbed | Unresolved for cosmology |
| Cost | Same Qin20 inputs/threads/output on PC and non-PC; include source extraction, source feedback, sink updates, UVB iterations and transport; report full history walltime and peak memory | Local chemistry microbenchmark only |

The promising next implementation is therefore: retain the event geometry for
sharp, optically thick fronts; add an explicitly defined interior attenuation
and sink treatment, a bounded chemistry update, and validated recombination
reopening. A cheap frozen-Gamma predictor with selective correction may be useful,
but cannot be called correct until those tests pass. Retaining the native ES
Gamma as a deliberate approximation is also possible; it would be a hybrid
closure and must be labeled and tested as such.

Primary literature consulted:

- Mellema et al., *C2-Ray: A new method for photon-conserving transport of ionizing
  radiation*, https://arxiv.org/abs/astro-ph/0508416 — photon-conserving transport
  coupled to an analytic relaxation treatment of ionization chemistry.
- Sobacchi & Mesinger, *Inhomogeneous recombinations during cosmic reionization*,
  https://arxiv.org/abs/1402.2298 — subgrid recombination sinks coupled to UVB and
  source feedback. The detailed implementation claims above are based on the
  audited repository source, whose hashes are in the JSON, not on assuming the
  current code is identical to a historical paper.
