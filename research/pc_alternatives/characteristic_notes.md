# Focused follow-up: cheap directional transport with certified shortcuts

2026-10-23. Research design and isolated geometry tests, not a production algorithm or a full radiative-transfer validation. The current production mode remains unchanged.

## Recommendation

The strongest next candidate is **exact Cartesian characteristic traversal for a discrete angular quadrature**, retaining the native additive 21cmFAST source field. In the infinite-speed sharp-front approximation, continuous beam rates can be advanced by cell-exhaustion events rather than repeated packet batches; see the event-model section below. Begin without angular or spatial source merging. Use sparse active beams/packets, common-direction path caches, and certified transparent-region jumps. These optimize a defined transport discretization rather than changing its trajectories. Introduce approximate merging only after the unmerged reference passes the screen/rotation tests and its actual cost is measured.

This is a recommendation for the next paired benchmark, not a claim that it is already fast enough or physically complete. Exact ray geometry fixes donor-cell transverse diffusion; it does not fix finite angular sampling, source placement, coarse-cell chemistry, time integration, or unresolved absorption.

## What existing merging methods establish

- **FlexRT §3.3:** exempts the globally brightest `N_ex*N_cells` packets, groups the rest by cell and a randomly oriented HEALPix bin, sums photon counts, and averages positions; directions are assigned to the bin. This is number conserving but changes trajectories. The described global exemption implies a bound on total/mean packet count, not a strict per-cell maximum. The paper reserves optimized performance tests for Paper III. No matched 21cmFAST cost was found in the current search. [F24]
- **TRAPHIC §4.2.3:** merges received packets within reception cones with photon-weighted direction averaging. Its source-count independence comes from finite angular resolution; it does not make arbitrary original rays identical. [PS08]
- **ARTIST §5:** the directly stated comparison is 250 Mpc, 128³: 12,000 single-core minutes over 160 redshifts versus 20 minutes/80 redshifts for 21CMMC, hence 600 times total or 300 times per sampled redshift. This is not a cost estimate for our proposed characteristic kernel. [M19]
- **TREVR2:** combines distant sources in reverse tree walks, using transmission averages and a complex-source model to reduce errors from nonuniform absorbing material. It explicitly targets approximate radiation fields, primarily nonperiodic zoom simulations; geometric opening angle alone does not certify attenuation or a source-resolved absorption budget. [W24]
- **Osborne 2025:** hierarchical DDA and sparse/multiresolution opacity representations accelerate formal ray integration. It explicitly warns that transparent regions can contain arbitrarily intricate radiation fields and that ray averaging there can destroy information needed at later absorbers. Its solar non-LTE timings are not transferable to Qin20. [O25]

Sources: F24 <https://arxiv.org/html/2409.04521v2>; PS08 <https://arxiv.org/html/0802.1715v2>; M19 <https://arxiv.org/html/1901.03340v2>; W24 <https://arxiv.org/html/2310.15235v2>; O25 <https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf055/8322417>.

## Exact shortcuts: assumptions and proofs

The following deductions are ours. They concern a specified packet discretization, not the continuous angular solution. A packet state must retain position `x`, direction `n`, spectral group, clock/flight budget, and any source-dependent range constraint. Photon number is its weight `w`.

### 1. A transparent block can be crossed without changing the ray

For pure absorption along a characteristic,

\[
\frac{dw}{ds}=-\kappa(\mathbf x+s\mathbf n,t+s/c,\nu)w.
\]

If an enclosing block certifies `kappa=0` throughout the relevant interval, then exactly `w'=w`, `n'=n`, `x'=x+L*n`, `t'=t+L/c`. Choose `L` no greater than the ray/block exit, causal/time horizon, remaining source range, or domain/periodic boundary. This is the same transport operator as visiting every zero-opacity leaf.

No intensity averaging is performed. In particular, two rays retain distinct impact positions and clocks. The absence of interaction allows geometry to be skipped, not the radiation field to be coarsened. An ionized-cell flag alone is not a certificate if residual HI, recombinations, dust, or subgrid sinks still cause absorption.

For a first-hit capacity model, replace opacity by the set of cells with positive remaining capacity. Then the proof applies to that model's empty cells, without proving that the capacity model is equivalent to physical opacity/chemistry.

### 2. Safe Chebyshev-distance jumps

Let `D_i=min_{j opaque} ||i-j||_infinity` in cell-index units. All cells satisfying `||j-i||_infinity<D_i` are empty. Their union contains the box

\[
[(\mathbf i-D_i+1)\Delta x,(\mathbf i+D_i)\Delta x].
\]

Compute the ray exit from this box using the packet's **actual subcell position**. A simpler conservative physical jump is `max(D_i-1,0)*dx`; jumping `D_i*dx` from an arbitrary subcell position is not certified. Stop at domain boundaries unless the distance construction and traversal correctly handle periodic images.

If capacity only decreases, the opaque set shrinks. A cached old distance is therefore a lower bound on the new distance and remains safe, although slower. Recombination or new sinks can increase the opaque set, invalidating that argument. Track a medium epoch and rebuild or conservatively update the certificate.

### 3. Geometry can be shared without merging packets

On a uniform Cartesian mesh, all rays emitted from cell centres with the same direction have DDA cell-offset sequences and segment lengths related by an integer translation. Precompute the offsets/lengths once per angular node to the required range; translate by each source cell. Preserve the path cursor and within-segment offset for later steps.

This is exact for the chosen cell-centred/angular discretization. Periodic wrapping changes the cell lookup, not the geometrical path length. Source-specific rotations, spatial relocation during merging, and adaptive splitting require another cache key or a fallback DDA. Caching saves geometry work, not the unavoidable interactions with different opacity fields. Opacity/path-sum caches require medium-version checks; geometric caches do not.

### 4. Exact merging has a stricter condition than cell and direction

The sufficient state equivalence is the same position/characteristic, direction, spectral response, time bin, and remaining causal/range rules. Future evolution then depends on the sum of weights because pure transport is linear in weight. Independent source provenance can be retained as a diagnostic or dropped only when no source-specific later rule requires it.

**Same voxel plus same direction is insufficient.** Parallel rays at different transverse positions can hit different downstream cells. Our executable counterexample places two equal-weight packets in one voxel, with identical direction; separate rays absorb 0.5 in a small opaque obstacle, while the packet at their averaged position absorbs 1.0.

A more general exact merge is possible when the entire downstream transfer operator is proved identical over a certified horizon. Usually that is not known before traversal. A lazy bundle that retains all constituent trajectories may share tree traversal through empty space and split before an interaction; this saves work without claiming that its members became one ray.

## Continuous-rate events: a sharper candidate than packet batching

Root proposed this model during the focused research. Each static source has a constant luminosity over a selected interval, divided among fixed rays. Each ray deposits its entire rate into its first positive-capacity cell. Cells are opaque until their shared capacity is exhausted, then transparent. There is no recombination, scattering, finite-speed delay, or interior residual opacity in this version. These are explicit physical/discretization approximations.

Let `A_i` be the sum of **all** incident beam rates at opaque cell i. Between changes of ray destinations, `dC_i/dt=-A_i`. Thus its next exhaustion time is exactly `t_now+C_i(t_now)/A_i`. A heap finds the earliest event. Settle that cell's capacity, reroute its incident rays to their next positive-capacity cells using exact characteristic traversal, and update only those destinations' rates and event times. Settle each destination under its old rate before changing the rate. Process simultaneous events consistently; final-time partial cells require lazy settlement too.

Induction over events proves this is an exact event integrator for the declared finite-ray model: before the minimum exhaustion time, no ray destination changes and the capacity ODEs have constant slopes. At that time the set of zero-capacity cells is determined, so rays can be advanced and the same argument repeated. This removes the artificial dependence on how a constant source interval is divided into batches. It does not prove equivalence to continuous-angle RT.

Under monotone capacities, there are at most N_cells successful exhaustion events. Beam arrivals, heap updates, and ray-cell visits remain additional work. A conservative complexity statement is `O((N_exhaust+N_arrivals)*log(N_cells)+N_ray_segment_visits)`, plus setup; do not advertise simply N_cells work. Incident rates at a still-opaque cell can only increase under constant source rates, so an indexed decrease-key heap avoids accumulated stale records. Source switch-off, recombinations, changing density and sink opacity require additional events or a more general solver.

Beam objects carry rates, not a stored photon count. At final T, emission is the time integral of luminosity, absorption is the capacity decrease, and escaped/unreachable contributions are time integrals of their rates. Counting beam rates again as a photon reservoir would double count. Periodic no-cutoff rays also require explicit closed-orbit/unreachable handling; a rational direction can circulate through transparent cells forever while neutral cells remain elsewhere.

**Closest verified prior art:** Abel & Wandelt 2002 §3 integrates constant-source front jump conditions to obtain passing times along rays, recovering a single-source front history with one radial integration. Its per-cell passing time is the maximum across rays, rather than the proposed shared-capacity sum-rate rule. Their §4 also discusses skipping unchanged low-opacity ray segments and assigning ray-specific opacity time steps. IFT, described by its authors in Iliev et al. 2006 §2.10, assumes sharp fronts/equilibrium interiors and integrates directional front positions; it includes relativistic front speeds but at that time supports one source. [AW02, I06]

Lee et al. 2016 uses asynchronous equilibrium/non-equilibrium cell evolution and fast identification of nearest front cells; it is not the exact exhaustion-heap formulation. Subsweep 2025 §2.10 directly discusses the source/direction competition problem: separate directional chemistry creates ordering asymmetry, while independent per-direction atom caps do not ensure shared capacity conservation. Its chosen cap is N_atoms per direction and its remaining nonconservation is acknowledged. Steele & Wu 2022 uses explicit time-stepped polar-cell ray attenuation, not a shared-capacity event heap. [L16, P25, SW22]

Targeted searches did not identify the exact proposed heap formulation. This is not evidence of novelty. Describe it as an **event-driven discrete implementation of ionization-front tracking**, with its particular shared-capacity rule and proofs stated explicitly.

## Bounded approximations

For a fixed, non-emitting, non-scattering opacity field, suppose skipped path segments have total optical-depth upper bound `T`. The fraction of input weight whose absorption outcome can change is at most

\[
m\leq w_0(1-e^{-T}).
\]

Couple the exact and skipped solutions until an absorption in an omitted segment. Only that event can alter the eventual outcome. Consequently the L1 error of the vector containing every cell's absorbed count **and** the outgoing/residual count is at most `2*m`. For many packets, sum these absolute bounds. Accounting still closes exactly because skipped photons remain in the transport field.

This is a bound for the frozen linear problem. It does not automatically bound the evolving ionization map: altered absorption changes later opacity and fronts. A global small photon error also need not imply small relative error in a weakly illuminated cell. Carry numerical-error estimates separately from physical recombination/residual ledgers.

For approximate source clustering, source radius `a` at distance `D>a` gives a directional spread no greater than `asin(a/D)` and geometric inverse-square bounds using distances `D-a` and `D+a`. With path optical depths bounded between `tau_min` and `tau_max`, a total luminosity `L` has point-flux bounds

\[
\frac{L e^{-\tau_{\max}}}{4\pi(D+a)^2}
\leq F \leq
\frac{L e^{-\tau_{\min}}}{4\pi(D-a)^2}.
\]

Use these only where the bounds cover every member-to-target path and emission time. Opening angle alone is inadequate: arbitrarily narrow opaque/transparent apertures make an arbitrarily small positional or directional perturbation change transmission by order unity. Reverse receiver-by-receiver flux sums additionally require a source budget mechanism before they constitute a conservative allocation.

## Can the predominantly neutral phase be very cheap?

Potentially, because most photons may interact nearby and active paths terminate quickly. This is an empirical occupancy property, not a worst-case guarantee. Native conditional source fields may be positive in nearly every cell, so allocating all `N_cells*N_angles` packets eagerly is especially wasteful.

One option is lazy creation: analytically process the source cell and instantiate only significant escaped packets. For a centred point source in a frozen homogeneous cube, every escape path is at least `dx/2`, so the total escaping count is bounded by `Q exp(-kappa*dx/2)`. This requires a lower bound on opacity **throughout** the emission/transport interval; ionizing away the cell can invalidate it. Ignored escaped counts must stay in an explicit unresolved reservoir with a tolerance, not be silently converted into local absorption.

The native voxel emissivity does not specify a point-source position. Centre placement is an added spatial approximation. Uniformly distributed emission within the voxel has sources arbitrarily near faces: its optically thick escape scales as a boundary layer, `O(1/(kappa*dx))`, rather than the exponential point-source bound.

For an isolated homogeneous sharp-front model only, a centred source whose spherical front remains inside the voxel's inscribed sphere has no escaped photons: `r_front<dx/2`, or `Q<(pi/6)*n_H*dx^3` without recombinations. Neither `Q<C_cell` alone nor this isolated-source condition establishes the result when neighbouring sources can preionize channels. Source clustering is least useful when those distinct nearby opacity paths must be resolved.

Cumulative snapshot budgets do not remove this geometric information requirement. Reusing a previous transparent field requires consistent incremental emission, ionized-atom, surviving-photon, and recombination ledgers; restarting with cumulative photons on the already ionized field double spends earlier photons. Instantaneously tracing all cumulative photons also assumes a quasi-static/infinite-speed model unless packet clocks enforce causality.

## Reproducible isolated checks and next benchmark

Run `python research/pc_alternatives/characteristic_certificates.py`. It writes `characteristic_certificates_results.json`.

| Probe | Result |
|---|---|
| Three transparent-block ray tests | Fine/accelerated absorption agrees to 1.2e-15 |
| Traversal segment counts | 51→8, 55→10, 31→6 |
| Translated characteristic cache | Segment lengths agree to 2.4e-15 |
| Invalid cell+direction merge | Absorbed fraction 0.5→1.0 |
| Low-opacity skip | Absorption+residual L1 error 5.05e-4 ≤ certified 6.41e-4 |

These are geometry/operator tests, not runtime benchmarks and not nonlinear chemistry tests. The next paired native benchmark should report initial/peak/terminated packet counts, fine interactions versus jumps, angular quadrature, cycles, all reservoirs, memory and walltime. Compare equal source/sink fields and threads, with skip disabled/enabled producing the same allocation within numerical tolerance. Include oblique screens, crossing beams, cell-boundary source translations, the equal-sphere 1/8 cone, and angular/spatial refinement. Only then evaluate approximate merging against the unmerged solver.
