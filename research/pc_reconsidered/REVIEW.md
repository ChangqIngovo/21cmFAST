# Independent review of source-adaptive angular events

Scope: `hybrid.py` and `hybrid.cpp`, compared with the pre-existing
`pc_alternatives/event_front.cpp`. This is a mathematical/code review of the
static-density, constant-emissivity, infinitely fast, opaque-voxel model. It does
not validate chemistry, native Qin20 feedback, or a full reionization history.

## What the event algorithm actually guarantees

For source s with angular count K_s, every launched direction has constant rate
q_s/K_s. Until its origin cell clears, all source emission is locally absorbed.
Creating rays only when that cell clears is an exact storage optimization within
this model. All sources and rays share the same exhaustion clock; they are not
applied sequentially as complete photon packets.

For a cell with remaining atom capacity C and incident total rate F, the next
exhaustion is scheduled at t+C/F. A newly arriving ray first consumes the old
rate over the elapsed interval, then changes F and updates that one event.
When a cell clears, all incoming beams advance along their original directions.
They are never pooled and relaunched isotropically from the receiving cell.

Consequently, apart from floating-point error, each source obeys

    emitted = absorbed + escaped + finite-distance remainder,

and cell absorptions cannot exceed initial neutral atom capacities. Assigning
different K_s to different sources does not change this accounting argument.
The finite-distance remainder is an accounting location at the original source,
not an evolved radiation phase-space state. Adding it to the source next step
would reset its path/range and would require a separately declared model.

These guarantees are stronger than a global ES amplitude fit, but are not a
proof of continuum radiative-transfer accuracy. Every nonempty voxel is opaque
until completely exhausted; partial transmission, subcell ionization structure,
finite light speed, spectrum and recombinations are absent.

## A severe angular predictor defect found and repaired

The first predictor version enlarged K only for sources whose own cell was
already exactly transparent. A source with arbitrarily small positive neutral
capacity inside a large transparent bubble therefore received the isolated-source
K, even though it cleared almost immediately and illuminated the entire wall.

An independent test used a 24^3 unit-density grid, an initially ionized radius-7
sphere, and a central source emitting 0.01 photons in normalized units. The
comparison used K=12288 with the same initial state, not a continuum truth map.

| Remaining capacity in central source cell | Original adaptive K | Sum abs(map-reference) / emitted |
|---:|---:|---:|
| 0 | 12288 | 0 |
| 1e-8 | 24 | 1.90445773 |
| 1e-6 | 24 | 1.90426919 |

The maximum possible L1/emitted discrepancy between two fully absorbing maps is
2. The failure is therefore almost a complete redistribution of the increment,
despite an accurate photon ledger.

The implementation author repaired this by including initial transparent
components that share a face with a source cell, as well as the component
containing that source. No arbitrary near-transparency threshold is introduced.
Independent reruns after that change gave K=12288 and zero difference from the
same fixed-K map for source-cell remaining capacities 0, 1e-8 and 0.005. Maximum
relative source-ledger errors in these runs were below 4.6e-13. The new unit
regression also covers a partially neutral source embedded in a cavity.

## Remaining accuracy and cost limits

1. **Newly formed collective bubbles.** The component predictor uses the initial
   transparent mask. A cluster can collectively clear a much larger region
   during the interval than any isolated source budget predicts. Cached ES
   influence radii can improve this estimate, but do not certify angular error.
   Uniform high-K comparisons and a nonzero minimum K remain necessary.

2. **Time subdivision changes the approximation.** With fixed directions and
   constant rates, restarting the infinite-c model from its ionized state should
   reproduce a subdivided evolution, provided prior remainders are not injected
   again. In the adaptive variant, the source increment shrinks and the initial
   cavity grows when subdividing. This changes K_s, and Fibonacci quadratures
   at different K are not nested. A timestep test therefore probes both temporal
   and angular changes. Hold K fixed to isolate time correctness, then refine
   both independently for the adaptive model.

3. **Periodic bounds were repaired and independently checked.** The first
   predictor treated any component touching a box face as having the whole
   horizon. The current implementation joins face-connected labels across
   opposite periodic faces and uses the complement of the largest empty
   coordinate gap as each component's circular bounding interval. A source is
   aligned with the nearest periodic image of that interval's centre. For a
   face-connected component with a non-full coordinate projection this gives
   the appropriate continuous interval, including a source in an adjacent
   almost-cleared cell. A genuinely full coordinate projection still falls back
   to the horizon. Independent checks found exact predictor translation
   covariance on 16 random 12^3 masks at four transparent fractions. A radius-3.5
   cavity in a 24^3 box, with a partly neutral central source, retained K=3072
   and bitwise-identical translated absorption maps when moved across one or
   three box faces. A winding tube retained the horizon bound; two separate
   components on opposite sides of a source had the same bound under translation.
   Thus artificial boundary-crossing cost is repaired, while genuinely large
   connected regions can still require expensive angular sampling.

4. **No absorber endpoint.** In a fully transparent periodic box, every new
   photon is necessarily a finite-distance remainder. This can be returned
   exactly without constructing any rays. The straightforward implementation
   otherwise selects the maximum K for every source and traces it to the
   horizon, an avoidable late-time cost. The author implemented this fast path
   with a dedicated regression. Open boundaries keep explicit tracing because
   the finite-distance horizon distinguishes escape from retained photons.

5. **Thin directions and opacity changes.** The 10th-percentile density and
   bounding-box radius are heuristic predictors, not bounds on underdense
   tunnels, opening angles or shadow boundaries. A few sparse rays can miss a
   low-solid-angle absorber or leak between resolved directions while conserving
   counts exactly. An angular convergence report must accompany reported map or
   power-spectrum accuracy.

6. **Source grouping is a separate approximation.** Grouping sources deep in a
   transparent region can reduce late-time work. A small group-size/distance-to-
   absorber ratio bounds its apparent positional angle but does not bound map
   L1 error across sharp shadow boundaries. Any grouping must preserve original
   source totals, declare the changed directional geometry, and be validated
   separately. It is not an exact continuation of the current source-anchored
   transport.

## Review disposition

The simultaneous, source-anchored event construction has a coherent conservative
interpretation in its declared sharp-front model. The repaired cavity defect is
an example of why exact accounting alone is insufficient. The adaptive solver
is a credible candidate for native-source experiments; it is not yet justified
as an unrestricted full-Qin20 replacement. A native integration must recompute
the morphology-dependent photoionization, reionization-time, recombination and
thermal fields before the next evolving node. Replacing only xHI while retaining
those native ES fields would mix incompatible histories.
