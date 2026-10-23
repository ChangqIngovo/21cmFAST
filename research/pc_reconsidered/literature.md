# Reconsidering photon conservation after the source-budget audit

This is an update to [the existing survey](../pc_alternatives/literature.md),
not a replacement. It separates source normalization, photon accounting, spatial
allocation, and angular transport. The last two require different accuracy claims.
Primary sources were checked during this reassessment. Local code observations refer to base
commit `5243bd107b8dbd3466d2dc49eb3fc62301a338a0`.

## What the new result establishes

The supplied Qin20 audit finds a sign reversal caused by the counting measure:

| Native ES state | Analytic source Q | Volume UV stock + recombinations, divided by Q | Mass UV stock + recombinations, divided by Q |
|---|---:|---:|---:|
| z approximately 14.846 | 9882.333 | 0.9999307 | 1.0884078 |
| z approximately 10.094 | 63721.998 | 0.9731851 | 1.0919392 |

These are supplied audit measurements, not a new run performed for this report.
At the latter state, mass UV stock minus volume UV stock is 7567.248 in the
same normalized units. A scalar closure against the second ledger therefore
reduces the amplitude even though the first ledger has a deficit. This does not
measure one unambiguous overlap loss. It exposes a mismatch that must be resolved
before choosing a transport correction.

Removing the density factor would close an effective volume-based calibration;
it would not establish conservation of physical hydrogen atoms. Conversely,
interpreting a volume-normalized conditional ratio as a photon count while
charging density-weighted absorptions is not a justified source model.

There is also no general theorem that native ES only loses photons. Choudhury &
Paranjape (2018), Appendix C, test native 21cmFAST **with recombinations disabled**:
their source-to-ionized-mass ratio is below one at mass ionization fraction 0.5
and above one at 0.8. The numerical artifact can have either sign [1].

For fixed sources, gas, and a UV-only static problem, define

\[
L_{\rm ES}=Q_{\rm source}-\sum_i N_{{\rm H},i}x_{{\rm HII},i}^{\rm ES}.
\]

If all available photons are absorbed, a conserving correction increases the
ionized atom count exactly when this independently measured deficit is positive.
In an evolving problem the corresponding difference also includes changed
recombinations, free/escaped photons, and, if enabled, source feedback. Neither
the sign of a volume-mean ionization change nor a power-spectrum suppression is
by itself a validation of conservation.

## The overlooked native option: an additive Lagrangian source field

The repository already exposes `SOURCE_MODEL="L-INTEGRAL"`, documented as the
ESF-L construction: integrate source properties on the initial Lagrangian grid,
then map them to the Eulerian grid with 2LPT. The relevant code is
`src/py21cmfast/wrapper/inputs.py` and `src/py21cmfast/src/EmissivityFields.c`.
The latter supplies additive emissivity fields and a `fesc_weighted_sfrd` path
for Lagrangian source models with recombinations. Array units, normalization,
remapping conservation, and temporal meaning still need explicit adapter tests.
In particular, an array already expressed per mean cell must **not** acquire an
extra density factor just because absorption counts do.

Trac et al. (2022), section III.3, construct collapsed mass on equal-mass
Lagrangian elements and deposit it after displacement. Their tests favor ESF-L
over Eulerian ESF source fields in mean collapsed fraction, bias, and correlation
with N-body halos. This is a source-model result; AMBER's separate abundance
matching of ionization times does not conserve local source-to-sink photons [2].
Davies & Furlanetto (2022), Appendix A, explicitly identify ESF-L as a more
accurate alternative to their FFRT-P source model without added computational
expense in that comparison [3]. This is not yet a runtime measurement for our
Qin20/MCG build.

**Recommendation:** prioritize this existing source path before building a new
source generator. Preserve exact Qin20/E-INTEGRAL as the original baseline. An
L-INTEGRAL experiment must be named “Qin20 astrophysical parameters with a changed
source model,” and must have its own matched PC/no-PC control. Changing sources
and transport in only one member of a pair cannot isolate photon conservation.

For an E-INTEGRAL compatibility path, explicitly choose between preserving the
native local source realization and mass-normalizing it to an analytic total.
These are different model contracts. If a local quantity is defined as photons
per local hydrogen atom, its additive count is that quantity times the local
hydrogen count. Computing separate conditional source estimates at each filter
scale need not equal filtering a single additive source grid.

## Candidate ranking after accounting is repaired

The table ranks development priorities, not measured performance on identical
inputs. “Spatial conservation” below means that accepted positive transfers
debit identifiable source budgets and credit atom capacities; it does not imply
that those transfers follow correct rays.

| Family | What it can provide | Main limitation and disposition |
|---|---|---|
| Park/global redshift or amplitude calibration | Very cheap correction of a global history using native ES morphology | No local transfer ledger; calibration and temporal feedback can interact. Retain as a control, not the spatial reference [4]. |
| Modern synchronous SCRIPT-style redistribution | Explicit source and atom counts; local distance-ordered allocation; practical CPU cost evidence | Overlap surplus loses some original source direction. Leading cheap spatial candidate if directional errors fit the intended observables [5,6]. |
| Positive FFT shell/cone allocation | Source-count-independent convolutions; exact nonnegative transfer ledger is possible | Scalar shells lose direction and can absorb behind screens. Cones must retain direction between transfers, not reset it. Prototype contender, not established RT. |
| BEORN/GRIZZLY profile overlap repair | Fast profile painting and nearby surplus placement | Profile medium and volume/count conventions must match our density field. Fast pooling can join disconnected regions. Useful competitor, not a drop-in density-weighted proof [7]. |
| Davies mean-free-path filters / FFRT-P | Additive source filtering and smoother attenuation than a hard radius | Attenuating the ES filter does not itself prevent photons funding multiple ionized regions. Use for source/opacity modeling, not as the PC algorithm [3]. |
| AMBER rank/abundance matching | Very cheap prescribed global mass-ionization history with useful morphology | The history is an input, and source budgets do not constrain individual transfers. Surrogate/control only [2]. |
| Stochastic photon-conserving excursion-set partitions | Counting-correct stochastic bubble statistics in their stated construction | A bubble distribution is not a general 3-D multi-source transport map. White-noise results do not automatically extend to arbitrary CDM fields [8]. |
| Optimal transport / obstacle / diffusion closure | Explicit global capacity constraints; scalable elliptic or allocation solvers | Minimum distance or diffusion is not straight-line radiation; shadows and angular quotas can fail. Mathematical allocator/control, not a physical transport proof. |
| Hierarchical source/cell aggregation | Promising acceleration of an otherwise valid allocator | A tree is an acceleration structure, not a closure. Axis, translation, opening-angle and opacity errors require convergence tests; receiver-based ray sums need a separate emitted-photon ledger [9]. |
| ARTIST and direction-preserving characteristics | Stronger angular and causal information while retaining 21cmFAST inputs | Source/angular resolution and time stepping raise costs. Use as a higher-fidelity comparison or localized refinement; published ARTIST comparisons do not establish a small overhead [10]. |
| M1/OTVET/FLD moment methods | Source-count-independent conservative radiation evolution | Beam crossing/shadow closure errors and chemistry/time-step costs remain. Broader solver change than an ES correction [11]. |
| C2Ray/pyC2Ray GPU transport | Independent photon-conserving RT reference with realistic chemistry tests | Different hardware and algorithm. Large speedups over CPU C2Ray are not speedups over 21cmFAST; useful reference, not a CPU-cost claim [12]. |

The existing local experiments already exclude scalar conservation as a
sufficient acceptance test: positive FFT allocation, hierarchy, obstacle flow,
and a SCRIPT-like toy method close their ledgers while failing specific screen,
translation, or two-bubble angular tests. See `counterexamples_results.json`,
`fft_results.json`, `hierarchy_results.json`, and `ACCURACY_REPAIR_REPORT_zh.md`
under `research/pc_alternatives`. These failures apply to those implementations
and contracts, not automatically to every published variant.

Modern SCRIPT deserves renewed attention because the original 2018 cost is
misleading for current comparisons. Choudhury, Paranjape & Bosman (2021), section
2.1, report 2–4 seconds on one processor for a 128-cubed map [5]. Maity & Choudhury
(2022), footnote 2, describe synchronous independent redistribution rounds that
remove sequential processing dependence and simplify parallelization [6]. Their
6–7-second full-history figure uses a 256 Mpc/h box with **16 Mpc/h cells**, hence
16-cubed ionization cells, not a 256-cubed ionization grid. None of these figures
determines our own accuracy/cost frontier.

## Minimal approximately physical model to develop

1. Produce one nonnegative additive source-count grid and one atom-count grid.
   Include ACG and MCG contributions with declared units. Keep the original
   analytic mean and realized box total as distinct diagnostics.
2. Use native ES scores/bubbles only as a proposal or morphology prior. Every
   accepted ionization must debit the same source grid; overlapping proposals
   cannot each spend a shared photon. A genuinely conservative allocator can
   alter the native ES map and is therefore not just a new scalar ES criterion.
3. Compare synchronous local shell redistribution against a positive FFT
   implementation on identical inputs. Use explicit remainders at finite range;
   reaching the algorithmic iteration/radius limit must not discard photons.
4. Add a directional budget only where benchmarks show it is necessary. Reusing
   the existing high-angular-resolution front solver as a reference is sensible;
   using a low-angular-resolution run of the same closure as “truth” is not.
5. For evolution, integrate instantaneous escaped emissivity over each interval.
   Do not interpret a negative Eulerian cumulative-map difference as zero newly
   emitted photons: advection, source redistribution, and changing normalization
   can all produce negative differences without negative physical emission.
   Carry photons/sinks/history explicitly, and establish the stationary-gas
   contract before adding conservative gas/source remapping.

This is a staged approximation, not a proof that a cheap global correction has
recovered full radiative transfer. Start with static, no-recombination snapshots;
only then add fixed sinks and coupled source feedback. A frozen-source ablation
separates transport errors from changes in source physics.

## Benchmarks and failure criteria

- **Budget gate:** all-partial cells, a saturated box, finite-range leftovers,
  separate ACG/MCG sources, and a uniform density rescaling. An all-partial,
  no-sink state should close the chosen source/atom ledger without a fitted
  amplitude. Source weights and atom capacities must never become negative.
- **Spatial gate:** one source, unequal sources, a dense sector, an opaque plane,
  disconnected cavities, translations and rotations, and the overlapping-bubble
  12.5% directional increment. Preserve the same density/source realization when
  regridding. Passing a global count while putting photons behind an opaque
  foreground is a failed transport test.
- **Approximation gate:** do not redefine failed angular tests as passed because
  the intended statistic is a power spectrum. Report them as limitations, then
  establish whether their effect is tolerable in representative cosmological
  fields. No universal acceptable threshold follows from the literature.
- **Time gate:** constant emissivity, a source switched off, a source moved across
  cells, unequal time steps, and a fixed source history integrated with successively
  smaller steps. Without specified evolution/remapping, a sequence of individually
  balanced snapshots is not demonstrated temporal conservation.
- **Independent reference:** analytical single-source/opacity limits, then an
  independently implemented converged RT calculation on a tractable box. C2Ray or
  pyC2Ray is a useful option if installation and hardware permit. Use the existing
  directional prototype as an additional comparison, not the only oracle.
- **Observable gate:** compare mass and volume ionization histories, morphology,
  map cross-correlation, ionization-density cross-power, and 21-cm power at both
  fixed redshift and fixed ionized mass fraction. A near-unity power ratio can
  conceal a wrong map; a suppression alone is not evidence of greater accuracy.
- **Cost gate:** same source grid, resolution, seed, source history, thread count,
  accuracy settings and hardware. Publish source construction, transport, chemistry,
  complete-run wall time, peak memory, and any precomputation. Measure early,
  overlapping, and near-complete stages; redistribution can become expensive near
  percolation. Choose the winner only after these matched measurements.

## Primary references

1. Choudhury & Paranjape 2018, *Photon number conservation and the large-scale
   21 cm power spectrum in semi-numerical models of reionization*, especially
   Appendix C: <https://arxiv.org/html/1807.00836v2>.
2. Trac et al. 2022, *AMBER*, section III.3 on ESF-L, section III.4 on abundance
   matching: <https://arxiv.org/html/2109.10375v2>.
3. Davies & Furlanetto 2022, *Improved treatments of the ionizing photon mean free
   path in semi-numerical simulations of reionization*, especially Appendix A:
   <https://arxiv.org/html/2103.09821v2>.
4. Park, Greig & Mesinger 2022, *Calibrating excursion set reionization models
   to approximately conserve ionizing photons*: <https://arxiv.org/abs/2112.05184>.
5. Choudhury, Paranjape & Bosman 2021, *Studying the Lyman-alpha optical depth
   fluctuations at z approximately 5.5 using fast semi-numerical methods*, section
   2.1: <https://arxiv.org/html/2003.08958v3>.
6. Maity & Choudhury 2022, *Probing the thermal history during reionization using
   a semi-numerical photon-conserving code SCRIPT*, section 2.1 and footnote 2:
   <https://arxiv.org/html/2110.14231v2>.
7. Schaeffer et al. 2023, *BEORN: a fast and flexible framework to simulate the
   epoch of reionisation and cosmic dawn*:
   <https://academic.oup.com/mnras/article/526/2/2942/7287610>.
8. Paranjape, Choudhury & Padmanabhan 2016, *Photon number conserving models of
   HII bubbles during reionization*: <https://arxiv.org/html/1512.01345v2>.
9. Wadsley, Baumschlager & Shen 2024, *TREVR2: Illuminating fast N log2 N
   radiative transfer*: <https://arxiv.org/html/2310.15235v2>.
10. Molaro et al. 2019, *ARTIST: fast radiative transfer for large-scale
    simulations of the epoch of reionisation*:
    <https://arxiv.org/abs/1901.03340>.
11. Petkova & Springel 2009, *An implementation of radiative transfer in the
    cosmological simulation code GADGET*: <https://arxiv.org/abs/0812.1801>.
12. Hirling et al. 2024, *pyC2Ray: A flexible and GPU-accelerated Radiative
    Transfer Framework for Simulating the Cosmic Epoch of Reionization*:
    <https://arxiv.org/abs/2311.01492>; official documentation:
    <https://pyc2ray.readthedocs.io/en/latest/>. The 2026 ASORA optimization report
    quotes 800-fold acceleration against multicore CPU C2Ray, with a different
    denominator from ES: <https://doi.org/10.1145/3815572.3815737>.
