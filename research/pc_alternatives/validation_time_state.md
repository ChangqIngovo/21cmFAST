# Time, finite range, and persistent radiation state audit

This audit does not change the event solver. Its constant-rate, infinite-c, no-recombination model can close every photon ledger while disagreeing with causal transport. The counterexamples below are expected limitations, not failed numerical assertions.

## What retained currently means

`event_front.cpp::advance` adds `ray.rate * (duration-now)` to a source-indexed ledger when a geometric search hits `maximum_distance`. The Python API returns that count and no ray positions, directions, emission/arrival times or traveled-distance state. Consequently `source_retained` is unresolved flux under a search cutoff, not a resumable finite-c photon population. Within the declared infinite-c model, total counts remain accounted for. Silently discarding this ledger loses photons; calling it absorbed is also incorrect.

## Exact counterexamples

| Probe | Current/proposed restart | Independent answer |
|---|---:|---:|
| Unit source, target 4.5 cells away, c=2, duration=1 | 1 absorbed | 0 absorbed; 1 in flight |
| Same source, duration=3 | 3 absorbed | 0.75 absorbed; 2.25 in flight |
| Set cutoff=c*dt=1, target 0.5 cell away | 1 absorbed | 0.5 absorbed; 0.5 in flight |
| Source emits 1 photon over t=0..1; restart its retained count at origin, cutoff=2 | After three intervals: 0 absorbed; 1 retained | At c=2, the same emission gives 0.75 absorbed by t=3 |
| Re-isotropize 0.5 carried photons known to move +x | 0.25 spuriously goes to +y | 0 to +y |
| Source emits at t=0..1; target delay=2.25; interval t=2..3 | Zero-source restart adds 0 | In-flight radiation adds 0.75 |
| Move a packet but reset its total range budget each step (Rmax=3) | Absorbs at distance 4.5 | Must remain unresolved at distance 3 |
| Feed cumulative source budgets 3 then 5 as fresh counts | Total absorption 8 | Correct incremental total 5 |

The source-restart examples test hypothetical adapters; the existing prototype does not implement such an adapter. Reinjected carry must not be counted as new physical emission. Ledger closure by itself cannot detect the direction error or early-arrival error. A separate one-dimensional delta-flash state oracle confirms exact split-step continuation: the entire flash reaches distance 4.5 at t=2.25 for c=2. This flash is not represented as an event_front constant-rate input.

A valid control splits the same infinite-c fixed-rate step in half, retaining gas capacity and unchanged directions: allocation L1 difference = 0. This control does not justify finite-c restarts or reuse of cumulative source counts.

For a uniform static medium with no recombinations, the independent relativistic sharp-front solution obeys `Qdot*(t-R/c) = 4*pi*n_H*R^3/3`. It follows by integrating the relativistic jump condition in [Iliev et al. 2006, §2.10, eq.3](https://arxiv.org/abs/astro-ph/0603199), following [Shapiro et al. 2006](https://arxiv.org/abs/astro-ph/0507677). For dimensionless Qdot=c=1 and n_H=3/(4*pi), t=1 gives R=0.68233, whereas infinite-c gives R=1: a 46.6% radius excess. This deliberately relativistic case is a falsification test, not an estimate of Qin20 error.

## Qin20 clock and scale audit

Read the actual saved native input `/workspace/scratch/40b418003e4a/qin20_validation/native_paired_z8/es_inputs.toml` (88 nodes, saved 8³ grid, box 200 cMpc). It uses h=0.6766, Ωm=0.30964144154550644, and ZPRIME_STEP_FACTOR=1.02. Main elapsed times/horizons below integrate native `hubble(z)`, which includes radiation and sets native SFR timescales. ΩΛ uses the actual Python property 1−Ωm; constants match native Constants.c/h. The default 64³ grid changes the cell scale only. These are calculated scales, not a new simulation.

The exact comoving distance light travels between nodes is `c/H0 * integral(dz/E(z))`; proper light crossing estimates use `L_com/(1+z_late)/c`. A fixed search radius, R_BUBBLE_MAX, and the time-dependent causal horizon have different meanings. In particular R_BUBBLE_MAX is an ES cutoff parameter, not proof that photons propagate that full distance.

| Late z | Node dt (Myr) | Light horizon (cMpc) | 64³ cell crossing (Myr) | 50 cMpc crossing (Myr) | 50 cMpc / horizon | Default event diagonal / horizon |
|---:|---:|---:|---:|---:|---:|---:|
| 19.909 | 5.284 | 34.211 | 0.487 | 7.800 | 1.462 | 10.126 |
| 14.846 | 8.014 | 39.319 | 0.643 | 10.292 | 1.272 | 8.810 |
| 10.095 | 13.680 | 46.997 | 0.919 | 14.699 | 1.064 | 7.371 |
| 7.923 | 18.959 | 52.382 | 1.142 | 18.276 | 0.955 | 6.613 |
| 6.036 | 27.043 | 58.914 | 1.449 | 23.179 | 0.849 | 5.880 |

The 32³ frozen cost fixture instead uses nodes 35→12→8→7.5. Its last interval spans 57.243 Myr with the native hubble clock, with light horizon 153.462 cMpc. Native `dtdz` omits radiation: integrating that separate expression gives 57.312 Myr. Native recombinations actually use the endpoint approximation `abs(dtdz(z_late))*dz`, giving 61.538 Myr. These are distinct definitions, not quadrature error or different Astropy parameters. The candidate integration must explicitly choose and consistently apply a clock; this audit changes none of them.

The frozen source grid is cumulative and is reinterpreted as rates for duration=1 in the microbenchmark; it has no conversion to a physical duration. Its speed measurements therefore do not validate temporal evolution.

The default 50 cMpc crossing time is comparable to a native evolution interval; the event default diagonal, sqrt(3)*200 cMpc, is several intervals. This prevents claiming a causal approximation is automatically negligible. It also does not prove large physical errors: relevant absorption distances, source variability, and front speeds can be much smaller. Quasistatic validity requires (path/c)/t_source ≪1, (path/c)/t_opacity ≪1, and front speeds ≪c on the paths that carry substantial flux. Those histories and distances have not yet been measured for the candidate. One frozen field cannot supply them.

## Minimum correct state and acceptance gates

A finite-c continuous-beam extension can retain piecewise-constant rate segments instead of individual photons. It needs their original emission intervals, current positions and directions, current time or delayed arrival intervals, spectral group, integrated remaining counts, and cumulative traveled distance (including periodic wraps) if an absolute range limit is used. Source attribution is optional for physics but required for a per-source ledger. Gas ionized stock and recombination/thermal history must persist, with fresh source increments distinct from stored photons. Arrival, source-on/source-off, and cell-exhaustion events all change incident rates; only an exhaustion queue is insufficient.

An unresolved numerical cutoff is neither physical absorption nor boundary escape. Store the stopped trajectory and remaining time profile, or reject a run when its unresolved fraction exceeds a declared error tolerance. A periodic ray is not escaped merely because it wraps. Empty-space certificates need invalidation when recombinations or changing density restore opacity.

Required acceptance tests (not yet passed by an integrated candidate):

1. Exact transparent-path step/flash arrival and source-off continuation, including split intervals and checkpoint/resume; total emitted = absorbed + escaped + in-flight + explicitly unresolved, without recounting carry.
2. Preserve direction/subcell position, periodic path length, and absolute range across resume; the +x/+y and range-reset counterexamples must vanish.
3. Recover the relativistic uniform-front radius and the nonrelativistic limit; check no absorption outside the causal light cone, and converge interval size and angular/grid resolution.
4. Derive nonnegative native physical source increments with a validated spatial closure; prevent repeated spending of cumulative emission, and include previous ionization plus recombination sinks exactly once.
5. Evolve matched Qin20 histories with finite c and infinite c, then compare xHI, source/atom ledger, ionization times and P21 at matched z and matched xHI. Report actual source/opacity variability and distance distributions. A chosen science tolerance (for example 1% mean xHI and 5% P21 over stated resolved k bands) is a project acceptance criterion, not a theorem or literature guarantee.

[FlexRT, §3.1](https://arxiv.org/html/2409.04521v2) gives primary precedent for preserving packet positions/directions between light-speed steps; its §5.4 demonstrates why finite-c versus infinite-c can change early front positions. Its method is not identical to this candidate.

Generated from event_front.py SHA256 `1fea8965df8f480d06857a01b15b08e8ad03741420552daa7eda5dac6e374688` and event_front.cpp SHA256 `d97be4a8bd20132d8ed6da7314e1bef1d148c0346435d91d49fb2a36aa628fd8`. JSON records the numerical counterexamples, input hashes, and scale calculations. No production code was modified.
