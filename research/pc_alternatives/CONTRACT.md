# Research contract: alternatives to global-budget

The production `global-budget` mode at f35833db is retained unchanged.
This directory contains isolated candidate algorithms and falsifiable probes;
none is a production replacement until its assumptions and tests are reported.

Goal: retain the 21cmFAST source/thermal machinery and multiscale ES organization,
seek explicit photon allocation with materially stronger spatial constraints,
and measure additional cost rather than assuming it is small.

Distinguish three levels: global ledger closure; nonnegative source/cell allocation
with no duplicate spending; physically correct directional propagation and sinks.
Passing one level does not prove the next. A leftover photon reservoir must not
be counted as absorbed. No claim that every possible algorithm is exhausted.

Common numerical inputs for toy tests: finite 3D periodic arrays `source` and
`capacity`, both nonnegative float64 counts per cell. Explicitly state whether
the algorithm uses cumulative sources or time-step photon increments. Real
Qin20 source statistics are not assumed to be additive across ES filter radii.

Required probes where applicable:
- zero source, weak fractional source, saturation, nonuniform capacity;
- isolated source sphere and radial profile;
- two equal sources / overlapping bubbles, and unequal separated sources;
- source permutation, translation, rotation, photon and capacity monotonicity;
- opaque screen / shadow: local capacity constraints alone do not imply transport;
- allocation-radius support; partial absorption and residual/unreachable photons;
- fixed-seed timings vs the SAME grid/thread count, separate microbenchmark from
  full 21cmFAST walltime; report interactions/iterations/FFTs and memory scaling.

For two spheres R separated by 1.5 R, directions crossing the first sphere's
surface inside the second cover (1 - d/(2R))/2 = 1/8 of an isotropic source.
Do not confuse this with the entire apparent solid angle of the second sphere.
Angular reference tests must state which of these they measure.

All scientific claims must cite primary publications or source code. Counterexamples
and unresolved assumptions are results, not reasons to conceal a candidate failure.
