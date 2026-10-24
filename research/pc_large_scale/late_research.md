# Static late-stage transport: findings and next experiment

## What changes in this revision

The opaque eight-subcell transport rule is unchanged. The new optional distance
controller increases the allowed path until successive solutions meet declared
retention, absorbed-field and optional scalar-statistic tolerances. Every solve
starts from the same gas state, with the same emitted photons and angular bank.
No photons are added to match an ionized fraction, and no retained photons are
re-emitted from source centres. A finite ladder can fail, which is reported.

The angular predictor previously used the transport distance as a cap on its
estimated radius. Changing distance could therefore change both geometry and
angular sampling. `angular_reference_distance` now allows these controls to be
varied independently, and the follow-up checks the actual per-source counts.

## Small native fixture: distance sensitivity

`horizon_probe_results.json` uses the supplied native 32 fixture, a fixed
200 cMpc box at z=8. The bias band has 92 full Fourier modes with integer
Fourier radii 1 to 3, rather than the production 500 h^-1 cMpc band's modes.
All entries use the same emitted budget and begin with neutral gas.

| Emitted photons / atoms | Path cutoff / L | Retained / emitted | Actual Q_mass | b_HI^2 |
|---:|---:|---:|---:|---:|
| 0.8 | 1/3 | 0.0141991 | 0.7886407 | 107.9797 |
| 0.8 | 2/3 | 0.00137458 | 0.7989003 | 119.4993 |
| 0.8 | 4/3 | 0.0000195251 | 0.7999844 | 120.7831 |
| 0.8 | 8/3 | 0 | 0.8000000 | 120.7996 |
| 0.9 | 1/3 | 0.0356283 | 0.8679345 | 202.6866 |
| 0.9 | 2/3 | 0.00854521 | 0.8923093 | 267.4850 |
| 0.9 | 4/3 | 0.000768326 | 0.8993085 | 291.9351 |
| 0.9 | 8/3 | 0.0000104710 | 0.8999906 | 294.5107 |

The last doubling changes b^2 by 0.0136% and 0.8822%, respectively. Angular-bank
hashes are unchanged and ledger residuals are <=1.08e-16. Geometry queries from
L/3 to 8 L/3 grow by factors 1.223 and 1.717. These are single-configuration query
costs; the controller pays for successive solves as well. Power changes here
include changes in global ionization. They are not fixed-Q morphology changes
and must not be substituted for the production 500 results.

A periodic path longer than the box can revisit the same periodic medium. This
tests sensitivity to a numerical cutoff, not a physical mean-free-path model
or finite-light-speed propagation.

At late times, normalize retained photons also to the remaining neutral atoms:

    N_ret / N_HI = (N_ret / N_gamma) * (N_gamma / N_H) / (1 - Q_mass).

The production run's roughly 0.46% photon retention at Q~0.9 corresponds to about
4.2% of remaining neutral atoms. A small emitted-photon fraction alone cannot
establish accuracy of normalized HI fluctuations.

## Why final-state re-emission is not a repair

The probe also reproduces a two-source 12^3 test with 384 directions. A direct
long-distance solve absorbs 1384.114922 photons. Re-emitting the retained source
totals from the final state of a short-distance solve absorbs 1399.313282, with
L1/emitted=2.02519% and maximum cell fraction difference 0.502505. Both photon
ledgers close. The source-total reservoir has lost direction and event-time
information. Exact continuation would need the original coupled event history;
independent longer-distance solves avoid that reconstruction.

## Separate output projection from solving a coarser problem

For a fine solution f, its conservative projection p onto coarse cells, and a
direct coarse solution c, the measured bias ratio decomposes exactly as

    b2_f / b2_c = (b2_f / b2_p) * (b2_p / b2_c).

`diagnose_coarsening.py` computes this without new photon transport. The second
factor uses identical cell windows, atom capacities and matter denominator. It
still includes both changes to input source/density fields and to transport.
It is diagnostic, not a correction to the original resolution result; averaging
can hide smaller-scale disagreements, so cross-correlation and field L1 are
reported too.

In the earlier small native 12^3/6^3 smoke, at Q=.2 and seed 8123, the original
fine/coarse ratio 0.67405 decomposes into 0.68412*0.98529. Thus most of that apparent
32.6% difference comes from output projection. At Q=.8, the same-mesh factors
are 0.91028 and 0.88389 for the two seeds: a substantial residual remains.
`coarsening_projection_smoke.json` records all input hashes and comparisons.
The production raw grids are on Gadi; their decomposition has not yet been run.

## Input smoothing factorial on native 32

`coarsening_factorial.py` independently smooths density, sources, or both on the
fine mesh, then measures every ionization map on the same 16^3 mesh. Fixed 384
directions, Q=.8 and horizon 4 L give no retained photons in any arm.

| Fine-mesh input | Common-mesh b_HI^2 | Change from original |
|---|---:|---:|
| Original | 195.0290 | 0 |
| Density smoothed | 192.4370 | -1.329% |
| Source smoothed | 197.9078 | +1.476% |
| Both smoothed | 195.5588 | +0.272% |
| Direct coarse solve | 201.3541 | +3.243% |

The fine/coarse same-mesh ratio is 0.968587 at 384 directions and 0.969251 at 768
directions. The residual therefore survives angle doubling in this fixture.
Input smoothing alone nearly cancels here. The remaining effect combines
relocating sources to coarse centres with coarser front geometry; these are not
separately measured. This is one fixture and 18 modes, not the production ensemble.
`coarsening_factorial_validation.json` provides input/code hashes and audits.

Simply replacing 8 children by 64 is not yet a valid refinement of the lazy rule:
only the central 8 of 64 subcells touch a central source, whereas all current 8
children do. A finer spatial reference must retain source positions and handle
propagation to the other subcells explicitly.

## Production follow-up

The 120-case 32-worker job reuses the completed large run's native fields. It
tests both late doses at path factors 1/2/4/8 and doubles angles at the longest
distance, verifies reproduction of each original same-budget map, and calculates
the existing fine/coarse projection decomposition. It preserves the original
data and checks hashes before and after execution. Recombination, density
evolution, thermal evolution and feedback coupling remain outside this static
experiment.

## Validation

The archived-source PBS body passed 126 tests and a 20-case, four-worker
structural smoke, including baseline reproduction, all photon ledgers, input
checksums and four common-mesh diagnostics. Native modules were mocked locally;
this does not test Gadi scheduling. `late_validation.json` records the details.
The distance controller was also run on the native32 dose0.9 fixture with a
1% bias criterion; all three convergence conditions pass at8L/3. Its actual
results and solver hashes are in `distance_control_validation.json`.
