# Sources of the emulator prediction error

The late-stage error is primarily amplified by an inaccurate absorbed-photon
count and its interaction with global projection. During the active phase,
correcting the count leaves most spatial-field error intact. These conclusions
come from controlled interventions on fixed model weights, not a newly fitted
model.

## Experiment

The model selected in commit `8d6d656d` and all its saved training artifacts were
kept unchanged. Tests use the same 28 snapshots from seeds 2109–2112, with native
21cmFAST inputs in a 50-cMpc box on a 16³ output grid. The 24 active snapshots
have `0.05 < teacher mass-weighted xHI < 0.95`; the four late snapshots are at
z=6. Power errors average shells and then snapshots, using individual modes
`0 < k < 0.5 k_Nyquist`. Teacher labels enter the controls below only to diagnose
existing errors.

Four controls were evaluated:

1. The unchanged deployed prediction.
2. The same final latent field projected to the **teacher absorbed count**.
3. The teacher count imposed at every frozen-base and final projection, with
   field weights unchanged.
4. The teacher field projected to the **predicted absorbed count**.

The fourth control asks how close a field could remain to the teacher while
using the wrong global count. With capacity-weighted L2 projection it minimizes
capacity-weighted field MSE at that count. It is not a unique budget contribution
to power error, and the interventions do not give an additive causal partition.

## Accurate counts remove much of the late error

| Mean low-k power-ratio error | Active: 24 snapshots | z=6: 4 snapshots |
| --- | ---: | ---: |
| Deployed prediction | 15.91% | 254.77% |
| Correct final count; unchanged latent | 15.48% | 50.72% |
| Correct counts through the frozen-base chain | 14.14% | 49.51% |

Late xHI RMSE falls from 0.06129 to 0.04017 with the final count corrected, and
to 0.03724 with the base corrected as well. Active RMSE changes much less:
0.06687 → 0.06585 → 0.06500. Thus the count correction helps late predictions
substantially, while most active spatial error persists.

All controlled fields remain bounded. Their independent photon residuals are
at most `4.76e-17`. The deployed ledger already passed: its failure is the
prediction of **how many photons are absorbed versus left unabsorbed**, not
failure to add those predicted reservoirs correctly.

Results: [oracle_budget.json](error_sources_20261116/oracle_budget.json).

## Why a small absolute neutral-fraction error gives a large power error

At fixed density and redshift, the current brightness law is `B = C(z) rho xHI`.
Writing `delta_B = B/mean(B) - 1` gives, for nonzero means and reference powers,

```text
P_pred/P_teacher = (mean(B_pred)/mean(B_teacher))²
                  * P_delta,pred/P_delta,teacher.
```

The mean ratio is also the mass-weighted neutral-fraction ratio because both
fields share the same density. For seed 2112 at z=6, teacher xHI is 0.004098,
while the model predicts 0.018768. The neutral-mass ratio is 4.58. Its squared
mean factor is 20.98; the mean fractional-contrast power ratio is 0.407. Their
product is the observed mean power ratio **8.54**, or a 754% mean excess.
Correcting only the final absorbed count reduces this sample's power error
from 753.6% to 34.8%.

The fractional-contrast term still contains density–ionization correlations;
it is not pure geometry. A second exact shellwise decomposition uses signed
coherent response `a = real(P_cross)/P_teacher` and orthogonal power
`P_perp = P_pred - P_cross²/P_teacher`. Both identities are checked numerically.
Zero predicted means retain their zero power and undefined normalized contrast.

Results: [power_attribution.json](error_sources_20261116/power_attribution.json).

## Three findings about the absorption estimator

**A flat output branch.** The deployed scalar head uses
`eta = 1 + min(tanh(logit), 0)`. Positive logits produce eta=1 with zero local
gradient. This occurs in 19/28 external snapshots; in ten of them the teacher
efficiency deficit exceeds `1e-5`. When photons exceed the atom count, eta=1
forces every cell to be fully ionized. Seed 2109 at z=6 is one such case.

**Limited training coverage.** The 35 training snapshots cover
`eta >= 0.93151`, whereas external z=6.5 examples reach 0.89647 and 0.91464.
The largest mass-ionization error, 0.03388, occurs in this lower-efficiency
region and is not caused by the flat branch. Only three external snapshots
leave any individual training-feature range; their typical nearest-training
distances are small. The evidence supports sparse coverage of the relevant
input–absorption relation, rather than a blanket claim of input distribution
shift or proof that the chosen summaries cannot encode it.

**The absolute-error objective and optimization.** At z=6, the mean absolute
mass-neutral error is 0.00587, but its mean relative error is 156%. An absolute
absorbed-fraction loss does not assign the same importance to this small
neutral component as its fractional signal error would.

The new sigmoid candidate avoided the flat branch but had worse validation
performance. Its best checkpoint was the last requested epoch, 200, with loss
still decreasing. Twenty leave-one-training-seed-out probes compared 200 and
2000 epochs without using validation/external labels for fitting. Longer
sigmoid training reduced relative-neutral MAE from 1.007 to 0.277, but increased
absolute mass MAE from 0.00562 to 0.00736. Additional epochs alone therefore
did not consistently improve generalization. No candidate was promoted from
these diagnostic results.

Results: [budget_head.json](error_sources_20261116/budget_head.json) and
[feature_support.json](error_sources_20261116/feature_support.json).

## Projection can erase information; active errors also require better ranking

For seed 2109 at z=6, the final latent field contains 4096 distinct scores.
The predicted all-absorbed budget nevertheless maps all 4096 cells to xHI=0.
The frozen base is also saturated. Correcting the final count alone gives a
minimum shell cross-correlation of 0.539; correcting the base count as well
raises it to 0.916. Final-only versus recursive power errors are 85.4% and
43.6% for this sample. Thus an inaccurate budget affects both the final count
and the spatial information available to the residual model.

A further diagnostic fitted the best atom-weighted monotone map from fixed
scores to teacher xHII, grouping identical scores before fitting. It preserves
the exact teacher absorbed count. This uses the full teacher field and is
an oracle lower bound for **capacity-weighted MSE**, not a deployable method or
a lower bound on power error.

| Capacity-weighted MSE | Active | z=6 |
| --- | ---: | ---: |
| Deployed prediction | 0.005435 | 0.003104 |
| Best monotone map of deployed latent | 0.004799 | 0.000856 |

In the active phase even this ideal remapping lowers MSE by only 11.7%.
The remaining mismatch cannot be repaired by a monotone global recalibration
of those scores; their spatial ordering must change. This identifies a field
representation/prediction limitation without identifying a unique missing
feature or neural architecture.

For the saturated seed-2109 example, the oracle using latent scores has MSE
0.000583; the final projected field has only one tied score and can produce
only a uniform oracle field, with MSE 0.001330.

Results: [ranking.json](error_sources_20261116/ranking.json).

## Measured angular and exposure sensitivity is smaller than the emulator discrepancies

Each z=6 teacher was repeated, then rerun with 96→384 directions and separately
32→64 exposure steps. The original configuration reproduced all four saved
fields bitwise. Twelve runs took 294 s; all completed with maximum independent
photon residual `1.15e-16`.

| Seed | Angular refinement MAE | Exposure refinement MAE | Emulator MAE | Low-k PS change: angle / exposure / emulator |
| --- | ---: | ---: | ---: | --- |
| 2109 | 0.000614 | 0.000293 | 0.003689 | 3.69% / 11.58% / 100.00% |
| 2110 | 0.000732 | 0.000270 | 0.007699 | 10.18% / 9.65% / 131.04% |
| 2111 | 0.002078 | 0.000627 | 0.012494 | 5.89% / 3.38% / 34.48% |
| 2112 | 0.000995 | 0.000350 | 0.021050 | 4.50% / 8.27% / 753.58% |

Using the refined teachers changes emulator MAE only slightly. Angular and
exposure sensitivity therefore cannot account for the observed model errors
in these cases. This comparison does not determine the teacher's spatial-grid
error or certify full transport convergence.

Results: [teacher_sensitivity.json](error_sources_20261116/teacher_sensitivity.json).

## Implementation and next experiment

Only diagnostic code and tests were added. The deployed field/budget networks,
projection, teacher kernel and original results were not changed. Thirty new
targeted tests pass, covering budget interventions, tied-score monotone fits,
endpoint cases, analytic Fourier identities and counterfactual conservation.

The next model change should separately test (i) a trainable nonnegative
absorption-deficit target with absolute and relative-neutral validation, and
(ii) field features or spatial models that change the score ordering. More
independent late-stage source configurations are required to cover the current
efficiency gap. The already inspected seeds 2108–2112 are now regression data
for that development; new seeds should supply the next untouched final test.

Reproduce the principal controlled experiment from the repository root:

```bash
python -m research.pc_analytic_emulator.diagnose_errors \
    --model /absolute/emulator_results_smooth/model \
    --teacher /absolute/teacher_results_external \
    --output /absolute/new_oracle_diagnostics \
    --seeds 2109 2110 2111 2112
```

Additional standalone diagnostics are
[`decompose_power.py`](../pc_analytic/validation_20261116/reproduction/decompose_power.py),
[`diagnose_budget.py`](../pc_analytic/validation_20261116/reproduction/diagnose_budget.py),
[`diagnose_ranking.py`](../pc_analytic/validation_20261116/reproduction/diagnose_ranking.py)
and [`diagnose_late_teacher.py`](../pc_analytic/validation_20261116/reproduction/diagnose_late_teacher.py).
Each exposes its required input paths through `--help`. Full control fields,
refined teachers, budget probes and source/receipt hashes are retained with
the diagnostic results.
