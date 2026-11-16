# Development findings

The selected emulator uses 88 native-input field features and model seed 83.
On the same 28 validation snapshots, it reduces the projected validation
objective by **28.70%** and field MSE by **40.94%** relative to retraining the
24-feature model on the same expanded dataset.

| Model | Common validation objective | Full-grid field MSE |
|---|---:|---:|
| Unchanged previous model | 0.272472507 | 0.007232253 |
| Retrained, 24 features | 0.140976060 | 0.006004078 |
| Selected, 88 features | **0.100520162** | **0.003545936** |

The two retraining configurations differ only in `feature_set`. They share
164 training snapshots, 28 validation snapshots, hidden widths, optimization
controls, objective and four initialization seeds: 11, 23, 47 and 83. Every
88-feature initialization has a lower validation objective than every
24-feature initialization. Relative to the previous model, the selected model
reduces the objective by 63.11%; that comparison also includes changes to the
training data and training procedure.

## What changed

The field input expands from 24 to 88 columns by adding 64 descriptors of the
native source, atom and ES fields. These include periodic face/edge/corner
moments, gradient magnitudes and alignments, face contrasts, and Gaussian
moments at one and two cells. They expose local spatial structure without
using teacher fields as inference inputs. The global budget still receives
the same 14 native summary features.

The ranking objective is the mean across complete validation cubes of a
front-weighted field MSE divided by `max(teacher variance, 0.005)`, plus
`0.1` times the equal-shell mean squared complex Fourier residual of the
21-cm brightness proxy, normalized by teacher shell variance with a floor.
Fourier modes extend to half Nyquist, with a shell-variance floor of
`1e-6` times fully neutral mean-square brightness; numerically zero-reference
shells are excluded. The auxiliary raw loss
has training weight `0.05` and is excluded from model selection. Thus the
objective tests spatial phase as well as amplitude; it is distinct from a
power-ratio error or the unweighted MSE reported in the table.

The saved model is `warm_started_residual`, with a `latent_linear` correction
to a frozen base. Seed 83 selects residual epoch 105 of 120. Its budget is
`tabulated`, and its projection is `capacity_l2`: the nearest bounded field
in the atom-capacity-weighted L2 metric at the predicted global absorbed
count. The budget and base remain frozen during residual fitting. Both
retrained finalists retain the same tabulated-budget file hash.

## Candidates that were not adopted

A deficit MLP did not improve the scalar budget objective. A separate fixed
sweep of 24 log/logit-deficit kernel regressors also failed to improve it:
the best kernel scored `0.003097277`, compared with `0.001410663` for the
tabulated model and `0.005115041` for the deficit MLP. All candidates and
training-only target-floor sensitivity results were retained. The kernel
has no production integration. Scalar budget ranking did not replace the
full-field validation criterion used for final model selection.

## Data separation and physical scope

Training uses seeds 2101–2112 and 2121–2128; validation uses 2113–2116.
Previously inspected seeds 2101–2112 are explicitly development data.
Blind seeds 2117–2120 and geometry-transfer seeds 2131–2132/2141–2142 are
disjoint and excluded from training and selection. The selected model files
and inference settings were frozen before the evaluation-start receipt.
This development report contains no blind or transfer scientific metrics.

The native 21cmFAST binary, physical parameters and teacher solver remain
unchanged. The main grid is 16³ in a 50 cMpc box, with native initial-condition
DIM=96. Teachers use cumulative native sources, 96 directions, 32 exposure
steps and `source_periodic_cube` boundaries with recorded escape. Native ES
uses `no-photoncons`. The calculation retains frozen coeval density and an
initially neutral medium, without recombinations, thermal evolution,
photoheating feedback or independent helium chemistry.

The independent audit of all 220 primary teacher snapshots gives a maximum
relative photon residual of `1.086e-16`. Emulator conservation is the global
accounting of predicted absorption and unabsorbed photons; its projection
does not supply source-resolved transport or independently measured escape.

Exact metrics, model hashes, split definitions, physics mapping and verified
source identities are recorded in [development_evidence.json](development_evidence.json).
