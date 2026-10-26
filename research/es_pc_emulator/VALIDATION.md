# Validation recorded 2026-10-27

Implementation base: `2a57851d5d28c2af6038c66932883293f5516bd5` on
`Photon_conserving`. Validation used genuinely compiled native 21cmFAST and
the pinned C2-Ray H+He backend, not mocked transport outputs.

## Checks

| Check | Result |
| --- | --- |
| New ES-to-C2-Ray module, including actual native diagnostic tests | 54 passed |
| Official input/template regression | 219 passed |
| Compiled C2-Ray invariant tests | 5 passed |
| C2-Ray build contract | 4 passed |
| New module and changed build adapter Ruff | Passed |
| Three Gadi shell/PBS files, Bash syntax | Passed |
| Diff whitespace checks | Passed |

The larger existing evolution/teacher/emulator regression run recorded
361 passes and 12 optional/native skips. One source-integrity test detected
files being edited during development; its isolated rerun after freezing
the sources passed. The genuinely compiled tests above exercise the relevant
native backends separately. Counts from different groups overlap.

The new radius diagnostic was enabled and disabled on otherwise identical
native runs. Neutral-fraction, source `n_ion`, and density arrays were bitwise
identical. Nonzero largest-crossing radii and the separate hashed sidecar
were verified. Unsupported photon-correction and entire-sphere modes reject
the diagnostic rather than returning a misleading field.

Compiled C2-Ray tests cover opaque/transparent primary photon closure,
one/four-thread agreement, periodic translation, coincident-source splitting
and stable time averages. Modern libgfortran required a logging-only FORMAT
correction in conservative-mode build copies; its patch and hashes are
recorded, while original upstream files and native-reference bytes remain
untouched.

## Genuine end-to-end dataset and fit

The complete generator ran three independent IC seeds:

| Seed | Split | Native sources | RT evolution / seconds | Maximum primary transport residual |
| --- | --- | ---: | ---: | ---: |
| 111 | Train | 4096 | 170.233 | 6.07e-13 |
| 222 | Validation | 4096 | 189.963 | 6.67e-13 |
| 333 | Test | 4096 | 191.738 | 8.22e-13 |

This was a plumbing test: 50 cMpc, native source/output mesh 16 cubed,
IC DIM=48, z=8 to 7.9, one source interval and one RT step, RT refinement=1,
angular order=4 (96 rays/source), one worker, frozen contrast at z=8.
The source histories, native ES xHI/radius fields, MPI C2-Ray calculation,
raw-binary audits, pair export and combined independent-seed manifest all
completed. Running the completed generator again with `--resume` reaudited
and reused all three seeds without emitting additional photons.

The full training CLI fitted all three feature arms with one neural seed,
up to 20 epochs, 1024 cells/snapshot and one 16-node hidden layer. Serialization,
full held-out reconstruction, field metrics and auto/cross spectra completed.
Validation selected the zero-correction ES baseline in this deliberately tiny
dataset; no improvement in physical accuracy is claimed. A separate
independent-seed synthetic correction test verifies that fitting can improve
held-out predictions and that inference does not read the target.

Local MPI validation used an unbound launcher inside the inherited CPU set
because this container cannot bind cores. Gadi retains its original binding
and MPI/OpenMP layout. These elapsed times are correctness-run measurements,
not a forecast for the default 48-cubed RT training histories or Gadi speedup.

## Interpretation

This establishes an executable dataset-to-training pipeline. It does not
establish cosmological spatial/angular/time convergence, useful NN accuracy
across source parameters, or strict conservation of neural predictions.
Generated labels retain `convergence_certified=false`; training on them
requires the explicit exploratory opt-in until convergence is independently
established. The native ES and teacher source/gas closures differ as documented
in the README, so the learned target is ES-to-RT emulation rather than an
isolated photon-loss correction.
