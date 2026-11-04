# Local validation: analytic absorption / SCRIPT / C²-Ray

Executed 2026-11-03 on local CPUs. All three methods completed on two fresh native 21cmFAST snapshots: seed 222, **50 cMpc, 16³**, z=8 and 7.5. These are genuine native fields, not synthetic random cubes and not a high-resolution 256-cMpc result.

## What changed

- The new model uses neutral-depleting analytic column absorption and characteristic paths to actual voxel faces. Each full chord is absorbed once. Shared neutral atoms couple simultaneous directions; unequal chords use a controlled local integration.
- This replaces the initial interpolated propagation prototype, whose one-cell slab absorption was 7.49% low. The revised slab agrees with the analytic result 0.51011987435525 to within 1.1e-16 for 1/2/4 cells across the same slab and 1/16 exposure batches.
- The module is an opt-in research workflow. The existing production ionization driver and physical defaults are unchanged.

## Common inputs and differences

All backends consume exactly the same native cumulative `n_ion` and atom-capacity arrays at each redshift; hashes and native input parameters are in `validation_summary.json`. The original native brightness routine supplies all 21-cm fields. No source-budget renormalization or matched-mean adjustment is applied.

Density is frozen and initially neutral. Recombination, collisional ionization, thermal/spin-temperature evolution, heating feedback, separate helium chemistry and finite light travel time are absent. The historical native effective H+He-nuclei count convention is retained, with H opacity per effective atom. Analytic transport and C²-Ray share the declared monochromatic 18-eV cross-section; SCRIPT has no optical-depth equation.

The analytic run uses 96 directions and 32 exposure batches. Actual pinned pyC2Ray CPU Fortran transport and chemistry are executed through the external driver (32 exposure steps, iteration tolerance 1e-6, up to 2000 iterations). Original upstream files are byte-verified. SCRIPT uses the existing original unmodified Fortran allocation kernel.

Analytic and C²-Ray trace one periodically wrapped source-centred cube, without recycling photons leaving that tracing domain. Analytic escape is explicitly measured. SCRIPT closes a periodic photon box, so its boundary treatment differs. C²-Ray native surface-loss counters are retained but are not independent escape measurements.

## Measured comparison

MAE is the unweighted mean absolute difference of xHI, on its 0–1 scale. These are model differences, not known errors against a true field.

| z | Analytic xHI (mass) | SCRIPT xHI (mass) | C²-Ray xHI (mass) | MAE vs SCRIPT | MAE vs C²-Ray |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.62626 | 0.62622 | 0.63266 | 0.01784 | 0.01637 |
| 7.5 | 0.47743 | 0.47522 | 0.48843 | 0.04037 | 0.02423 |

Maximum absolute 21-cm power-ratio differences in the **three shells** k=0.1604, 0.2803, 0.3939 cMpc⁻¹:

| z | Analytic / SCRIPT | Analytic / C²-Ray | Analytic escape / emitted |
|---:|---:|---:|---:|
| 8 | 3.04% | 11.66% | 0.0112% |
| 7.5 | 7.25% | 8.44% | 0.4203% |

Independent analytic photon-ledger residuals are at most **7.49e-17** relative to emitted photons, including escaped, remaining and unemitted photons. This verifies counting, not field accuracy.

C²-Ray is not assumed exact: its field-derived emitted-minus-ionized budget differs from zero, and independent escape/reservoir measurements are unavailable. The report leaves its strict conservation residual undefined instead of assigning this deficit to escape. A separate contained-front test gives a roughly 6.3% native deficit even after exposure refinement; this is specific to this pinned coarse-grid backend/setup, not a universal C²-Ray accuracy claim.

## Numerical sensitivity at z=7.5

| Refinement | xHI MAE | Max 21-cm power change in the three lower-k shells |
|---|---:|---:|
| d24_e32_vs_d96_e32 | 0.018607 | 2.692% |
| d96_e32_vs_d384_e32 | 0.012116 | 0.830% |
| d96_e16_vs_d96_e32 | 0.001022 | 0.178% |
| d96_e32_vs_d96_e64 | 0.000498 | 0.089% |

The 96→384-direction maximum change across **all** retained shells is 3.19%. The dose changes decrease from 16→32 to 32→64; angular sensitivity remains larger. The single voxel-column closure, finite angular sampling and unresolved subcell geometry remain approximations. Neither a tiny ledger residual nor a half-Nyquist cut establishes physical/spatial accuracy.

Geometric MFP comparisons use 10,000 common random variates per phase and retain capped rays; pair summaries and sampling uncertainty are included in the JSON. They measure thresholded bubble/island geometry, not the optical absorption mean free path.

## Validation and use

The optional z=7 C²-Ray sample was manually stopped after 23/32 steps and approximately 1382 s to bound the additional reference cost. It did not produce a completed field, is excluded from all comparison tables, and is not labelled a numerical failure or a timeout. The two reported epochs completed all three methods.

- 113 targeted tests passed across the five new test files; Ruff and shell syntax checks passed.
- A fresh git archive → Gadi preparation → local execution of the generated PBS script → all three backends → frozen-source verification passed. This was a **local workflow test**, not an actual Gadi submission.
- `README.md` contains the Gadi command and explicitly labels its optional 16³ filtering.
- Full input identity, settings, numerical summaries and reference diagnostics are in `validation_summary.json`. Spatial convergence is not certified.

## Common-realization grid sensitivity

At z=7.5, all three methods were rerun on conservative 8³ source/atom block means from the same native 16³ realization. For comparison, fine xHI was atom-weighted onto the coarse grid and fine brightness volume-averaged; physical source and atom totals are identical. This changes resolved input substructure as well as transport resolution.

| Method | xHI MAE, N8 vs restricted N16 | Power ratio at k=0.1604 | Power ratio at k=0.4660 |
|---|---:|---:|---:|
| analytic | 0.03984 | 1.0111 | 1.4930 |
| script | 0.04643 | 1.0536 | 1.4262 |
| c2ray | 0.03257 | 0.9873 | 1.5045 |

Only the first shell is below half the coarse-grid Nyquist frequency. Near the coarse-grid scale, all three methods vary substantially. These two resolutions do not certify a converged morphology or establish that one code is universally more accurate. The native ES restriction control gives MAE=0 and every power ratio=1.

## Gadi N16 baseline and the N32 verification

The user-supplied report from
`analytic_pc_01aa674c8b_20261104T035222_bZTV/results` completes all three
backends for seed 222 at z=8 and 7.5. It uses the original 256-cMpc N128 native
archive, filtered to N16 (16-cMpc cells), with D96 and 32 exposure batches.

| z | xHI MAE vs SCRIPT | xHI MAE vs C²-Ray | Maximum 21-cm power difference vs C²-Ray, k < 0.09817 cMpc⁻¹ |
|---:|---:|---:|---:|
| 8 | 0.000356 | 0.003784 | 6.04% |
| 7.5 | 0.002696 | 0.012641 | 9.69% |

The analytic emitted-normalized ledger residuals are 6.50e-17 and 7.64e-17.
C²-Ray's field-derived emitted-minus-ionized fractions are 0.1153% and
0.3591%; its independent escaped/reservoir counts are unavailable, so its
strict conservation residual remains undefined. The table measures differences
between methods, not error against a known exact field.

The grid-verification launcher keeps the same original input and numerical
controls, selects z=7.5, and increases all three methods to N32. It reuses the
completed N16 baseline and automatically produces a checksum-verified
comparison on the common coarse grid. The exact submission command is in
`README.md`. This launcher update does not change any absorption or transport
kernel. The N32 scientific outcome must be read from the completed new Gadi
run; the baseline alone cannot establish the cause of the 9.69% difference.

The grid-analysis update passes 80 targeted tests across preparation, comparison
and grid comparison. Its CLI was also executed against the unchanged, real
50-cMpc N8/N16 archive above: it reproduces all three xHI MAEs, verifies equal
physical source/atom totals, and returns zero native-ES restriction MAE. The
native-ES brightness power ratios differ from unity by at most 3.43e-8 from
floating-point averaging. This tests the analysis workflow; it is not a new
256-cMpc N32 simulation.
