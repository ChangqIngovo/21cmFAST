# Count-consistent photon-conservation comparison

Adaptive source-dependent angular events give the best accuracy/cost compromise among the implemented candidates in these frozen-field experiments. They preserve original sources and ray directions, share one exhaustion clock across sources, and retain photons that reach the explicit distance limit. This is a research recommendation, not certification of full Qin20 evolution.

## Common inputs and physical scope

Every candidate receives identical photon counts, total atom capacities, and initial ionized counts. The independent ledger checks emitted = absorbed + explicitly unabsorbed and verifies local capacity bounds. The ES control is deliberately nonconserving; its missing photons fail this check instead of being declared a radiation reservoir.

The reference is the independently audited event-front model with many angles. It assumes point emitters at voxel centres, constant emission within a step, infinite light speed, opaque remaining atoms, and no recombinations. Finite-angle convergence is measured separately. It is not full radiative-transfer truth. The isolated-source continuum sphere is independently integrated over cell volumes.

Toy fields use 24^3 cells. The supplied native field uses 32^3 cells from a coarse-history Qin20 preparation at z=7.5; density and X-ray ionization are frozen. Its source is the NEW density-weighted, species-normalized conditional source closure, not unchanged native filtered E-INTEGRAL fields. Current-step recombinations, source feedback, and subsequent thermal evolution are absent. Doses 0.2 and 0.6 rescale source amplitudes at fixed density and are not other redshifts.

All candidate timings use one worker with compilation excluded. Wall times on a shared host are exploratory and fluctuate. Large-grid/Gadi costs and end-to-end PC histories remain unmeasured by this suite.

## Native frozen-field accuracy and cost

Final reference: 768 -> 3072 angles per source. At the actual dose, their L1 difference is 0.0546% of emitted photons. The table uses the final 3072-angle reference and single timed scalar-candidate passes. Adaptive rows were refreshed after the final cavity-predictor fixes using the median of three passes; saved high-angle references were reused. Earlier repeated timings were consistent in order of magnitude.

| Method | Seconds | L1 absorption / photons | Resolved neutral-count auto-power discrepancy | Explicit unabsorbed / photons |
|---|---:|---:|---:|---:|
| es-control | 0.02539 | 10.924% | 5.893% | 0.000% |
| es-global | 0.0385 | 7.579% | 24.850% | 0.000% |
| es-front | 0.02735 | 7.953% | 21.583% | 0.000% |
| radial-shell | 24.87 | 3.516% | 5.331% | 0.000% |
| voronoi | 0.5573 | 5.066% | 8.691% | 0.000% |
| adaptive-event | 0.5795 | 0.388% | 0.168% | 0.281% |

The auto-power metric uses mode-count-weighted shell-mean power of remaining neutral atom counts, wavelengths >=4 cells, and shells with >=20 full-FFT modes. It is a frozen saturated-spin morphology proxy; it does not include spin-temperature or velocity effects. Residual-field power and cross coherence are also recorded because equal auto powers do not establish equal morphology.

At dose 0.2, 0.6, and actual 0.82415 photons per available atom, adaptive-event L1 errors are 0.0604%, 0.3475%, and 0.3884%; reference angular changes are 0.0098%, 0.0537%, and 0.0546%. These small errors are relative to the voxel reference and do not bound its physical error.

The separate native 21cmFAST ionization-only measurement on the same fixture used four threads: ES median 13.14 s and the previous global-budget mode 16.72 s. Those native calls include source integration, radius filtering, and recombination work. Adaptive transport alone took about 0.58 s, suggesting affordable overhead on this small fixture, but different physics and timing scopes prevent a fair whole-algorithm speedup claim or extrapolation to Gadi.

## Does it restore missing photons?

At the actual native dose, the matched-source ES control leaves 11.139% of photons unaccounted for. The high-angle reference has an explicit 0.276% distance-limited radiation remainder, and adaptive-event 0.281%. Thus it restores most of the control deficit instead of imposing a lower unrelated global budget. The reconstructed neutral fractions are:

| Method | Volume mean xHI | Mass mean xHI |
|---|---:|---:|
| reference | 0.193962 | 0.169528 |
| es_control | 0.284595 | 0.254740 |
| adaptive_event | 0.193998 | 0.169565 |

The original native output had volume mean xHI=0.286674. Its closeness to the additive-source ES control (0.284595) is informative, but the experiment is not a recombination-consistent rerun of the full native history. In this controlled comparison PC makes the gas more ionized by about 0.0906 in volume mean xHI. More ionization is not a universal axiom: a nonconserving baseline can also overcount, and different sink/source measures invalidate that inference.

## Geometry tests reject scalar conservation as sufficient physics

| Case | ES global L1/Q | ES front L1/Q | Radial shell L1/Q | Voronoi L1/Q | Adaptive event L1/Q | Reference angular change |
|---|---:|---:|---:|---:|---:|---:|
| all-partial | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.000% |
| isolated | 26.28% | 26.28% | 15.28% | 15.28% | 2.06% | 0.109% |
| equal-overlap | 25.89% | 25.89% | 13.14% | 12.89% | 2.03% | 0.099% |
| unequal-overlap | 34.75% | 34.75% | 13.51% | 14.98% | 2.19% | 0.090% |
| transparent-overlap-increment | 193.76% | 193.76% | 137.29% | 137.29% | 1.96% | 0.581% |
| opaque-plane | 115.92% | 115.92% | 134.14% | 134.14% | 2.47% | 0.718% |
| dense-correlated | 18.91% | 16.09% | 12.89% | 19.90% | 0.65% | 0.314% |

The transparent-overlap case starts with two binary ionized spheres of radius 4 cells separated by 6 cells; a source adds only 0.01 photons at the first centre. L1/Q can approach 200% because it measures the placement of the small increment, not the almost unchanged total ionized field. Scalar methods put the added photons on the wrong parts of the front. The continuum 12.5% cap fraction applies to direction labels; the recorded absorber-centre cone fraction is explicitly only a spatial diagnostic and does not independently certify exact 12.5% propagation through voxelized bubbles.

In the opaque-plane case only rays that hit the plane before the 12-cell distance limit may absorb. The independent geometric result is (1 - 3.5/12)/2 = 0.3541667. All four scalar conserving methods absorb 100% in the plane: a photon ledger passes while the angular constraint fails. Adaptive-event absorbs 35.4167% and explicitly retains the remainder. Thus a global budget, nearest shell, or nearest-source partition cannot by itself guarantee even elementary opacity geometry.

Toy sparse references use 65536 -> 262144 angles; dense correlated reference uses 3072 -> 12288. The latter remains uncertain at 0.314% L1/Q and is explicitly a correlated lognormal toy, not a cosmological realization.

## Remaining accuracy limits

- The isolated radius-4 continuum sphere differs from the high-angle voxel reference by about 11.4% L1/Q. Consequently a 2% adaptive-vs-reference discrepancy is not 2% physical accuracy. The separate hybrid refinement study finds the expected decline with increasing radius in cells.
- The adaptive angular radius predictor is a heuristic. Transparent cavities and collective fronts can make initially weak sources important far away; the existing angle cap is not a rigorous error bound. Spatial refinement, angular refinement, and time partition must remain separate checks. The separate temporal study predicts about 29.8 million potential source-angle packets in the second half-step of the native test, and declines that allocation under its explicit resource cap. Therefore the fast neutral-coeval results do not establish cheap incremental evolution through already transparent cavities.
- The finite-distance remainder is a radiation reservoir/support cutoff, not ionized atoms. Recycling or discarding it in a native history requires an explicit physical closure.
- Native current-step recombinations, cross-snapshot photon/state bookkeeping, feedback response, and density evolution require separate integration tests. The exact photon ledger alone does not validate these processes.
- Only one frozen 32^3 native field and a few small geometries are covered. No large-volume, resolution-converged, multiple-seed, Gadi scaling, or full end-to-end Qin20 validation is established here.

## Reproduction

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/pc_reconsidered/compare_models.py --reference-angles 65536 --dense-reference-angles 3072 --repeats 3 --output research/pc_reconsidered/comparison_final_results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/pc_reconsidered/compare_models.py --fixture /path/to/source_capacity.npz --fixture-dose-fractions 0.2 0.6 --cases native-dose-0.2 native-dose-0.6 native-frozen --dense-reference-angles 768 --output research/pc_reconsidered/comparison_native_refined_results.json
python -m unittest discover -s research/pc_reconsidered -p test_compare_models.py -v
python research/pc_reconsidered/plot_comparison.py
```

Eight independent benchmark/oracle tests pass. Every PC candidate and reference in the final JSONs passes independent count accounting and capacity bounds. The baseline ES control intentionally fails when photons are missing. JSONs retain exact inputs hashes, source-code hashes, candidate audits, reference convergence, and per-shell spectra. Optional binary fields are reproducible scratch outputs and need not be committed.
