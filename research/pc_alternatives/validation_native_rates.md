# Native Qin20 instantaneous UV rate validation

The rate adapter is executable and passes its unit and native-field checks. This does **not** reduce the remaining event-front integration to interface plumbing: using the native SFRD as a UV rate changes the temporal photon model relative to the native cumulative `nion` construction, and a single additive grid also changes its spatial source closure. Source normalization, feedback timing, gas/radiation history and chemistry still need explicit choices and controlled validation. Production files were not changed.

## Scope and reproducibility

`validation_native_rates.py` reads the existing 32³ frozen step at z=8→7.5. The frozen IC seed is 1234, BOX_LEN=200 cMpc, DIM=96 and N_THREADS=4; nodes are [35,12,8,7.5]. The installed `InputParameters.from_template('Qin20')` astrophysical, matter and cosmological parameters match exactly; the existing `native_benchmark.json` records empty physical-difference dictionaries. In particular EH and V_CB_NONE are template settings, not changes introduced by this audit. The numerical grid and very coarse chronology are research choices, not a converged Qin20 realization.

The original frozen fixture omitted EmissivityFields and TsBox at z=8. A short replay from its saved ICs recovered these, reaching z=8 in 19.39 s. Density, neutral fraction, cumulative recombinations and both stored conditional cumulative source histories matched the original z=8 fixture bitwise. Current z=7.5 emissivity was then evaluated with the actual z=8 feedback. All frozen input SHA256 hashes were checked unchanged.

Available native fields are `n_ion`, `sfrd_acg`, `sfrd_mcg`, and `xray_emissivity`. `fesc_weighted_sfrd` is not allocated for this E-INTEGRAL configuration. Recovered HDF5 and adapter arrays are in `/workspace/scratch/40b418003e4a/validation_native_rates_grid32`; the reproducible summary is `validation_native_rates.json`. Run the script with `--extract` when those recovered files are missing. The scratch fixture must be present.

## Units and the executable adapter

For species j and Eulerian cell i, the adapter returns

\[
\dot q_{i,j}=\frac{f_{{\rm esc},j}N_{\gamma,j}}{\bar\rho_b}
                 {\rm SFRD}_{i,j},\qquad
\Delta q_i=\int_{t_n}^{t_{n+1}}\sum_j\dot q_{i,j}\,dt.
\]

`hmf.c:sfrd_conditional_*` and `sfrd_unconditional_*` divide by `sfr_timescale=t_STAR/hubble(z)` in **seconds** and multiply by `RHOcrit*OMb` in Msun/cMpc³. Hence the C arrays store **Msun s⁻¹ cMpc⁻³**, not Msun yr⁻¹ cMpc⁻³. `EmissivityFields.c:get_integral_properties` already multiplies these fields by 1+δ for Eulerian sources. Multiplying the adapter by density again is wrong. Here ρ̄_b=6.220220637×10⁹ Msun/cMpc³.

Actual Qin20 parameters are t_STAR=0.5, ALPHA_ESC=0, log10 fesc_ACG=−1.22, log10 fesc_MCG=−2.22, POP2_ION=5000, POP3_ION=44021. With constant escape fractions the species SFRD multiplication is valid. The adapter rejects nonzero ALPHA_ESC: that requires the mass integral of fesc(M)×SFR(M), not an arbitrary average escape fraction. The Lagrangian `fesc_weighted_sfrd` definition already incorporates the photon yield; it must not be multiplied by that yield a second time.

The adapter's q is a dimensionless **native-effective source count per mean-density cell**. Native Q and capacity ρ(1−xe) may consistently share that effective convention, but this audit does not identify it with a physically calibrated number of H absorptions. Under a separately declared yield-per-stellar-nucleon interpretation, multiply q by ρ̄_b Vcell Msun/mp to obtain emitted photon counts. An independent direct SFRD×volume×time×yield conversion agrees to 2.28×10⁻¹⁶ relative error. For photons per mean H atom the conversion additionally divides by X_H=1−Y_He; here 1/X_H=1.315789. `Constants.h:No` is H nuclei and `N_b0=No+He_No` is H+He nuclei. These are not interchangeable with total nucleon number. Helium absorption, secondary ionizations and sink normalization remain separate physics.

At z=8 and z=7.5, the measured identity

\[
t_{\rm SFR}\left\langle\frac{\dot q_{\rm ACG}+\dot q_{\rm MCG}}{\rho}\right\rangle
\simeq\langle n_{\rm ion}^{\rm instantaneous}\rangle
\]

has ratios 0.9999999034 and 0.9999999048. This validates seconds, species yields, escape factors and the density placement using actual native fields. It is a mean identity: the cellwise RMS differences divided by mean nion are 4.60% and 4.91%. Native combined nion and individual SFRD fields receive independent mean corrections.

`mean_fix_emissivities` normalizes **mean(SFRD/ρ)**, not volume-integrated SFRD. The ratios Σqdot/[Ncell mean(qdot/ρ)] are ACG 1.12183→1.13320 and MCG 1.03273→1.02970. Forcing the integrated source sum back to the conditional-mean budget is an additional global normalization model. It is not a unit conversion.

## Two native clocks

Native `hubble()` includes radiation and determines the SFR timescale; `dtdz()` uses a matter+Λ expression without radiation. `CosmoParams.OMl` is exactly 1−OMm. The audit records both native clocks without attributing either to Astropy:

| Clock for z=8→7.5 | Duration |
|---|---:|
| Integral of native dtdz, used for the primary table below | 57.312311 Myr |
| Integral of 1/[(1+z)H_native(z)], with radiation | 57.243075 Myr |
| Native late-endpoint dtdz×Δz approximation | 61.537624 Myr |

The first two differ by 0.12095%; this is a convention difference, not numerical quadrature uncertainty. SFR unit checks always use the actual radiation-inclusive native H. Applying the Hubble interval to the same endpoint rates changes the total rate/legacy ratio from 0.5091505 to 0.5085355, which cannot explain the much larger source-model difference.

## Measured interval: cumulative nion is not an instantaneous rate

All entries below are sums of native-effective counts over the 32³ grid, not absolute physical photons.

| Source | Legacy cumulative ΔQ | Original endpoint SFRD trapezoid | Rate integral / ΔQ |
|---|---:|---:|---:|
| ACG | 4727.0875 | 1782.2493 | 0.377029 |
| MCG | 942.3204 | 1104.3328 | 1.171929 |
| Total | 5669.4079 | 2886.5821 | 0.509151 |

Both species' rate-derived increments are positive in every cell without clipping. Their endpoint total rates in s⁻¹ are ACG 8.5746288×10⁻¹³→1.1133974×10⁻¹² and MCG 6.2951360×10⁻¹³→5.9168794×10⁻¹³. Holding the original left or right rates throughout the interval would give totals 2689.3462 or 3083.8179. These are diagnostics, not rigorous bounds without monotonicity between samples.

The original endpoints use different previous-feedback states: emissivity at z8 uses feedback z12, while z7.5 uses feedback z8. To separate this from quadrature, a second native calculation held z8 TsBox and IonizedBox for left/mid/right evaluations. The left value was evaluated at z=8−10⁻⁶ with the corresponding newly perturbed density, because the wrapper correctly requires z12 feedback at exact z8. The offset is about 107 years in a 57 Myr interval. The midpoint z=7.74109104 is at half the native dtdz elapsed time. This is a held-feedback diagnostic; it is not a finer, coupled native history.

| Source | Held-feedback trapezoid | Held-feedback Simpson | Simpson/trapezoid | Simpson/legacy ΔQ |
|---|---:|---:|---:|---:|
| ACG | 1782.2497 | 1777.3626 | 0.997258 | 0.375995 |
| MCG | 1048.5401 | 1054.1794 | 1.005378 | 1.118706 |
| Total | 2830.7898 | 2831.5420 | 1.000266 | 0.499442 |

Updating the left MCG feedback from z12 to z8 lowers its rate by 9.8008%; the ACG change is 5.51×10⁻⁷ fraction, including the small time offset. Within this particular held-feedback family, midpoint sampling barely changes the integral. Thus the discrepancy in this example is not mainly a trapezoid-versus-Simpson artifact. It is **not** evidence that all Qin20 histories have a 50% source error, or that either model is scientifically calibrated by this one test. More temporal samples and coupled feedback refinement remain necessary.

The distinction also follows from native equations, independently of coarse sampling. For a fixed population and escape/yield factors, the instantaneous source proxy is proportional to stellar mass, while qdot is proportional to that mass divided by t_SFR. Its time derivative equals qdot only under the additional growth condition d ln(Mstar)/dt=1/t_SFR. Native cumulative source histories instead update the previous history by a difference of conditional nion integrals, evaluated at old/new growth with current turnover thresholds, followed by their native normalizations. They do not integrate the stored SFRD over time. Moreover, `compute_emissivity_fields` uses previous spin feedback while native ionization uses the current spin box; the wrapper explicitly documents this distinction.

Differencing separately budget-normalized cumulative packet grids also fails a general local-positivity guarantee: the MCG map has one negative cell increment, −0.00338260, while its global increment is +942.3204. This tiny mass does not establish a large numerical effect, but it is a concrete counterexample to interpreting arbitrary cellwise cumulative differences as nonnegative emitted photons. The adapter neither uses these differences nor clips them.

## Integration boundary and remaining acceptance tests

The clean research boundary is **actual instantaneous emissivity fields → positive time-integrated source increments → transport and chemistry with persistent state**. The minimal adapter is implemented at that first boundary. `compute_ionization_field` currently discards supplied EmissivityFields for E-INTEGRAL and passes a dummy object into native ionization, so production integration needs an explicit alternative interface rather than merely passing a new array through an existing argument.

Required decisions and checks remain:

1. Choose the temporal source model. Raw native SFRD integration changes the legacy cumulative photon budget. If instead preserving native ΔQ, positive spatial rate weights times ΔQ/Σweights define a separate budget-preserving model. It requires an explicit policy for any negative global ΔQ and must not be presented as raw SFRD physics.
2. Choose and validate the additive spatial localization. Native scale-dependent conditional histories are not convolutions of a single positive source grid, as recorded in the separate native-source audit.
3. Specify feedback timing and demonstrate redshift-step convergence with coupled Ts, LW and photoheating evolution. A held-feedback Simpson check does not prove coupled-history convergence.
4. Carry prior ionized atoms, recombinations and any remaining radiation state across steps. Emitted cumulative counts cannot be reinjected, and current xHII cannot be equated to lifetime absorbed photons once recombinations occur. Evolving density introduces an additional gas-advection/accounting problem.
5. Validate Γ, helium/X-ray conventions and recombination normalization with the atom ledger and applicable physics. This source-only adapter proves none of these couplings.
6. Compare the new transport with a reference RT method on the **same source increments, capacities, emission history, sinks and light-speed convention**. Then perform chronological matched-source tests before a full Qin20 PC/no-PC comparison. A native no-PC versus event-SFRD run changes source closure as well as transport; its power-spectrum difference cannot be attributed solely to photon conservation.

Nine adapter tests pass: units/species factors, declining-rate positivity, exact integration of a linear rate under interval subdivision, cumulative ledger addition, rejection of negative rates, rejection of unsupported mass-dependent escape, invalid-input rejection, native-clock additivity/analytic-age agreement, and escape saturation/zero yield. Native assertions additionally verify the mean source identity, independent physical unit conversion and unchanged frozen inputs. No transport, sink or production end-to-end claim is made by these tests.
