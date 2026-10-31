# Numerical improvements

Physical inputs remain the archived 21cmFAST sources, cumulative photon budgets,
gas trajectories, physical light speed and recorded effective-atom opacity.
Numerical changes are tested against independent solutions and then against
the same native histories. SCRIPT is an external comparison, not a convergence
target. Case-to-case changes measure numerical sensitivity, not a true-error
bound.

## 1. Continuous source--absorption coupling

The earlier sequence injected photons after its final absorption half step.
In optically thick cells this left an artificial endpoint inventory of about
`q * dt / 2`. It also contaminated a spatial refinement test: a smaller cell
reduces the CFL timestep and hence this inventory, even for a homogeneous field
with no spatial gradients.

The new local half step evolves neutral atoms `N`, directional photons `P_d`
and a constant isotropic emission rate `q` together:

```text
dN/dt   = -kappa * N * sum(P_d)
dP_d/dt = weights_d * q - kappa * N * P_d
```

Let `D = sum(P_d_initial) - N_initial` and

```text
F(s) = exp[-kappa * (D*s + q*s*s/2)]
J    = integral_0^h F(s) ds
K    = integral_0^h (h-s)*F(s) ds
T    = 1 / (1 + kappa*N_initial*J).
```

The exact constant-source solution is

```text
N_final   = N_initial * F(h) * T
P_d_final = T*P_d_initial + weights_d*q*T*(h + kappa*N_initial*K).
```

The directional incoming radiation keeps its own attenuation factor; it is
not isotropized. The code evaluates positive Gaussian moments after scaling
out their maximum exponent. Long-double intermediates and large-argument
moment expansions avoid overflow and loss of a weak source by subtraction.
The ion increment is evaluated as a positive moment expression, so sources
far below neutral-count roundoff still contribute their photons.

For a generic linearly changing source, each reaction half step uses its exact
mean rate. The total injected count remains the exact trapezoid, and the
time-dependent solution converges at second order. The native-cumulative
workflow already has a constant source rate within each matter substep.
Midpoint expansion factors and photon streaming are unchanged in this step.

### Independent verification

A homogeneous test at redshift7, physical18eV opacity and0.4 emitted photons
per atom has reference ionization0.3997386834672251 and radiation inventory
0.0002613165327749 photons per atom. The old CFL0.4 result was0.3800 and0.0200.
The coupled update agrees with the independent high-precision solution to
double-precision roundoff at CFL0.4,0.2 and0.1. For linear sources, halving the
timestep reduces the error by approximately four.

Seventeen new tests check weak and stiff absorption, front completion,
anisotropic incoming radiation, varying emission, a1e-30 source, and transparent
or fully ionized limits. The full causal RT suite passes199 tests after this
change. Numerical refinement of the native fields is reported separately;
homogeneous agreement alone does not establish spatial convergence.

### Native-history result

Four complete local seed222 histories reuse the exact earlier source and gas
archives. All audits pass, with maximum photon residual1.08e-14. The change
in the first power shell under CFL0.4 to0.2 is1.107% (previously1.084%);
CFL0.2 to0.1 gives0.408% (previously0.418%). Spatial8^3 to16^3 refinement
gives1.686% (previously1.706%). Thus this fixes an identified local integration
error but does not resolve the native-field convergence problem. Atz7 the
mass-neutral fraction changes0.37016808 to0.37000505, with the source and atom
inventories unchanged.

Measured RT times are24.65,33.96,56.24s for the three native-grid CFL cases,
and187.13s for16^3/CFL0.4. These are local case timings under shared load,
not controlled speedup measurements or Gadi estimates. Exact benchmark and
native refinement results are in `source_absorption_validation.json`.
