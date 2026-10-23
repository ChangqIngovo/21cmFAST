# Elliptic obstacle control

This candidate is retained as a mathematical control, not selected for UV fronts.
Let `L` be the periodic positive-conductance nearest-neighbour negative Laplacian,
`s >= 0` newly supplied photon counts, and `c >= 0` available atom counts. For
`sum(s) < sum(c)` solve

\[
\min_{u\geq0}\;\tfrac12 u^T L u+(c-s)^T u.
\]

The KKT conditions are `g = Lu+c-s >= 0`, `u >= 0`, `u*g = 0`.
Define absorption `a = s-Lu`. At `u_i>0`, `a_i=c_i`; at `u_i=0`,
`(Lu)_i <= 0`, hence `a_i>=s_i>=0`, while `g_i>=0` gives `a_i<=c_i`.
Periodic telescoping gives `sum(a)=sum(s)`. Antisymmetric edge fluxes account
for local transfers. This is stronger than a global efficiency correction.
The proof requires the M-matrix sign structure, not an arbitrary spectral filter.

For `sum(s)>sum(c)` the objective is unbounded along positive constants. The
implementation handles this separately as saturation and an explicit reservoir.
For equality, the potential is nonunique up to an additive constant. Disconnected
graphs require the budget condition separately on each connected component.

`obstacle.py` solves this model by accelerated projected gradient and reports
feasibility/KKT residuals without clipping the resulting absorption. Its toy
single-source sphere is well behaved; the 32^3 R=5 test converged in 140 iterations
with count residual 2.7e-13 and capacity violation 1.6e-10.

The physical closure is diffusion. It does not keep source directions. On an
already transparent union of two R=8 spheres separated by12 cells, a small
increment from the first centre allocates only about2.79% into the outer region
of the `cos(theta)>=0.75` cone, versus12.5% of continuum isotropic source
directions. This voxel measurement also has spatial discretization error, but
the discrepancy is not a ledger error. `obstacle_results.json` stores it.

The high-capacity planar-wall probe receives42.9%, so it *passes* the50% upper
bound in this particular setup. That passing test must not be turned into a
general shadow/transport guarantee. The overlap counterexample is sufficient
to reject the diffusion closure for the requested directional physics.

All elapsed times are toy, preliminary and shared-host measurements. No native
21cmFAST mode was changed or benchmarked by this file.
