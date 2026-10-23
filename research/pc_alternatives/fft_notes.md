# Positive-kernel FFT source–capacity allocation

Status: implemented and falsified as a physically sufficient transport solver.
It is a useful nonnegative allocation construction, not a production replacement
for `global-budget` and not a radiative-transfer solution. Production code is
unchanged. This candidate was developed directly from the allocation equations
below; the numerical evidence is reproducible in `fft_allocation.py`,
`test_fft_allocation.py`, and `fft_results.json`.

## Declared source closure

Inputs are an explicit additive photon-count field `source` and available
atom-count field `capacity` on a periodic grid. They may represent one time-step
increment, or a static cumulative budget with consistently defined sinks; the
implementation does not supply time evolution, recombinations, redshifting,
helium, or a finite light speed. Remaining photons are retained, not counted as
absorbed or silently redistributed outside the allowed radius.

Using this field with Qin20 would introduce a declared spatial source closure.
Native radius-conditioned conditional collapsed-fraction integrals cannot in
general all be recovered by smoothing one nonnegative cell-source field. See
`native_source_notes.md` and `native_source_audit.py`; no claim of exact
preservation of those radius-dependent statistics is made here.

## Construction and proof

Let `p_i >= 0` be the remaining photons at source cell i, `c_j >= 0` the
remaining capacity of target cell j, and K a nonnegative symmetric periodic
convolution kernel. A normalized real-space ball or shell is allowed; the
normalization cancels from the allocation.

For one simultaneous round define

\[
D_i=(Kc)_i,\quad u_i=\begin{cases}p_i/D_i&D_i>0\\0&D_i=0,\end{cases}
\quad O_j=c_j(Ku)_j,\quad a_j=\min(1,c_j/O_j),
\]

with `a_j=1` if `O_j=0`. There is an implicit nonnegative edge flow

\[
F_{ij}=K_{ij}u_i c_j a_j.
\]

Its column sum is `min(O_j,c_j)`, so no target is overfilled. Its row sum is

\[
d_i=u_i[K(ca)]_i\le u_i(Kc)_i\le p_i.
\]

The same edge flow therefore establishes both source debits and target receipts;
their sums are equal. No source-by-target matrix is stored. Subtract receipts
and debits, repeat on the residuals, then increase the support radius.

For non-symmetric directional kernels, the denominator and debit must use the
adjoint kernel. Reusing the forward directional kernel would invalidate the
proof. The current scalar implementation uses symmetric kernels only.

The construction proves a feasible nonnegative source–target allocation. It
does not prove angular emission quotas, optical-depth ordering, shadowing,
uniqueness, or convergence to a physical radiation field. Local-first absorption
is an additional unresolved-cell closure: it consumes `min(p_i,c_i)` before
nonlocal allocation.

## Numerical qualifications

Each round uses three FFT convolutions: capacity normalization, target offers,
and accepted-flow backprojection. Negative FFT roundoff is clipped. To avoid
dividing by spurious positive FFT leakage into truly unsupported cells, a
denominator floor `128*eps64*max(capacity)` is used. This explicitly sacrifices
extreme dynamic-range interactions: in the saved fixture, a local capacity of
1 coexists with an unreachable capacity of `1e15`, and 0.5 photons are retained
instead of incorrectly amplifying numerical leakage. They are not lost from
the ledger. A sparse direct convolution or compensated/local treatment would
be needed in that regime.

Source debits and target capacities are clipped by at most rounding error in
ordinary tests; the global discrepancy is reported rather than repaired by a
global efficiency factor. At 128³, the preliminary random fixture has total
source `7.129399298e5` and ledger discrepancy `-3.96e-9`, about `5.6e-15`
relative. This is numerical accounting, not an exact floating-point identity.

## Reproducible probes

Seven unit tests pass, including zero/weak sources, saturated total capacity,
shared nonuniform capacities, source enumeration, translation and cubic-axis
rotation, plus tests that deliberately keep the physical failures visible.
Arbitrary continuous rotations are limited by grid discretization and were not
claimed to be exact symmetries.

| Probe | Result | Consequence |
| --- | --- | --- |
| Uniform isolated source, lattice radius 5, 515 photons; radii 1,2,4,8 | 222.058 photons (43.1%) absorbed outside the ideal sphere | Coarse scales distribute to outer neutral cells before inner cells saturate |
| Same source; integer radius steps | Outside count about `5.1e-12` | Fine radius progression fixes this special isotropic lattice fixture |
| Same source; radius ratio 1.1 | Outside count about `2.9e-11`, 31 rounds / 204 FFTs | A native-like radius ladder resolves this fixture at higher cost |
| Two collinear target cells, each capacity 100, source budget 1, one broad radius | Foreground 0.5, downstream 0.5 | Sends photons behind an unsaturated foreground absorber |
| Equal-radius equal-size opposite targets with capacities 100 and 1 | Absorbed 0.990099 and 0.00990099 | Routes photons by capacity; does not preserve isotropic equal-angle photon quotas |
| Target outside all allowed radii | Absorbed 0; source remainder 2 | Finite allocation support works |
| Two overlapping source supports, equal budgets 515+515 | Absorbed 1030 to roundoff; no overfill | Shared-capacity accounting works, without establishing correct overlap geometry |
| Unequal overlapping sources, budgets 515+128.75 | Absorbed 643.75 to roundoff | Same accounting guarantee with unequal sources |
| Sharp-k Fourier cutoff transformed to real space | Minimum `-7.47e-4`; negative weight sum 3.099 | Native sharp-k kernel cannot directly be an allocation probability kernel |

The collinear fixture isolates the first-absorber requirement. It is not a
full finite-solid-angle ray integration. The opposite-target fixture concerns
relative equal-angle quotas when both targets remain opaque; the solver also
wrongly spends photons from empty directions on those targets. Neither fixture
is a calibrated physical benchmark, but both disprove properties required of
the proposed matching operator. Thin radial shells address the first type of
coarse gap only in restricted geometry. They do not restore angular quotas.

Monotonicity is not guaranteed by the row/column proof. With the default two
rounds per radius, a fixed-seed 5³ example adds 15.100178 photons at `(0,1,0)`
yet reduces absorption at `(1,1,4)` from `3.9714e-5` to `1.5389e-5`.
Increasing to four rounds removes this particular counterexample; that is
not a general convergence or monotonicity proof. A second fixed-seed example
scales every capacity by 1.076862 while holding sources fixed; one cell's
ionized fraction increases from `3.5728e-5` to `4.7354e-4`. These failures are
small in absolute ionization, but rule out an unconditional monotonicity claim.
Both fixtures regenerate from seeds in `monotonicity_probes`, with no external
binary fixture dependency.

## Compute and memory

The callable interface for independent uncontended timing is:

```python
from fft_allocation import benchmark_adapter, es_filter_baseline
result = benchmark_adapter(source, capacity, radii, workers=4, rounds_per_scale=2)
```

The adapter returns `absorbed`, `remaining_source`, `source_debit`, `available`,
and `audit`. Its timer includes validation, initial copies, local absorption,
kernel construction, FFTs, and final bookkeeping. It does not retain a
cross-call kernel cache. Counts are six transforms per round plus one kernel
transform per used radius. Memory is O(N), roughly 18–24 float64 grid equivalents
for this unoptimized NumPy implementation, plus FFT backend workspace and
interpreter/library memory: approximately 288–384 MiB at 128³. This is an
array-liveness estimate, not measured peak RSS or a hard bound.

The preliminary benchmark uses random 2%-active additive sources, lognormal
capacity, total photons/capacity 0.3, radii 1,2,4,8,16, and four FFT workers.
Allocation consumes the source after eight rounds (52 actual FFTs). The
comparison filter smooths the same two fields at the same radii, reusing their
forward transforms: 17 FFTs, including five kernel transforms. It excludes
native source integrals, thermal histories and full 21cmFAST overhead.

| Grid | Allocation seconds, 3-run range | Filter seconds, 3-run range | Median paired ratio |
| --- | --- | --- | --- |
| 32³ | 0.040–0.101 | 0.010–0.063 | 2.62 |
| 64³ | 0.350–0.742 | 0.068–0.130 | 5.25 |
| 128³ | 2.835–3.956 | 0.629–1.130 | 3.50 |

These timings were collected while other research jobs shared the host and are
not final speed claims. Their original candidate timer also excluded input
validation, copies and local absorption; the finalized adapter includes them.
The root agent's serialized benchmark is authoritative. `fft_results.json`
preserves the original raw timing entries with this qualification.

### Fixed angular channels

Keeping K independent source photon reservoirs `p_i,m` can enforce coarse
emission cone quotas if each directional kernel has the correct support and
adjoint. The target acceptance is shared across all directions. Optimally:

1. One shared forward FFT of capacity, K inverse FFTs for denominators.
2. K forward and K inverse FFTs for offers, summed over direction.
3. One shared forward FFT of accepted capacity, K inverse FFTs for source debits.

That is `4K+2` transforms per round, plus K kernel transforms per used radius.
For K=48 it is 194 versus 6 transforms, roughly 32 times the scalar Fourier
work. Batching improves call overhead and parallel scheduling, not this
asymptotic work. Retaining photon reservoirs, source coefficients, and kernel
spectra costs about three K grid equivalents: about 2.25 GiB at 128³,K=48,
before other fields and FFT workspace. Streaming/regenerating coefficients or
kernels can trade that memory for more transforms.

Even this angular extension lacks intervening opacity. Conical shells can
exchange budget across rays within a cone and skip foreground cells across
radial increments. Enforcing first absorption requires a directional sweep,
line attenuation, or a local transport operator; adding channels alone does
not make FFT matching physically correct. The angular-transport candidate is
therefore the more appropriate place to implement that requirement.

## Recommendation

Retain this method as an explicit feasible-allocation control or a possible
proposal/preconditioner. Do not freeze its absorbed field as physically correct
bulk ionization. It proves that conservation plus positive multiscale kernels
does not select the physical solution, even when an isolated-source sphere is
reproduced. It is not recommended as the next production photon-conservation
solver on the evidence above.

Reproduce with the research Python environment:

```bash
python -m unittest -v test_fft_allocation.py
python fft_allocation.py --workers 4 --output fft_results.json
```
