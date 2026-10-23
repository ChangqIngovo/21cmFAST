# Validation recorded 2026-10-23

These are implementation checks on a small grid. They do not establish spatial
convergence, reproduce the paper's numerical volume/resolution, or validate a
particular change in the 21-cm power spectrum. Gadi submission was not performed
from the development environment because its SSH hostname could not be resolved.

## Full Qin20 pair

The unchanged Qin20 physical template was evolved from z=35.402256 to z=8,
72 output nodes per arm, with 8³ ionization cells, 24³ initial-condition cells,
200 cMpc box, seed 1234 and four OpenMP threads. One set of actual initial arrays
was copied into both arms; source and feedback histories evolved independently.
Only `PHOTON_CONS_TYPE` differed. Every node was checked for finite fields,
valid fractions, unchanged prior-step state and an independently reconstructed
source/sink ledger using extended-precision sums of the delivered arrays.

| Quantity at z=8 | Native ES | Global-budget PC |
| --- | ---: | ---: |
| Volume-mean neutral fraction | 0.3699899401 | 0.3745407348 |
| Mass-weighted neutral fraction | 0.3654105855 | 0.3699902447 |
| Effective UV source count | 316.984556151 | 316.984556151 |
| UV-only ionized stock | 319.190972715 | 316.846187175 |
| Current cumulative recombinations | 0.203202151 | 0.138302267 |
| Explicit remaining photons | N/A | 0.0000667090 |
| Source minus UV stock, recombinations and remainder | -2.409618714 | 4.52e-14 |

Counts use the effective convention in [README_global_budget.md](README_global_budget.md).
The PC remaining fraction at z=8 was 2.10449e-7, below its reported absorption
tolerance. Across the 72 PC nodes, 62 had status 1 and 10 early nodes had status 3.
The largest remaining fraction was **1.38223% at the first node**, where the
total effective source count was only 8.99068e-4. This is explicitly unresolved
absorption, not a claim that all those photons were absorbed. All 20 nodes at
z<=12 had status 1; their maximum remaining fraction was 9.28000e-7.

Maximum absolute independent ledger residual over the four-thread history was
4.52e-14. Density, neutral fraction, X-ray fraction, recombinations, Gamma and
brightness-temperature arrays of the ES arm at z=8 were bitwise identical to
an independent run with the unmodified official 4.3.1 package.

## Parallel and boundary checks

The active z=8 PC ionization step was repeated from identical current/prior
fields with one and four threads. Neutral fraction, Gamma, recombinations,
mean free path, reionization redshift and kinetic temperature were all bitwise
identical. The scalar sums differed only at floating summation precision.

A separate final-build test retained all Qin20 physics with deliberately coarse
nodes 35,20,12,9,8,7.5. Its final state had nonzero recombination consumption
(11.1914 in the same effective units). The complete frozen input fixture was
saved to HDF5 and replayed in a fresh process; its one/four-thread fields also
matched bitwise. The coarse chronology is a numerical execution check, not an
alternative scientific evolution.

Six dedicated tests passed, including exact-zero UV, float32-small positive UV,
and surplus after full ionization. The small positive case correctly reported
status 3 instead of claiming complete absorption. Another 240 upstream input,
template and serialization tests passed. Changed Python files passed Ruff;
PBS/preparation scripts passed Bash syntax checks.

The production comparison runner also passed a short paired test using its
shared cache and unchanged Qin20 flags. It wrote histories, selected fields and
native thermal 21-cm spectra, and enforced independently reconstructed counts
before writing completion markers. Release-tag metadata is required during
installation so upstream HDF5 cache-version checks work correctly.

## Reproduce

```bash
python -m pytest tests/test_global_budget.py -q -W error
python tools/validate_global_budget.py --threads 4 --output /path/to/fresh/validation
python tools/validate_global_budget.py \
  --frozen-step /path/to/fresh/validation/frozen_native_step \
  --output /path/to/fresh/replay
```

For the 32-core, 64³ paired Gadi run down to z=5.5, use the commit-pinned
`qsub` workflow in [README_gadi_qin20.md](README_gadi_qin20.md). Its results still
need to be obtained and assessed; the local small-grid checks are not those
larger-run results.
