# Native ES global UV-budget mode

Select `PHOTON_CONS_TYPE="global-budget"`. This mode operates inside
`ComputeIonizedBox`; its output enters the ordinary 21cmFAST temperature,
recombination, source-feedback and next-redshift calculations. It is an
experimental global count closure for the native effective UV model, not a
source-by-source radiative-transfer solver.

## Count convention

For the supported Eulerian integral source model, the native unconditional
ACG and MCG histories define the cumulative UV source count

\[
Q=N_{\rm cell}\left(n_{\rm ion,ACG}^{\rm uncond}
                  +n_{\rm ion,MCG}^{\rm uncond}\right).
\]

Counts are in units of the mean baryon content of one simulation cell. The
calibrated field obeys

\[
Q=\sum_j(1+\delta_j)\max(1-x_{{\rm HI},j}-x_{X,j},0)
  +\sum_jN_{{\rm rec},j}^{\rm current}+Q_{\rm remaining}.
\]

Here `x_X` is the current native spin-temperature calculation's electron
fraction, including its residual-ionization baseline. Following the count
convention assumed by the native ES barrier, this ledger treats the stored
recombination field as per mean baryon: it does **not** multiply it by density
again. The homogeneous option instead multiplies its one stored
recombination value by the number of cells.

This is a defined effective UV ledger. It excludes the X-ray baseline from UV
ionizations and charges the current native recombination prescription. It does
not independently conserve helium electrons, X-ray photon number, or radiation
energy. The original full and partial ES branches account for sinks differently;
enforcing this ledger changes that accounting as well as the crossing amplitude.

## Field construction

1. Execute the native source integrals and Fourier filters once per scale.
   Cache each cell's native crossing threshold, physical ionization rate and
   local partial-ionization source value.
2. Select an amplitude `a` for the ES condition
   `a * nion_R > (1 - xe_R) * (1 + rec_previous_R)`. Assign the largest
   crossing radius, or a local partial ionization when there is no crossing.
   `a` does not change escape fractions, unconditional photon production,
   conditional source histories, or the physical source normalization of Gamma.
3. For every trial, compute the **current** cumulative recombination field from
   that trial's ionized fraction and Gamma, using the native rate table.
   Retain a cost bracket with one affordable and one unaffordable endpoint.
   The cost need not be monotonic because Gamma changes at ES crossings.
4. At a discrete crossing, interpolate the two neutral-fraction and Gamma
   fields and recompute recombinations. Do not interpolate recombination
   counts independently. This fractional boundary is an approximation to the
   morphology, not an angular transport solution.
5. Sum the actual stored float32 fields. Preserve an explicit nonnegative
   remainder and report the count residual. Previous ionization histories are
   read-only; reionization redshifts and temperatures follow the delivered map.

The new work uses OpenMP and a read-only GSL interpolation lookup. It avoids
sharing mutable GSL accelerators across its recombination workers. Cached scale
statistics cost O(number of cells times number of radii) memory. There is no
repeated FFT in the amplitude search. Existing `no-photoncons` and legacy
parameter-calibration modes remain available.

## Diagnostics and limits

`IonizedBox` includes `pc_photons`, UV-only `pc_ionized_atoms`,
`pc_recombinations`, `pc_recombination_previous`, `pc_remaining_photons`,
`pc_budget_residual`, `pc_count_tolerance`, amplitude endpoints, boundary
weight, evaluation count and status. Additional counts expose total ionization,
the raw X-ray baseline and its float32 representation error.

| Status | Interpretation |
| --- | --- |
| 0 | This mode was not used. |
| 1 | Absorbed cost matches Q within the reported count tolerance. |
| 2 | All positive-density cells are ionized at the reached amplitude; surplus remains. |
| 3 | A discrete boundary/sink or search limit leaves an explicit remainder. |

Check both the residual **and** `pc_remaining_photons / pc_photons`.
Algebraic ledger closure alone does not establish absorption completion. At
very early times, float32 neutral fractions near one cannot encode arbitrarily
small UV increments. The native recombination table also has a low-Gamma
cutoff. Neither limitation is hidden by calling retained photons absorbed.
Status 2 does not prove a maximum over all possible ionization-rate fields:
larger amplitudes can still select different radii and Gamma after the neutral
field saturates. The remaining count is a cumulative ledger remainder, not an additional photon
source to add again at the next redshift. A budget smaller than already incurred
recombinations is rejected.

Current support requires `SOURCE_MODEL="E-INTEGRAL"`, centre-cell ES and
`USE_EXP_FILTER=False`. Qin20 meets these restrictions unchanged. No calibration
run or redshift remapping is used. Count closure does not demonstrate spatial,
time-step or power-spectrum convergence, nor does it guarantee a particular
sign of PC/ES power suppression.

## Validation and Gadi

Run `python -m pytest tests/test_global_budget.py -q` for configuration contracts.
`python tools/validate_global_budget.py --output /path/to/fresh/directory`
evolves both full-Qin20 arms from copied identical initial arrays, checks prior
state immutability and independently reconstructs the ledger from output grids.
Its default 8-cubed grid is an integration test, not a science-resolution result.

See [README_gadi_qin20.md](README_gadi_qin20.md) for the larger paired run,
commit-pinned preparation, `qsub` submission and output files.
