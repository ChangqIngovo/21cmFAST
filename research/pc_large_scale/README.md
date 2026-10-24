# Large-volume 21cmFAST PC / ES comparison

This experiment compares the latest adaptive eight-subcell model with a
source-matched sharp-k excursion-set control, using the statistical tests of
[Choudhury & Paranjape (2018)](https://arxiv.org/abs/1807.00836), especially
Figs. 1 and 5–7. It generates real native 21cmFAST density and emissivity fields,
then solves the static ionization problem at fixed redshift over an efficiency
sweep. It does **not** run SCRIPT or reproduce its N-body/source realization.

## Default experiment

| Setting | Value |
|---|---|
| Periodic box | 500 h^-1 cMpc = 737.46 cMpc |
| Redshift | 8 |
| Initial-condition grid | 800^3, native 2LPT, approximately 0.922 cMpc cells |
| Native source / ionization grid | 100^3 |
| Ionization comparison grids | 100^3, 50^3, 25^3: cells 5, 10, 20 h^-1 cMpc |
| Seeds | 1234, 2345, 3456, 4567 |
| Input photon doses and matched mass fractions | 0.1 to 0.9 in steps of 0.1 |
| Cosmology | h=0.678, Om=0.308, Ob=0.0482, ns=0.961, sigma8=0.829 |
| Reference scale | 0.04 <= k < 0.06 h cMpc^-1, near paper k0=0.05 |
| Angular predictor | min 24, max 12288, surface sampling 4 |
| Maximum transport distance | L/3 = 166.67 h^-1 cMpc; separate 2x-distance checks |
| Extra tests | First seed, finest grid, Q=0.5 and 0.8: 2x angular sampling and 2x distance |
| PBS resources | hugemem, 32 CPUs, 320 GB, 24 h limit, 16 GB jobfs |

The 112 independent cases share no live gas/front state. Initial fields are
generated sequentially per seed with all 32 OpenMP threads. Thereafter up to
32 separate case processes execute concurrently, each with a serial transport
kernel and one FFT thread. This is case parallelism, not parallel ray updates.
The 320 GB request allows the 800^3 native construction and concurrent transport
workers; 24 h is a scheduling limit, not a measured runtime estimate.

The native source prescription retains the previously tested restricted Qin20
ACG astrophysical parameters with `SOURCE_MODEL=L-INTEGRAL`: no MCGs, X-ray/spin
evolution, recombinations or photoheating feedback. The paper instead uses
N-body density and mass-proportional sources above a hard minimum halo mass.
Thus compare resolution trends and conservation effects, not absolute bias
values as an exact reproduction. A coeval efficiency sweep is not a coupled
reionization history.

## Controlled comparisons

For each seed generate the finest source and capacity arrays once. A coarse
cell contains the **sum** of its child counts. Both total photons and atoms
therefore remain unchanged. The source field already contains the native
Lagrangian-to-Eulerian mass weighting; no extra density factor is applied.

Each case saves four independently defined fields:

* `pc_same_budget` and `es_same_budget`: identical emitted photons and gas.
* `pc` and `es_matched_mass`: independently adjust a scalar source amplitude to
  reach the requested ionized **mass** fraction, with absolute tolerance 2e-4.
  The ES field is then matched to the actual PC mass fraction. The original
  fixed-budget fields are preserved. No ionization fields are interpolated.

The amplitude bracket/history is saved. An unreachable target or an ES barrier
discontinuity is marked unmatched and excluded from matched-resolution ratios.
The extra PC amplitude corrects only the comparison stage, and does not erase
retained photons from the ledger. Significant retention (>1e-4 of emitted
photons by default) is separately flagged. Both PC maps have independent
array-level global, per-source and local-capacity audits.

The sharp-k control uses `k R * 0.413566994 <= 1`, centre-cell crossings, and a
1.1 geometric filter ladder as in native 21cmFAST. It is a double-precision
source-matched implementation, labelled separately from the native ES box.
The native ES result at the original emissivity is also saved, with its
difference from this comparator. Source scaling after field extraction avoids
changing source bias when matching the mean ionization fraction.

## Statistics and acceptance

With rho = capacity / mean(capacity), use

    Q_mass = sum(absorbed) / sum(capacity)
    delta_HI = (capacity - absorbed) / mean(capacity - absorbed) - 1
    b_HI^2(k) = P_delta_HI(k) / P_delta_matter(k)

This is the paper's fractional HI density statistic, not the power of xHI.
Spectra use P = V |FFT|^2 / Ncell^2 and full-FFT mode multiplicities, including
the odd-grid case. The k0 statistic is a ratio of band-averaged powers, not an
average of mode ratios. No RSD or grid-window deconvolution is applied.
The 21-cm conversion uses the paper convention 22 sqrt((1+z)/7) mK and saturated
spin temperature; both fractional and dimensional spectra are retained.

Paired-seed fine/coarse ratios are saved, with their sample standard deviation.
The source and matter input powers in the same k0 band are also saved at every
resolution, so input smoothing or aliasing can be distinguished from changes
caused by the ionization algorithm.
The requested convergence criterion is 5% in b_HI^2 for angle, distance and
spatial-resolution tests. These are measured outcomes: a non-converged case
does not abort collection or get labelled converged merely because its photon
ledger closes. Near a bias minimum, inspect the absolute spectra and fields as
well as ratios. Four seeds quantify variation but do not establish percent-level
ensemble precision by themselves.

## Gadi

From the existing fork checkout:

```bash
(
  set -euo pipefail
  cd /g/data/qp00/sz4640/21cmFAST
  git fetch https://github.com/ChangqIngovo/21cmFAST.git Photon_conserving
  git switch Photon_conserving
  git merge --ff-only FETCH_HEAD
  bash research/pc_large_scale/gadi_prepare.sh --submit
)
```

The shared preparation pins a detached worktree, builds with Python >=3.12,
copies and hashes the experiment configuration, and records packages, binary,
commit and PBS script. `PC_CONFIG=/absolute/path/to/config.json` selects an
explicit alternative before preparation; resource overrides use `PC_THREADS`,
`PC_MEM`, `PC_WALLTIME`, `PC_QUEUE`. Compute nodes never install packages.

The job runs all existing transport/adapter tests plus the new analysis tests
before generating the large fields. Results live under the printed run directory:

* `run_<jobid>.log`, `run_manifest.json`, `experiment_config.json`;
* `results/summary.json`, `summary.csv`: ledgers, mass matches, paired resolution
  ratios, angle and distance checks;
* `results/comparison.{png,pdf}`, `bias_spectra.{png,pdf}`, `morphology.{png,pdf}`;
* `results/fixtures/seed*/n*/`: source, capacity and ES threshold NPY arrays;
* `results/fixtures/seed*/metadata.json`: native inputs, original native ES check;
* `results/cases/*/fields.npz` and `result.json`: all four absorption fields,
  source photon ledgers, complete shell spectra and timing;
* `results/logs/`: per-seed and per-case logs.

Each completed fixture/case has file checksums and an atomic receipt. A failed
job can be resubmitted with `qsub -v RUN_DIR=/same/run /same/run/submit.pbs`;
unchanged completed tasks are verified and skipped. A PBS-killed job may leave
`.running`; confirm the job has ended with qstat before removing that lock.
`COMPLETED` means all calculations finished; scientific convergence flags must
still be read from `summary.json`.

Local unit and small-native end-to-end validation is recorded in `validation.json`.

## Late-stage follow-up and common-mesh diagnosis

The large run's native fine/coarse power ratio includes both output projection
and changes to the solved ionization field. `diagnose_coarsening.py` sums the
already solved fine absorbed counts onto the coarse mesh and compares them to
the direct coarse solve. It records the exact decomposition

    b2_fine / b2_coarse = (b2_fine / b2_projected) * (b2_projected / b2_coarse)

The first factor measures output projection, including aliasing and the matter
denominator. The second compares both solutions on identical cells and
capacities; it still combines input coarsening and transport discretization.
It also measures the common-mesh cross-correlation and count-field L1 error.
It does not alter or replace the original resolution comparison. Use existing
data without another transport solve:

```bash
python research/pc_large_scale/diagnose_coarsening.py \
    --root /path/to/large_run/results --output /path/to/parent_coarsening.json
```

The late follow-up reuses the completed parent's immutable native source and
capacity files and its Python environment, with new code archived at the
selected commit. It runs 120 cases: four seeds, three grids, photon doses 0.8
and 0.9, path-cutoff factors 1/2/4/8, and doubled angular sampling at factor 8.
Every path-length
comparison uses the same per-source angular bank, checked explicitly. The
source amplitude is not calibrated to compensate for retained photons.
Therefore inspect the reported change in actual Q with each power change.

```bash
bash research/pc_large_scale/gadi_late_prepare.sh /path/to/large_run --submit
```

Omitting the parent path selects the most recently completed `large_pc_*` run
under `/g/data/qp00/sz4640/21cmfast_runs`; the selected path is printed and
recorded. Defaults are 32 independent workers, 320 GB, 6 h scheduling limit. The
job also calculates the common-mesh diagnosis for the original comparison.
Read `results/summary.json` for individual and combined retention, field-L1,
and bias checks, and `results/parent_coarsening.json` for the decomposition.
Scientific nonconvergence is recorded as a result rather than an execution
failure. Failed tasks are resumable after input/source hash verification.

Local follow-up findings and their scope are in `late_research.md`.
