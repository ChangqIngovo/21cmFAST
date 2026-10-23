# Qin20 PC / ES comparison on Gadi

The runner uses `InputParameters.from_template("Qin20")` for both models. Their only changed physical option is `PHOTON_CONS_TYPE`: `global-budget` versus `no-photoncons`. Minihalos, X-rays, spin-temperature fluctuations, inhomogeneous recombinations, sharp-k ionization filtering and photoheating feedback remain enabled as specified by Qin20. Both evolutions use one computed initial-condition realization. Their source and feedback histories subsequently evolve independently.

The default validation setup is **200 cMpc, 64³ ionization cells, 192³ initial-condition cells, seed 1234, 32 OpenMP threads**, evolving from the template's default starting nodes to z=5.5. These are declared test-volume and resolution choices, not the paper's original numerical setup. The 200 cMpc box preserves Qin20's `R_BUBBLE_MAX=50` within the upstream constraint `R_BUBBLE_MAX <= BOX_LEN/3`.

The PBS request is `normal`, 32 CPUs, 128 GB, 24 hours. No MPI launcher is needed: the native C calculation uses OpenMP. Python BLAS uses one thread to avoid oversubscription.

## Prepare and submit

In a checkout of the user's `Photon_conserving` branch on Gadi:

```bash
cd /g/data/qp00/sz4640/21cmFAST
git fetch origin Photon_conserving
git switch Photon_conserving
git merge --ff-only origin/Photon_conserving
bash tools/gadi_qin20_prepare.sh --submit
```

For the initial checkout:

```bash
cd /g/data/qp00/sz4640
git clone --branch Photon_conserving --single-branch \
  https://github.com/ChangqIngovo/21cmFAST.git
cd 21cmFAST
bash tools/gadi_qin20_prepare.sh --submit
```

The preparation script creates a fresh detached worktree pinned to the selected commit and an isolated virtual environment. It first fetches the official `v4.3.1` release tag if missing, checks its exact commit (`72d8a2d23702f1b69fbc85abd25d4592a7bcd3f5`) and verifies that the selected branch descends from it. This prevents an untagged checkout being assigned the invalid-for-caching `0.0.1.dev...` package version; it does not change the simulation's physics. It builds on the login node, where dependencies can be downloaded; the PBS job does not install packages or require external network access. The upstream branch requires **Python >=3.12 and NumPy >=2.0**, so the older `python3/3.11.7` / `tauefix` environment is not reused. The script selects a suitable Python module from the actual module inventory, or accepts `PC_PYTHON_MODULE` / `PC_PYTHON` explicitly.

To inspect inputs before submitting, omit `--submit`; the script prints the exact `qsub -v RUN_DIR=...` command. Each prepared run is launched once. A failed run should use a fresh preparation directory, leaving the previous logs intact.

Optional numerical/resource overrides:

```bash
PC_GRID=128 PC_THREADS=32 PC_MEM=320GB PC_WALLTIME=48:00:00 \
  bash tools/gadi_qin20_prepare.sh
```

The above only prepares the run. For 320 GB on 32 CPUs, select `hugemem` when submitting:

```bash
qsub -q hugemem -l ncpus=32,mem=320GB,walltime=48:00:00 \
  -v RUN_DIR=/path/printed/by/preparation /path/printed/by/preparation/submit.pbs
```

`PC_REV`, `PC_RUN_ROOT`, `PC_PROJECT`, `PC_SEED`, `PC_BOX_MPC` and `PC_ZMIN` can also be supplied. The default storage directives are for project qp00; a different project requires matching PBS storage directives.

## Outputs

The preparation command prints the unique run directory:

```text
/g/data/qp00/sz4640/21cmfast_runs/Qin20_<commit>_<time>_<suffix>/
```

- `run_<PBS_JOBID>.log`: runtime progress and errors.
- `COMPLETED` / `FAILED`: whole-job completion or exit status.
- `results/Qin20_ES.toml`, `Qin20_PC.toml`: complete resolved inputs.
- `results/run_config.json`: versions, commit, numerical settings and node redshifts.
- `results/initial_conditions_sha256.json`: checksums of the shared initial arrays.
- `results/ES/history.csv`, `PC/history.csv`: volume/mass-weighted neutral fractions, native brightness temperature, unconditional source budget, cumulative recombinations and PC diagnostics at every evolved node.
- `results/PC/validation_summary.json`: status counts, unresolved boundary-node count and maximum independently reconstructed budget residual.
- `results/history_comparison.csv`: same-redshift PC/ES history comparison.
- `results/{ES,PC}/fields_z*.npz`: selected snapshots, including density, neutral fraction, native brightness temperature, X-ray ionization, recombinations and ionization rate.
- `results/comparison_power_z*.csv`: shell spectra and PC/ES ratios at the same redshift. Snapshot nodes are the nearest native nodes to z=15,12,10,8,7,6,5.5.
- `results/cache/`: native caches, with PC and ES separated by the input-parameter identity.

The 21-cm spectra use the native thermal brightness-temperature field, not the saturated-spin-temperature approximation. Power spectra are mean-subtracted, use comoving Mpc units, count full-FFT modes correctly and stop at the one-axis Nyquist frequency. Ratios at fixed redshift combine history and morphology changes; they are not fixed-neutral-fraction morphology comparisons.

The count diagnostics include both the native PC ledger and an independent reconstruction from cell fields. Every PC node must pass finite-value, audit-agreement, no-overspending and budget-closure checks before successful completion is recorded. `pc_budget_residual` and `grid_uv_closed_residual` should be interpreted together with the documented count tolerance and the explicit photon reservoir. Numerical count closure is not a spatial or timestep convergence test.

PC status codes: 0 is the uncorrected ES path; 1 is a calibrated state; 2 retains photons after reaching full capacity; 3 retains an under-budget remainder when a discrete sink/boundary jump cannot be resolved within tolerance. Status 3 is exposed in the history and is not labelled a fully converged boundary solution.
