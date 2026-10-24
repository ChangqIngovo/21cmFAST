# Reproducing the PC refinement experiments

These are research benchmarks of the conservative opaque-voxel event solver.
Agreement with a high-angle version of that solver measures angular error; it
does **not** establish agreement with radiative transfer or remove the continuum
front error. `front` and the isolated-sphere cases separately report continuum
comparisons. Native benchmarks freeze the 21cmFAST source/capacity fields and
are not a new coupled cosmological lightcone simulation.

No raw native or reference NPZ cubes are committed. `reproduce.py` accepts their
locations explicitly or generates fixtures from checked-in scripts. Use an
isolated checkout of the desired commit and an environment containing the
matching installed 21cmFAST package, NumPy, SciPy, pytest and a C++ compiler.
Native preparation additionally requires the normal 21cmFAST build dependencies.
Run from any working directory; every script location is resolved relative to
the checkout.

## Quick checks and existing native fixtures

```bash
PC_REPO=/path/to/21cmFAST
PC_PYTHON=/path/to/environment/bin/python
PC_RESULTS=/path/to/new/results_directory

"$PC_PYTHON" "$PC_REPO/research/pc_refinement/reproduce.py" \
  --benchmarks tests audit theory speed front \
  --native32 /path/to/native32/source_capacity.npz \
  --native64 /path/to/native64/native_fields.npz \
  --threads 4 --repeats 3 --output-dir "$PC_RESULTS"
```

Omit either native argument to run only the remaining available cases. The output
directory must not already exist. Add `--dry-run` to inspect commands without
creating outputs or running them. Benchmark selection does not launch unrelated
experiments. `--threads` controls native preparation/OpenMP where implemented;
the event transport comparisons remain serial. The timing comparisons alternate
candidates where supported; compilation is excluded by the benchmark itself.

`reproduction.json` records the commit, dirty working-tree state, source hashes,
exact subprocess arguments, exit codes, durations, and fixture hashes. Keep
`preparation.json` beside a supplied native32 fixture and `native_comparison.json`
beside native64: their full contents are embedded in the reproduction manifest,
retaining snapshot, random seed and physics settings. If they are absent the
manifest explicitly marks native provenance as incomplete. Keep the NPZ inputs
and metadata together for future reproduction; a hash alone cannot recreate them.
Each stage has its own log and JSON result. `COMPLETED` appears only after all
selected stages succeed; failed runs retain `FAILED` and the completed stages.

## Generate both native fixtures and the angular references

```bash
"$PC_PYTHON" "$PC_REPO/research/pc_refinement/reproduce.py" \
  --benchmarks prepare-native32 prepare-native64 references tests audit theory speed budgeted \
  --threads 4 --repeats 3 --output-dir "$PC_RESULTS"
```

The two fixtures deliberately have different scopes and must not be presented
as resolution convergence of one identical physical model:

| Fixture | Checked-in generator | Settings retained |
|---|---|---|
| native32 | `research/pc_alternatives/native_benchmark.py prepare` | Qin20 E-INTEGRAL; 32 cells/axis, 200 cMpc, seed 1234, redshift nodes 35, 12, 8, 7.5; source/capacity from the final node; density and X-ray preionization included |
| native64 | `research/pc_reconsidered/native_driver.py` | Restricted L-INTEGRAL ACG experiment; 64 cells/axis, 200 cMpc, seed 1234, z=8; spin-temperature, recombination and feedback modules disabled |

Native32 preparation retains its native outputs and `preparation.json`; native64
retains `native_fields.npz` and `native_comparison.json`. The snapshot labels in
those manifests, rather than a filename alone, identify the fields.

`references` creates `comparison_fields/comparison_<case>.npz` and separate toy
and native generation reports with input hashes, angle counts and the change
under angular refinement. Sparse toy final references use 262144 angles; dense
toy references use 12288; native32 references use 3072. These can take much
longer and more memory than the candidate methods. The solver maximum travel
distance in native comparisons is 16 grid cells, fixed for these fixtures.

To reuse those references in another fresh output directory:

```bash
"$PC_PYTHON" "$PC_REPO/research/pc_refinement/reproduce.py" \
  --benchmarks budgeted geometry \
  --reference-dir /path/to/previous/results_directory/comparison_fields \
  --native64 /path/to/native64/native_fields.npz \
  --repeats 3 --output-dir /path/to/new/recheck
```

`budgeted` requires the four named toy cases and native-dose-0.2,
native-dose-0.6, native-frozen references; `geometry` requires all standard toy
cases. Missing inputs fail explicitly. Do not mix references from different
source, capacity or initial-ionization fields. Toy inputs are checked for exact
equality; native inputs are taken from the stored reference fixture and hashed.

## Incremental-front and sampling checks

```bash
"$PC_PYTHON" "$PC_REPO/research/pc_refinement/reproduce.py" \
  --benchmarks sampling incremental convergence faint density \
  --native32 /path/to/native32/source_capacity.npz \
  --reference-angles 3072 --output-dir /path/to/new/increment_check
```

This generates the reproducible initial/incremental fields before computing the
high-angle reference. To reuse them, omit `incremental convergence` and provide
`--increment-fields /path/to/incremental_fields.npz` and
`--increment-reference /path/to/incremental_reference.npz`. `convergence` reports
how its new angle count compares with the initial stored reference. A fixed
reference agreement should not be mistaken for a demonstrated continuum limit.

The delivered compact incremental profile has a separate repeated validation:

```bash
"$PC_PYTHON" "$PC_REPO/research/pc_refinement/reproduce.py" \
  --benchmarks incremental validated-incremental native-integration \
  --native32 /path/to/native32/source_capacity.npz \
  --threads 4 --repeats 2 --output-dir /path/to/new/final_validation
```

`validated-incremental` generates both K1536 and K3072 references, embeds hashes
of the exact source/capacity/initial state in each reference, and rejects any
input, angular-count or horizon mismatch before comparison. It then alternates
original and compact incremental solves on the identical initial state. To reuse
the initial state, omit `incremental` and pass `--increment-fields`. The separate
`native-integration` stage exercises the balanced profile through the restricted
native 21cmFAST interface and checks K1536/K6144 references.

`quadrature` also accepts native fixtures. Its standalone benchmark reuses the
original `research/pc_reconsidered/comparison_fields` cache only when available
and consistent with the checked-in generation reports, otherwise regenerating
the required reference. The runner's `--reference-dir` applies to `references`,
`budgeted` and `geometry`, not that historical quadrature cache. There is no need
to rerun this expensive negative experiment to exercise the budgeted candidate.

## Gadi PBS example

The example assumes the matching package is already installed in `PC_PYTHON`.
Save it as `reproduce_pc.pbs`; choose a new results path for every submission.
The resource request is an initial allocation, not a measured guarantee that
every optional high-angle reference will fit or finish in four hours.

```bash
#!/usr/bin/env bash
#PBS -N pc_refinement
#PBS -P qp00
#PBS -q normal
#PBS -l ncpus=4
#PBS -l mem=32GB
#PBS -l walltime=04:00:00
#PBS -l jobfs=4GB
#PBS -l storage=gdata/qp00+scratch/qp00
#PBS -l wd
#PBS -j oe
set -euo pipefail
: "${PC_REPO:?Pass the isolated 21cmFAST checkout}"
: "${PC_PYTHON:?Pass the installed environment's Python}"
: "${PC_RESULTS:?Pass a new result directory}"
module --force purge
module load gcc/12.2.0 fftw3/3.3.10-nci1 gsl/2.7.1 python3/3.11.7
export CXX=g++
"$PC_PYTHON" "$PC_REPO/research/pc_refinement/reproduce.py" \
  --benchmarks prepare-native32 prepare-native64 references tests audit theory speed budgeted \
  --threads "$PBS_NCPUS" --repeats 3 --output-dir "$PC_RESULTS"
```

Submit with paths appropriate to the account:

```bash
qsub -v PC_REPO=/g/data/qp00/sz4640/21cmFAST,PC_PYTHON=/path/to/venv/bin/python,PC_RESULTS=/scratch/qp00/sz4640/pc_refinement_new reproduce_pc.pbs
```

The driver itself never submits a job. `.gitignore` excludes generated NPZ
cubes, shared libraries, bytecode and the optional local `reproduction_runs/`
directory. Keep small selected JSON reports in git only when they are intended
research records; arbitrary output directories outside this tree remain the
caller's responsibility.
