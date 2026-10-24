# Adaptive eight-subcell ionization fronts

This experiment retains the existing eight-subcell gas model while allocating
resolved gas states only when they are needed. It uses original coarse-cell
source centres and an octant-balanced angular quadrature. Photon numbers, atom
capacities and the transport horizon are unchanged.

1. A parent with identical children and only its own isotropic source evolves
   as one scalar node. Its eight children have identical depletion histories.
2. The first materialized incident ray promotes that parent. Its absorption,
   remaining capacity and compensated arithmetic are divided by eight at the
   current event time, then the ray reaches its actual geometric child.
3. Children share the global event clock. Once every child clears and all its
   rays have been released, the parent becomes transparent and its child slot
   can be reused.
4. Nonuniform partial child states persist between steps. Replacing them with
   the parent mean changes the resolved front even when photon totals close.

For an untouched uniform parent, at time $t$ before its first external arrival,

$$
C_{i,\alpha}^{\rm rem}(t)=\frac{C_i^{\rm rem}(0)-Q_i t}{8},
\qquad \alpha=1,\ldots,8.
$$

The scalar and eight-node representations are therefore equivalent until the
first directional asymmetry is introduced. This equivalence relies on equal
octant budgets in the specified discrete quadrature, not just continuum isotropy.

## Usage

```python
from adaptive import solve

first = solve(source1, capacity, min_angles=1536, max_angles=1536)
second = solve(source2, capacity, initial_state=first["state"],
               min_angles=1536, max_angles=1536)
```

Inputs are emitted photons and atom counts per original coarse cell. `state`
contains cumulative coarse ionization and only the resolved partial children;
the capacity fingerprint rejects reuse with a different density field. Passing
`initial_ionized` instead assumes a uniform initial fraction inside each parent.
`return_dense=True` additionally returns fine-grid increments and ionized counts
for reference comparisons, at the cost of dense output memory. Direction counts
use the remaining-neutral-density correction by default; `effective_density="total"`
retains the historical predictor for controlled comparisons. Equal fixed counts
isolate spatial-model equivalence from this angular allocation choice.

The native restricted 21cmFAST adapter accepts `--solver adaptive-front`.
Its existing `balanced` default is unchanged. This is still the static-density,
constant-emission, infinite-light-speed, opaque-cell model; the native adapter's
no-recombination/heating/feedback restrictions remain enforced.

## Checks and reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  research/pc_adaptive_front -q --confcutdir=research/pc_adaptive_front
python research/pc_adaptive_front/benchmarks.py --help
```

The benchmark compares the original coarse solver, latest balanced coarse
profile, matched-quadrature coarse solver, full eight-subcell solver, and
adaptive solver. Fine-grid equivalence
is tested independently of timed runs; an integrated sphere provides a separate
continuum spatial reference. Native source fields are external fixture inputs.
See `audit.md` for mathematical conditions and `audit_state_check.json` for the
measured error from discarding subcell state.

```bash
# Native frozen-field accuracy, time and independent-process peak memory.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  research/pc_adaptive_front/benchmarks.py \
  --native32 /path/to/native32/source_capacity.npz \
  --native64 /path/to/native64/native_fields.npz --measure-rss \
  --output /path/to/results.json

# A second half-step with exactly the same resolved initial front in both models.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  research/pc_adaptive_front/incremental_benchmark.py \
  --native /path/to/native32/source_capacity.npz \
  --angles 384 --output /path/to/incremental_results.json

# Actual restricted native 21cmFAST coeval and matched transport comparison.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  research/pc_adaptive_front/native_check.py \
  --directory /path/to/new_native_run --threads 4 \
  --output /path/to/native_results.json
```

`initial_state` carries the gas geometry only. Every call receives a new emitted
photon budget; horizon-retained or escaped photons are reported separately and
are not silently re-injected. A fully transparent periodic box takes an exact
zero-ray shortcut, retaining every new photon in its reported ledger.

Native 21cmFAST uses its requested OpenMP threads; the event-transport kernel
remains serial. `REPORT.md` summarizes the measured accuracy and cost;
`results.json` retains all cases and source hashes.

## Pull and compare on Gadi

From the existing fork checkout on a Gadi login node:

```bash
(
  set -euo pipefail
  cd /g/data/qp00/sz4640/21cmFAST
  git fetch https://github.com/ChangqIngovo/21cmFAST.git Photon_conserving
  git switch Photon_conserving
  git merge --ff-only FETCH_HEAD
  bash research/pc_adaptive_front/gadi_prepare.sh --submit
)
```

The dedicated preparation creates a fresh detached worktree and editable
environment, selects an available Python >=3.12 module, and builds on the login
node with two build threads. It records the selected modules, package versions,
commit and native binary hashes. It does not reuse the older Python 3.11 or
Meraxes environment. `PC_PYTHON_MODULE` can select a module explicitly;
`PC_PYTHON` can supply an absolute path to an existing suitable interpreter.

The default job is **normal, 4 CPUs, 16 GB, 4 hours**, with no MPI launcher.
`PC_THREADS` (1–32), `PC_MEM`, `PC_WALLTIME` and `PC_QUEUE` override resources.
Omit `--submit` to prepare and print the exact submission command. The job
performs no package installation on compute nodes. The old
`tools/gadi_qin20_prepare.sh --submit` runs a different algorithm and is not the
entry point for this adaptive-front comparison.

The script prints the job ID and fresh directory, normally
`/g/data/qp00/sz4640/21cmfast_runs/adaptive_<commit>_<time>_<suffix>/`.

| Output relative to the printed directory | Contents |
|---|---|
| `jobid.txt`, `run_<jobid>.log` | Submitted job and progress |
| `run_manifest.json`, `prepared.sha256` | Build and input provenance |
| `results/comparison.json` | Sphere checks and matched-source accuracy/time/RSS |
| `results/comparison.png`, `results/comparison.pdf` | Three-panel comparison figure |
| `results/native32_comparison.json`, `results/native64_comparison.json` | Restricted native integration and transport checks |
| `results/native32/native_fields.npz`, `results/native64/native_fields.npz` | Sources, capacities, native ES and adaptive PC ionization/brightness |
| `results/incremental_comparison.json` | Fixed-K384 continuation from the same resolved front |
| `COMPLETED` / `FAILED` | Success / exit status |

Both native grids are newly generated at z=8, box=200 cMpc, seed=1234 using the
restricted L-INTEGRAL setup. All models in a comparison receive the same source
and capacity arrays. This newly generated 32³ fixture differs from the
E-INTEGRAL 32³ fixture used in the published local cost table. The job tests the
new spatial solver and its native interface; it does not evolve full coupled
Qin20 histories.
