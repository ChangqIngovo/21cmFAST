#!/usr/bin/env bash
# Freeze a full-history pilot while reusing the existing native/training runtimes.
set -euo pipefail
submit=false
parent=${FULL_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply a completed pilot run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/full_history_prepare.sh [--parent RUN] [--submit]'
            echo 'Parent preference: latest completed loss_sweep_*, component_nn_*, then small_nn_*.'
            echo 'Generates new full native histories with the SAME case IDs/parameters/splits; reuses Python environments.'
            echo 'FULL_Z_START/Z_END/INTERVALS/END_XHI: 20/4/40/0.01; initial state included.'
            echo 'Training starts only after initial/final evolution coverage passes; no completion is assumed from redshift alone.'
            echo 'FULL_EPOCHS/PATIENCE/DIAGNOSTIC_INTERVAL/SEEDS: 120/30/5/11,23.'
            echo 'FULL_WORKERS/NATIVE_PARALLEL/MEM/WALLTIME: 32/2/128GB/24h.'
            echo 'FULL_RUN_ROOT/PROJECT/STORAGE: /g/data/qp00/sz4640/21cmfast_runs/qp00/gdata+scratch qp00.'
            echo 'FULL_NATIVE_PYTHON/RUNTIME_PARENT/UPSTREAM_REPO optionally override existing native resources.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/full_history.py
revision=$(git -C "$repo" rev-parse HEAD)
root=${FULL_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
if [[ -z $parent ]]; then
    parent=$(python3 - "$root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for pattern in ('loss_sweep_*', 'component_nn_*', 'small_nn_*'):
    runs = []
    for path in root.glob(pattern):
        markers = [p for p in (path/'COMPLETED', path/'training/COMPLETED') if p.is_file()]
        if not markers or not (path/'runtime.env').is_file():
            continue
        if not any((path/name).is_file() for name in ('configuration.json', 'training/configuration.json')):
            continue
        pointer = path/'python_path.txt'
        python = Path(pointer.read_text().strip()) if pointer.is_file() else path/'train_venv/bin/python'
        if python.is_absolute() and python.is_file():
            runs.append((max(p.stat().st_mtime for p in markers), path))
    if runs:
        print(max(runs, key=lambda item: item[0])[1].resolve())
        break
else:
    raise SystemExit('No completed pilot found. Supply --parent /absolute/completed_run.')
PY
)
fi
parent=$(realpath "$parent")
[[ -f $parent/COMPLETED || -f $parent/training/COMPLETED ]] || {
    echo "Parent run is incomplete: $parent" >&2; exit 2;
}
if [[ -f $parent/python_path.txt ]]; then
    python=$(cat "$parent/python_path.txt")
else
    python=$parent/train_venv/bin/python
fi
[[ $python = /* && -x $python ]] || { echo "Missing training Python: $python" >&2; exit 2; }
native_python=${FULL_NATIVE_PYTHON:-}
if [[ -z $native_python ]]; then
    native_python=$(python3 - "$parent" <<'PY'
import sys
from pathlib import Path

path, visited = Path(sys.argv[1]).resolve(), set()
while path not in visited:
    visited.add(path)
    pointer = path/'native_python.txt'
    if pointer.is_file():
        print(pointer.read_text().strip())
        break
    parent = path/'parent_run.txt'
    if not parent.is_file():
        raise SystemExit('Cannot locate native_python.txt in parent chain; set FULL_NATIVE_PYTHON.')
    path = Path(parent.read_text().strip()).resolve()
else:
    raise SystemExit('Cycle in parent_run.txt chain')
PY
)
fi
[[ $native_python = /* && -x $native_python ]] || { echo "Missing native Python: $native_python" >&2; exit 2; }
runtime_parent=${FULL_RUNTIME_PARENT:-$parent}
runtime_parent=$(realpath "$runtime_parent")
test -f "$runtime_parent/runtime.env"
module --force purge
source "$runtime_parent/runtime.env"
module load "${PC_MODULES[@]}"
module load gcc/12.2.0 openmpi/4.1.4
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -c 'import torch, numpy, scipy, matplotlib; print("Training runtime:", torch.__version__)'
"$native_python" -c 'import sys; assert sys.version_info >= (3,12); import numpy, scipy, py21cmfast; print("Native runtime:", sys.executable)'
workers=${FULL_WORKERS:-32}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'Use 1 through 32 CPUs.' >&2; exit 2; }
mkdir -p "$root"
run=$(mktemp -d "$root/full_history_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* && $run != *$'\r'* ]] || { echo 'Invalid PBS run path.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" worktree add --detach "$run/source" "$revision"
cp "$runtime_parent/runtime.env" "$run/runtime.env"
printf '\nPC_MODULES+=(gcc/12.2.0 openmpi/4.1.4)\n' >> "$run/runtime.env"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
printf '%s\n' "$python" > "$run/python_path.txt"
printf '%s\n' "$native_python" > "$run/native_python.txt"
module -t list > "$run/modules.txt" 2>&1
export FULL_PREPARED_RUN="$run" FULL_SELECTED_PARENT="$parent" FULL_PREPARED_CPUS="$workers"
cd "$run/source"
"$python" - <<'PY'
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from research.es_pc_emulator.physical_pilot_training import load_manifest

run = Path(os.environ['FULL_PREPARED_RUN'])
parent = Path(os.environ['FULL_SELECTED_PARENT'])
training_configuration = parent/'training/configuration.json'
if training_configuration.is_file():
    previous = json.loads(training_configuration.read_text())
    manifest_path = Path(previous['manifest'])
else:
    previous = json.loads((parent/'configuration.json').read_text())
    manifest_path = parent/'dataset_manifest.json'
manifest = load_manifest(manifest_path)
if 'cases' in previous:
    wanted = {row['case_id']: row for row in previous['cases']}
    if len(wanted) != len(previous['cases']):
        raise ValueError('Duplicate previous case IDs')
    records = [row for row in manifest['cases'] if row['case_id'] in wanted]
    if len(records) != len(wanted):
        raise ValueError('Missing cases in parent manifest')
    for row in records:
        old = wanted[row['case_id']]
        if row['fields_sha256'] != old['sha256'] or row['split'] != old['split'] or row['seed'] != old['seed']:
            raise ValueError('Previous identity, split or hash changed')
else:
    records = manifest['cases']
allowed = {'train', 'validation', 'test', 'external_test'}
if any(row['split'] not in allowed for row in records):
    raise ValueError('Parent manifest must contain only selected training/evaluation cases')
counts = Counter(row['split'] for row in records)
if any(counts[name] == 0 for name in ('train', 'validation', 'test')):
    raise ValueError('Missing mandatory whole-history split')
for row in records:
    if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['fields_sha256']:
        raise ValueError('Parent field file checksum changed: '+row['case_id'])
manifest['cases'] = records
(run/'parent_dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
reference = [{key: row[key] for key in ('case_id', 'split', 'seed', 'theta')} for row in records]
(run/'reference_cases.json').write_text(json.dumps(reference, indent=2)+'\n')
(run/'selected_cases.txt').write_text(','.join(row['case_id'] for row in records)+'\n')
training = {key: previous[key] for key in ('history_steps', 'width', 'batch_size', 'learning_rate',
                                          'auxiliary_weight', 'power_floor')}
for name, default in [('epochs', 120), ('patience', 30), ('diagnostic_interval', 5)]:
    value = int(os.environ.get('FULL_'+name.upper(), default))
    if value < 1:
        raise ValueError(name+' must be positive')
    training[name] = value
seeds = [int(value) for value in os.environ.get('FULL_SEEDS', '11,23').split(',')]
if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
    raise ValueError('FULL_SEEDS must contain distinct nonnegative integers')
training.update(model_seeds=seeds, arms=['thermo_components_p21'],
                component_weight=.01, power_weight=.01, snapshot_scope='all',
                train_cases=counts['train'], validation_cases=counts['validation'],
                test_cases=counts['test'], external_cases=counts['external_test'])
z_start = float(os.environ.get('FULL_Z_START', 20.))
z_end = float(os.environ.get('FULL_Z_END', 4.))
end_xhi = float(os.environ.get('FULL_END_XHI', .01))
intervals = int(os.environ.get('FULL_INTERVALS', 40))
cpus = int(os.environ['FULL_PREPARED_CPUS'])
parallel = int(os.environ.get('FULL_NATIVE_PARALLEL', min(2, cpus)))
if not all(math.isfinite(value) for value in (z_start, z_end, end_xhi)):
    raise ValueError('Nonfinite history settings')
if not z_start > z_end >= 0 or not 0 < end_xhi < .1 or intervals < 2:
    raise ValueError('Require z_start > z_end >= 0, 0 < end_xhi < 0.1, intervals >= 2')
if not 1 <= parallel <= cpus:
    raise ValueError('FULL_NATIVE_PARALLEL must fit within FULL_WORKERS')
configuration = {
    'parent_run': str(parent), 'parent_manifest': str(manifest_path.resolve()),
    'parent_manifest_sha256': manifest['source_manifest_sha256'],
    'native': {'z_start': z_start, 'z_end': z_end, 'intervals': intervals,
               'include_initial': True, 'end_xhi': end_xhi, 'parallel': parallel},
    'training': training,
    'coverage_requirement': 'Every case must start near neutral and reach the declared final neutral threshold before training',
    'convergence_certified': False,
}
(run/'configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')
print('Runtime/data parent:', parent)
print('Whole-history case counts:', dict(counts))
print('Native endpoints:', intervals+1, 'redshifts:', z_start, 'to', z_end)
print('Required final mean xHI <=', end_xhi)
print('Native parallel cases:', parallel, 'CPUs per case:', cpus//parallel)
print('Training seeds:', seeds, 'snapshot scope: all')
PY
upstream=${FULL_UPSTREAM_REPO:-/g/data/qp00/sz4640/c2ray_cache/C2-Ray3Dm1D_Helium.git}
pinned=a363656ff1c3b4b4c7334f398578d8c469154515
if [[ ! -d $upstream ]]; then
    mkdir -p "$(dirname "$upstream")"
    git clone --mirror https://github.com/garrelt/C2-Ray3Dm1D_Helium.git "$upstream"
fi
git -C "$upstream" cat-file -e "$pinned^{commit}"
mkdir "$run/upstream"
git -C "$upstream" archive "$pinned" | tar -x -C "$run/upstream"
printf '%s\n' "$pinned" > "$run/upstream_commit.txt"
cp "$run/source/research/es_pc_emulator/full_history_run.pbs" "$run/submit.pbs"
command=(qsub -P "${FULL_PROJECT:-qp00}" -q normal
         -l "ncpus=$workers,mem=${FULL_MEM:-128GB},walltime=${FULL_WALLTIME:-24:00:00},storage=${FULL_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "FULL_HISTORY_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared: %s\n' "$run"
printf '%q ' "${command[@]}"
printf '\nResume after interruption: bash %q\n' "$run/resubmit.sh"
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
