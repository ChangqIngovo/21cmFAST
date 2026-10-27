#!/usr/bin/env bash
# Reuse the previous small-pilot data and CPU environment for loss comparisons.
set -euo pipefail
submit=false
parent=${COMPONENT_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply the completed small_nn run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/component_prepare.sh [--parent RUN] [--submit]'
            echo 'Default: latest completed small_nn_* under COMPONENT_RUN_ROOT or /g/data/qp00/sz4640/21cmfast_runs.'
            echo 'Reuses that run data/splits/Python; no RT generation or package installation.'
            echo 'COMPONENT_EPOCHS/PATIENCE/WEIGHT/POWER_WEIGHT/SEEDS override 80/15/0.01/0.01/11,23.'
            echo 'COMPONENT_WORKERS/MEM/WALLTIME/PROJECT/STORAGE override 16/64GB/8h/qp00/gdata+scratch qp00.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/spectral_components.py
revision=$(git -C "$repo" rev-parse HEAD)
root=${COMPONENT_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
if [[ -z $parent ]]; then
    parent=$(python3 - "$root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = [p for p in root.glob('small_nn_*') if
        (p/'training/COMPLETED').is_file() and (p/'training/configuration.json').is_file()
        and (p/'train_venv/bin/python').is_file() and (p/'runtime.env').is_file()]
if not runs:
    raise SystemExit('No completed small_nn run found. Supply --parent /absolute/small_nn_run.')
print(max(runs, key=lambda p: (p/'training/COMPLETED').stat().st_mtime).resolve())
PY
)
fi
parent=$(realpath "$parent")
test -f "$parent/training/COMPLETED"
test -f "$parent/training/configuration.json"
test -x "$parent/train_venv/bin/python"
module --force purge
source "$parent/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=$parent/train_venv/bin/python
"$python" -c 'import torch, numpy, scipy, matplotlib; print("Training runtime:", torch.__version__)'
workers=${COMPONENT_WORKERS:-16}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'Use 1 through 32 CPUs.' >&2; exit 2; }
mkdir -p "$root"
run=$(mktemp -d "$root/component_nn_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* && $run != *$'\r'* ]] || { echo 'Invalid PBS run path.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" worktree add --detach "$run/source" "$revision"
cp "$parent/runtime.env" "$run/runtime.env"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
printf '%s\n' "$python" > "$run/python_path.txt"
export COMPONENT_PREPARED_RUN="$run" COMPONENT_SELECTED_PARENT="$parent"
cd "$run/source"
"$python" - <<'PY'
import json
import math
import os
from collections import Counter
from pathlib import Path

from research.es_pc_emulator.physical_pilot_training import load_manifest

run = Path(os.environ['COMPONENT_PREPARED_RUN'])
parent = Path(os.environ['COMPONENT_SELECTED_PARENT'])
previous = json.loads((parent/'training/configuration.json').read_text())
manifest = load_manifest(previous['manifest'])
wanted = {row['case_id']: row for row in previous['cases']}
records = [row for row in manifest['cases'] if row['case_id'] in wanted]
if len(records) != len(wanted):
    raise ValueError('Previous training cases are missing from its dataset manifest')
for row in records:
    old = wanted[row['case_id']]
    if row['fields_sha256'] != old['sha256'] or row['split'] != old['split'] or row['seed'] != old['seed']:
        raise ValueError('Previous case identity, split or checksum changed')
manifest['cases'] = records
(run/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
counts = Counter(row['split'] for row in records)
configuration = {key: previous[key] for key in ['history_steps','width','batch_size','learning_rate',
                                                'auxiliary_weight','power_floor']}
for name in ['epochs','patience']:
    configuration[name] = int(os.environ.get('COMPONENT_'+name.upper(), previous[name]))
    if configuration[name] < 1:
        raise ValueError(name+' must be positive')
for name in ['component_weight','power_weight']:
    variable = 'COMPONENT_WEIGHT' if name == 'component_weight' else 'COMPONENT_POWER_WEIGHT'
    value = float(os.environ.get(variable, '.01'))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(variable+' must be finite and positive')
    configuration[name] = value
seeds = [int(x) for x in os.environ.get('COMPONENT_SEEDS','11,23').split(',')]
if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
    raise ValueError('COMPONENT_SEEDS requires distinct nonnegative integers')
configuration.update(model_seeds=seeds,
    arms=['thermo','thermo_p21','thermo_pxx','thermo_cross','thermo_nonlinear',
          'thermo_components','thermo_components_p21'],
    train_cases=counts['train'], validation_cases=counts['validation'],
    test_cases=counts['test'], external_cases=counts['external_test'])
(run/'configuration.json').write_text(json.dumps(configuration,indent=2)+'\n')
print('Reusing:', parent)
print('Whole-history case counts:', dict(counts))
print('Models:', configuration['arms'], 'seeds:', seeds)
PY
cp "$run/source/research/es_pc_emulator/component_run.pbs" "$run/submit.pbs"
command=(qsub -P "${COMPONENT_PROJECT:-qp00}" -q normal
         -l "ncpus=$workers,mem=${COMPONENT_MEM:-64GB},walltime=${COMPONENT_WALLTIME:-08:00:00},storage=${COMPONENT_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "COMPONENT_RUN=$run" "$run/submit.pbs")
printf 'Prepared: %s\n' "$run"
printf '%q ' "${command[@]}"
printf '\n'
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
