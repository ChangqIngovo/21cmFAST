#!/usr/bin/env bash
# Reuse completed component-pilot data and Python for a small loss-weight scan.
set -euo pipefail
submit=false
parent=${LOSS_SWEEP_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply a completed component_nn or small_nn run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/loss_sweep_prepare.sh [--parent RUN] [--submit]'
            echo 'Reuses the latest completed component_nn_* (fallback small_nn_*) in LOSS_SWEEP_RUN_ROOT.'
            echo 'Default root: /g/data/qp00/sz4640/21cmfast_runs; no new simulations or packages.'
            echo 'LOSS_SWEEP_EPOCHS/PATIENCE/DIAGNOSTIC_INTERVAL/SEEDS: 120/30/5/11,23.'
            echo 'LOSS_SWEEP_COMPONENT_WEIGHTS/POWER_WEIGHTS: 0.003,0.01,0.03; ANCHOR_COMPONENT_WEIGHT: 0.01.'
            echo 'LOSS_SWEEP_WORKERS/MEM/WALLTIME/PROJECT/STORAGE: 16/64GB/12h/qp00/gdata+scratch qp00.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/loss_sweep.py
revision=$(git -C "$repo" rev-parse HEAD)
root=${LOSS_SWEEP_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
if [[ -z $parent ]]; then
    parent=$(python3 - "$root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for pattern in ('component_nn_*', 'small_nn_*'):
    runs = []
    for path in root.glob(pattern):
        markers = [p for p in (path/'COMPLETED', path/'training/COMPLETED') if p.is_file()]
        if not markers or not (path/'training/configuration.json').is_file() or not (path/'runtime.env').is_file():
            continue
        pointer = path/'python_path.txt'
        python = Path(pointer.read_text().strip()) if pointer.is_file() else path/'train_venv/bin/python'
        if python.is_absolute() and python.is_file():
            runs.append((max(p.stat().st_mtime for p in markers), path))
    if runs:
        print(max(runs, key=lambda item: item[0])[1].resolve())
        break
else:
    raise SystemExit('No completed pilot found. Supply --parent /absolute/component_nn_or_small_nn_run.')
PY
)
fi
parent=$(realpath "$parent")
[[ -f $parent/COMPLETED || -f $parent/training/COMPLETED ]] || {
    echo "Parent run is incomplete: $parent" >&2; exit 2;
}
test -f "$parent/training/configuration.json"
test -f "$parent/runtime.env"
if [[ -f $parent/python_path.txt ]]; then
    python=$(cat "$parent/python_path.txt")
else
    python=$parent/train_venv/bin/python
fi
[[ $python = /* && -x $python ]] || { echo "Missing parent Python: $python" >&2; exit 2; }
module --force purge
source "$parent/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -c 'import torch, numpy, scipy, matplotlib; print("Training runtime:", torch.__version__)'
workers=${LOSS_SWEEP_WORKERS:-16}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'Use 1 through 32 CPUs.' >&2; exit 2; }
mkdir -p "$root"
run=$(mktemp -d "$root/loss_sweep_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* && $run != *$'\r'* ]] || { echo 'Invalid PBS run path.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" worktree add --detach "$run/source" "$revision"
cp "$parent/runtime.env" "$run/runtime.env"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
printf '%s\n' "$python" > "$run/python_path.txt"
export LOSS_SWEEP_PREPARED_RUN="$run" LOSS_SWEEP_SELECTED_PARENT="$parent"
cd "$run/source"
"$python" - <<'PY'
import json
import math
import os
from collections import Counter
from pathlib import Path

from research.es_pc_emulator.physical_pilot_training import load_manifest

run = Path(os.environ['LOSS_SWEEP_PREPARED_RUN'])
parent = Path(os.environ['LOSS_SWEEP_SELECTED_PARENT'])
previous = json.loads((parent/'training/configuration.json').read_text())
manifest = load_manifest(previous['manifest'])
wanted = {row['case_id']: row for row in previous['cases']}
if len(wanted) != len(previous['cases']):
    raise ValueError('Duplicate case IDs in previous training configuration')
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
configuration = {key: previous[key] for key in ['history_steps', 'width', 'batch_size', 'learning_rate',
                                                'auxiliary_weight', 'power_floor']}
for name, default in [('epochs', 120), ('patience', 30), ('diagnostic_interval', 5)]:
    value = int(os.environ.get('LOSS_SWEEP_'+name.upper(), default))
    if value < 1:
        raise ValueError(name+' must be positive')
    configuration[name] = value
for name in ['component_weights', 'power_weights']:
    values = [float(x) for x in os.environ.get('LOSS_SWEEP_'+name.upper(), '.003,.01,.03').split(',')]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values) or len(set(values)) != len(values):
        raise ValueError(name+' requires distinct finite positive weights')
    configuration[name] = values
anchor = float(os.environ.get('LOSS_SWEEP_ANCHOR_COMPONENT_WEIGHT', '.01'))
if not math.isfinite(anchor) or anchor <= 0:
    raise ValueError('LOSS_SWEEP_ANCHOR_COMPONENT_WEIGHT must be finite and positive')
if anchor not in configuration['component_weights']:
    raise ValueError('Anchor component weight must appear in LOSS_SWEEP_COMPONENT_WEIGHTS')
configuration['anchor_component_weight'] = anchor
seeds = [int(x) for x in os.environ.get('LOSS_SWEEP_SEEDS', '11,23').split(',')]
if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
    raise ValueError('LOSS_SWEEP_SEEDS requires distinct nonnegative integers')
configuration.update(model_seeds=seeds,
    train_cases=counts['train'], validation_cases=counts['validation'],
    test_cases=counts['test'], external_cases=counts['external_test'])
(run/'configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')
print('Reusing:', parent)
print('Whole-history case counts:', dict(counts))
print('Component weights:', configuration['component_weights'])
print('P21 weights at component weight', anchor, ':', configuration['power_weights'])
print('Network seeds:', seeds)
PY
cp "$run/source/research/es_pc_emulator/loss_sweep_run.pbs" "$run/submit.pbs"
command=(qsub -P "${LOSS_SWEEP_PROJECT:-qp00}" -q normal
         -l "ncpus=$workers,mem=${LOSS_SWEEP_MEM:-64GB},walltime=${LOSS_SWEEP_WALLTIME:-12:00:00},storage=${LOSS_SWEEP_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "LOSS_SWEEP_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared: %s\n' "$run"
printf '%q ' "${command[@]}"
printf '\nResume after an interrupted job: bash %q\n' "$run/resubmit.sh"
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
