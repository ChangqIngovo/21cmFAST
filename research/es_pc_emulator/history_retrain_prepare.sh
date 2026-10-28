#!/usr/bin/env bash
# Refit the full-history emulator using existing RT labels and Python runtime.
set -euo pipefail
submit=false
parent=${RETRAIN_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply a completed full_history run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/history_retrain_prepare.sh [--parent RUN] [--submit]'
            echo 'Reuses all cases from the latest completed full_history_*; no RT runs or installations.'
            echo 'RETRAIN_EPOCHS/PATIENCE/DIAGNOSTIC_INTERVAL/SEEDS: 120/30/5/11,23.'
            echo 'RETRAIN_LEARNING_RATE/RELATIVE_POWER_FLOOR/PRIOR_EPSILON: 0.0003/0.0001/0.0001.'
            echo 'RETRAIN_WORKERS/MEM/WALLTIME: 16/64GB/12:00:00.'
            echo 'RETRAIN_RUN_ROOT/PROJECT/STORAGE optionally override Gadi paths and allocation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/history_retrain.py
revision=$(git -C "$repo" rev-parse HEAD)
root=${RETRAIN_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
if [[ -z $parent ]]; then
    parent=$(python3 - "$root" <<'PY'
import sys
from pathlib import Path

runs = [path for path in Path(sys.argv[1]).glob('full_history_*')
        if all((path/name).is_file() for name in
               ('COMPLETED', 'native/dataset_manifest.json', 'coverage.json',
                'training/configuration.json', 'python_path.txt', 'runtime.env'))]
if not runs:
    raise SystemExit('No completed full_history_* run found; supply --parent RUN.')
print(max(runs, key=lambda path: (path/'COMPLETED').stat().st_mtime).resolve())
PY
)
fi
parent=$(realpath "$parent")
test -f "$parent/COMPLETED"
python=$(cat "$parent/python_path.txt")
[[ $python = /* && -x $python ]] || { echo "Missing existing training Python: $python" >&2; exit 2; }
module --force purge
source "$parent/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -c 'import torch, numpy, scipy, matplotlib; print("Training runtime:", torch.__version__)'
workers=${RETRAIN_WORKERS:-16}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'Use 1 through 32 CPUs.' >&2; exit 2; }
mkdir -p "$root"
run=$(mktemp -d "$root/history_retrain_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* && $run != *$'\r'* ]] || { echo 'Invalid PBS run path.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" worktree add --detach "$run/source" "$revision"
cp "$parent/runtime.env" "$run/runtime.env"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
printf '%s\n' "$python" > "$run/python_path.txt"
module -t list > "$run/modules.txt" 2>&1
cd "$run/source"
"$python" -m research.es_pc_emulator.history_retrain prepare --run "$run" --parent "$parent"
cp "$run/source/research/es_pc_emulator/history_retrain_run.pbs" "$run/submit.pbs"
command=(qsub -P "${RETRAIN_PROJECT:-qp00}" -q normal
         -l "ncpus=$workers,mem=${RETRAIN_MEM:-64GB},walltime=${RETRAIN_WALLTIME:-12:00:00},storage=${RETRAIN_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "HISTORY_RETRAIN_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared training-only run: %s\nData parent: %s\n' "$run" "$parent"
printf '%q ' "${command[@]}"
printf '\nResume after interruption: bash %q\n' "$run/resubmit.sh"
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
