#!/usr/bin/env bash
# Freeze a small in-sample fitting test using completed RT simulations.
set -euo pipefail
submit=false
parent=${FIT_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply a completed history run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/fit_ceiling_prepare.sh [--parent RUN] [--submit]'
            echo 'Reuses 1/2 training simulations; no RT generation or package installs.'
            echo 'FIT_EPOCHS/DIAGNOSTIC_INTERVAL/SEEDS/CASE_COUNTS: 400/10/11,23/1,2.'
            echo 'FIT_ARMS: comma-separated joint_constant16,joint_cosine16,joint_cosine32,field_cosine32,power_cosine32.'
            echo 'FIT_LEARNING_RATE/MIN_LEARNING_RATE: 0.0003/0.000003.'
            echo 'FIT_WORKERS/MEM/WALLTIME: 16/64GB/12:00:00.'
            echo 'FIT_RUN_ROOT/PROJECT/STORAGE optionally override Gadi paths and allocation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/fit_ceiling.py
revision=$(git -C "$repo" rev-parse HEAD)
root=${FIT_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
if [[ -z $parent ]]; then
    parent=$(python3 - "$root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for pattern, manifest in (('history_retrain_*', 'dataset_manifest.json'),
                          ('full_history_*', 'native/dataset_manifest.json')):
    runs = [path for path in root.glob(pattern)
            if all((path/name).is_file() for name in
                   ('COMPLETED', manifest, 'coverage.json', 'training/configuration.json',
                    'python_path.txt', 'runtime.env'))]
    if runs:
        print(max(runs, key=lambda path: (path/'COMPLETED').stat().st_mtime).resolve())
        break
else:
    raise SystemExit('No completed history run found; supply --parent RUN.')
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
workers=${FIT_WORKERS:-16}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'Use 1 through 32 CPUs.' >&2; exit 2; }
mkdir -p "$root"
run=$(mktemp -d "$root/fit_ceiling_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
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
"$python" -m research.es_pc_emulator.fit_ceiling_run prepare --run "$run" --parent "$parent"
cp "$run/source/research/es_pc_emulator/fit_ceiling_run.pbs" "$run/submit.pbs"
command=(qsub -P "${FIT_PROJECT:-qp00}" -q normal
         -l "ncpus=$workers,mem=${FIT_MEM:-64GB},walltime=${FIT_WALLTIME:-12:00:00},storage=${FIT_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "FIT_CEILING_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared fitting-ceiling run: %s\nData parent: %s\n' "$run" "$parent"
printf '%q ' "${command[@]}"
printf '\nResume after interruption: bash %q\n' "$run/resubmit.sh"
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
