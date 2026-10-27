#!/usr/bin/env bash
# Freeze code, install a separate CPU training environment, optionally submit.
set -euo pipefail
submit=false
manifest=
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --manifest) manifest=$(realpath "${2:?Supply dataset_manifest.json}"); shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/small_pilot_prepare.sh [--manifest FILE] [--submit]'
            echo 'Defaults: 32 train, 8 validation, 8 test, 4 external cases; 80 epochs; one seed.'
            echo 'SMALL_RUNTIME_PARENT / SMALL_NATIVE_PYTHON select the existing native runtime.'
            echo 'SMALL_TRAIN_CASES / SMALL_VALIDATION_CASES / SMALL_TEST_CASES / SMALL_EXTERNAL_CASES / SMALL_EPOCHS override size.'
            echo 'SMALL_RUN_ROOT / SMALL_WORKERS / SMALL_MEM / SMALL_WALLTIME / SMALL_PROJECT override output and allocation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/small_pilot_training.py
revision=$(git -C "$repo" rev-parse HEAD)
parent=${SMALL_RUNTIME_PARENT:-${C2_RUNTIME_PARENT:-/g/data/qp00/sz4640/21cmfast_runs/large_pc_5e40d8ab21_20261024T042855_XhUc}}
parent=$(realpath "$parent")
native_python=${SMALL_NATIVE_PYTHON:-$parent/venv/bin/python}
test -x "$native_python"
test -f "$parent/runtime.env"
module --force purge
source "$parent/runtime.env"
module load "${PC_MODULES[@]}"
module load gcc/12.2.0 openmpi/4.1.4
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$native_python" -c 'import sys; assert sys.version_info >= (3,12); import numpy, scipy, py21cmfast'
workers=${SMALL_WORKERS:-16}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'Use 1 through 32 CPUs.' >&2; exit 2; }
root=${SMALL_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
mkdir -p "$root"
run=$(mktemp -d "$root/small_nn_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* ]] || { echo 'PBS path contains a separator.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" worktree add --detach "$run/source" "$revision"
cp "$parent/runtime.env" "$run/runtime.env"
printf '\nPC_MODULES+=(gcc/12.2.0 openmpi/4.1.4)\n' >> "$run/runtime.env"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$native_python" > "$run/native_python.txt"
printf '%s\n' "$manifest" > "$run/input_manifest.txt"
module -t list > "$run/modules.txt" 2>&1

# Native dependencies are not changed. This venv only reads arrays and trains.
"$native_python" -m venv "$run/train_venv"
python=$run/train_venv/bin/python
"$python" -m pip install 'numpy>=1.26,<3' 'scipy>=1.12,<2' 'matplotlib>=3.8,<4'
"$python" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
"$python" -m pip freeze > "$run/training_packages.txt"
export SMALL_PREPARED_RUN="$run"
"$python" - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ['SMALL_PREPARED_RUN'])
limits = {'train_cases': (32, 64), 'validation_cases': (8, 16),
          'test_cases': (8, 16), 'external_cases': (4, 16), 'epochs': (80, 100000)}
config = {}
for name, (default, maximum) in limits.items():
    value = int(os.environ.get('SMALL_' + name.upper(), default))
    if not 1 <= value <= maximum:
        raise ValueError(f'{name} must be between 1 and {maximum}')
    config[name] = value
config.update(history_steps=3, model_seeds=[11], convergence_certified=False)
(run / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
PY
if [[ -z $manifest ]]; then
    upstream=${SMALL_UPSTREAM_REPO:-${C2_UPSTREAM_REPO:-/g/data/qp00/sz4640/c2ray_cache/C2-Ray3Dm1D_Helium.git}}
    pinned=a363656ff1c3b4b4c7334f398578d8c469154515
    if [[ ! -d $upstream ]]; then
        mkdir -p "$(dirname "$upstream")"
        git clone --mirror https://github.com/garrelt/C2-Ray3Dm1D_Helium.git "$upstream"
    fi
    git -C "$upstream" cat-file -e "$pinned^{commit}"
    mkdir "$run/upstream"
    git -C "$upstream" archive "$pinned" | tar -x -C "$run/upstream"
    printf '%s\n' "$pinned" > "$run/upstream_commit.txt"
else
    cd "$run/source"
    "$python" -m research.es_pc_emulator.small_pilot_training --help >/dev/null
    "$python" - "$manifest" <<'PY'
import sys
from research.es_pc_emulator.physical_pilot_training import load_manifest
load_manifest(sys.argv[1])
PY
fi
cp "$run/source/research/es_pc_emulator/small_pilot.pbs" "$run/submit.pbs"
project=${SMALL_PROJECT:-qp00}
storage=${SMALL_STORAGE:-gdata/qp00+scratch/qp00}
command=(qsub -P "$project" -q normal -l "ncpus=$workers,mem=${SMALL_MEM:-64GB},walltime=${SMALL_WALLTIME:-08:00:00},storage=$storage" -v "SMALL_RUN=$run" "$run/submit.pbs")
printf 'Prepared: %s\n' "$run"
printf '%q ' "${command[@]}"
printf '\n'
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
