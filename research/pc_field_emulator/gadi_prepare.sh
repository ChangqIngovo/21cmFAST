#!/usr/bin/env bash
# Prepare training against existing PC checkpoints; no new simulation is run.
set -euo pipefail

submit=false
input=${PC_EMULATOR_INPUT:-/g/data/qp00/sz4640/21cmfast_runs/evolution_pc_30e8497de2_20261024T122119_qC2V}
input_argument=false
for argument in "$@"; do
    case "$argument" in
        --submit) submit=true ;;
        --help)
            echo 'Usage: bash research/pc_field_emulator/gadi_prepare.sh [INPUT_RUN_OR_RESULTS] [--submit]'
            echo 'Defaults: completed four-seed PC run; 32 CPUs, normal, 64GB, 2 hours.'
            echo 'Environment: PC_EMULATOR_INPUT, PC_RUNTIME_PARENT, PC_PYTHON, PC_EMULATOR_RUN_ROOT,'
            echo 'PC_EMULATOR_EPOCHS, PC_MEM, PC_WALLTIME, PC_QUEUE.'
            echo 'The input run is read-only. The time request is an allocation, not an ETA.'
            exit 0 ;;
        --*) echo "Unknown option: $argument" >&2; exit 2 ;;
        *)
            if "$input_argument"; then
                echo 'Specify only one input path.' >&2
                exit 2
            fi
            input=$argument
            input_argument=true ;;
    esac
done

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/pc_field_emulator/train.py
git -C "$repo" cat-file -e HEAD:research/pc_field_emulator/gadi_train.pbs
git -C "$repo" diff --exit-code HEAD -- research/pc_field_emulator
if [[ -n $(git -C "$repo" ls-files --others --exclude-standard -- research/pc_field_emulator) ]]; then
    echo 'Commit new emulator files before preparing the archived job.' >&2
    exit 2
fi
input=$(realpath "$input")
test -d "$input"
parent=${PC_RUNTIME_PARENT:-$input}
if [[ $(basename "$parent") == results ]]; then parent=$(dirname "$parent"); fi
parent=$(realpath "$parent")
test -f "$parent/runtime.env"

pc_python=$(python3 - "$parent" <<'PY'
import json
import os
from pathlib import Path
import sys

parent = Path(sys.argv[1])
python = os.environ.get('PC_PYTHON')
if not python and (parent / 'python_path.txt').is_file():
    python = (parent / 'python_path.txt').read_text().strip()
if not python and (parent / 'run_manifest.json').is_file():
    python = json.loads((parent / 'run_manifest.json').read_text()).get('python')
if not python or not Path(python).is_absolute():
    raise SystemExit('No absolute Python path found; provide PC_PYTHON.')
print(python)
PY
)
test -x "$pc_python"
revision=$(git -C "$repo" rev-parse HEAD)
root=${PC_EMULATOR_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
mkdir -p "$root"
run=$(mktemp -d "$root/field_emulator_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
mkdir "$run/source"
git -C "$repo" archive "$revision" research/pc_field_emulator | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$input" > "$run/input_root.txt"
printf '%s\n' "$parent" > "$run/runtime_parent.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$parent/runtime.env" "$run/runtime.env"
cp "$run/source/research/pc_field_emulator/gadi_train.pbs" "$run/submit.pbs"

module --force purge
source "$run/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PC_EMULATOR_PREPARED_RUN="$run"
"$pc_python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib
import numpy
import scipy

run = Path(os.environ['PC_EMULATOR_PREPARED_RUN'])
python = (run / 'python_path.txt').read_text().strip()
if str(Path(sys.executable).absolute()) != python:
    raise RuntimeError('The running interpreter differs from the selected Python.')
epochs = int(os.environ.get('PC_EMULATOR_EPOCHS', '100'))
if epochs < 1:
    raise ValueError('PC_EMULATOR_EPOCHS must be positive.')
arguments = [
    '--input', (run / 'input_root.txt').read_text().strip(),
    '--output', str(run / 'results'),
    '--workers', '4', '--model-seeds', '11', '23', '47', '83',
    '--epochs', str(epochs), '--samples-per-snapshot', '4096',
    '--max-snapshots-per-seed', '24', '--subcells', '4',
    '--angles', '768', '--stride', '4',
]
with Path(sys.executable).resolve().open('rb') as stream:
    python_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
manifest = {
    'schema': 1,
    'experiment': 'pc-field-emulator',
    'scope': 'Coarse coeval field emulator of existing fixed-gas PC checkpoints',
    'commit': (run / 'commit.txt').read_text().strip(),
    'run_directory': str(run),
    'input_root': (run / 'input_root.txt').read_text().strip(),
    'python': python,
    'python_binary': str(Path(sys.executable).resolve()),
    'python_binary_sha256': python_sha256,
    'versions': {
        'python': platform.python_version(),
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'matplotlib': matplotlib.__version__,
    },
    'requested_cpus': 32,
    'trial_workers': 4,
    'blas_threads_per_worker': 8,
    'arguments': arguments,
}
(run / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
tracked = sorted(path for path in (run / 'source').rglob('*') if path.is_file())
tracked.extend(run / name for name in (
    'commit.txt', 'input_root.txt', 'runtime_parent.txt', 'python_path.txt',
    'runtime.env', 'run_manifest.json', 'submit.pbs',
))
with (run / 'prepared.sha256').open('w') as stream:
    for path in tracked:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        stream.write(f'{digest}  {path.relative_to(run)}\n')
(run / 'PREPARED').touch()
print('Training inputs:', manifest['input_root'])
print('Python:', python)
print('Parallel trials: 4; BLAS threads per trial: 8')
PY

printf 'Run directory: %s\nOutput directory: %s/results\n' "$run" "$run"
if "$submit"; then
    qsub -q "${PC_QUEUE:-normal}" \
        -l "ncpus=32,mem=${PC_MEM:-64GB},walltime=${PC_WALLTIME:-02:00:00},jobfs=4GB" \
        -v "RUN_DIR=$run" "$run/submit.pbs"
else
    printf 'Submit: qsub -v RUN_DIR=%q %q\n' "$run" "$run/submit.pbs"
fi
