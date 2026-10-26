#!/usr/bin/env bash
# Prepare training against existing PC checkpoints; no new simulation is run.
set -euo pipefail

submit=false
inputs=()
for argument in "$@"; do
    case "$argument" in
        --submit) submit=true ;;
        --help)
            echo 'Usage: bash research/pc_field_emulator/gadi_prepare.sh [INPUT_RUN_OR_RESULTS ...] [--submit]'
            echo 'Defaults: existing four-seed 200 cMpc run and 400 cMpc / 128^3 production checkpoints.'
            echo 'Resources: 32 CPUs, normal, 64GB, 2 hours; four fits with eight BLAS threads each.'
            echo 'Environment: PC_EMULATOR_INPUT, PC_RUNTIME_PARENT, PC_PYTHON, PC_EMULATOR_RUN_ROOT,'
            echo 'PC_EMULATOR_EPOCHS, PC_MEM, PC_WALLTIME, PC_QUEUE.'
            echo 'Explicit paths or PC_EMULATOR_INPUT replace both defaults.'
            echo 'Only checkpoints committed at preparation are used; no simulation is run.'
            echo 'Input runs are read-only. The time request is an allocation, not an ETA.'
            exit 0 ;;
        --*) echo "Unknown option: $argument" >&2; exit 2 ;;
        *) inputs+=("$argument") ;;
    esac
done
if (( ${#inputs[@]} == 0 )); then
    if [[ -n ${PC_EMULATOR_INPUT:-} ]]; then
        inputs=("$PC_EMULATOR_INPUT")
    else
        inputs=(
            "/g/data/qp00/sz4640/21cmfast_runs/evolution_pc_30e8497de2_20261024T122119_qC2V"
            "/g/data/qp00/sz4640/21cmfast_runs/evolution_production_ecd4bbdb88_20261024T144219_8iMG"
        )
    fi
fi
for index in "${!inputs[@]}"; do
    inputs[$index]=$(realpath "${inputs[$index]}")
    if [[ ! -d ${inputs[$index]} ]]; then
        echo "Input run is missing: ${inputs[$index]}" >&2
        exit 2
    fi
done

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/pc_field_emulator/train.py
git -C "$repo" cat-file -e HEAD:research/pc_field_emulator/gadi_train.pbs
git -C "$repo" diff --exit-code HEAD -- research/pc_field_emulator
if [[ -n $(git -C "$repo" ls-files --others --exclude-standard -- research/pc_field_emulator) ]]; then
    echo 'Commit new emulator files before preparing the archived job.' >&2
    exit 2
fi
parent=${PC_RUNTIME_PARENT:-${inputs[0]}}
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
python3 - "$run/input_roots.json" "${inputs[@]}" <<'PY'
import json
from pathlib import Path
import sys

roots = sys.argv[2:]
if len(set(roots)) != len(roots):
    raise ValueError('Input paths must be unique.')
Path(sys.argv[1]).write_text(json.dumps(roots, indent=2) + '\n')
PY
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
from dataclasses import asdict
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
epochs = int(os.environ.get('PC_EMULATOR_EPOCHS', '200'))
if epochs < 1:
    raise ValueError('PC_EMULATOR_EPOCHS must be positive.')
input_roots = json.loads((run / 'input_roots.json').read_text())
sys.path.insert(0, str(run / 'source/research/pc_field_emulator'))
from data import discover_samples, sample_key, validate_physics

samples = []
for root in input_roots:
    try:
        root_samples = discover_samples(Path(root), subcells=4, angles=768, stride=4)
    except (ValueError, FileNotFoundError) as error:
        raise RuntimeError(f'Cannot freeze requested input {root}: {error}') from error
    if not root_samples:
        raise RuntimeError(f'No committed PC snapshots in requested input: {root}')
    samples.extend(root_samples)
keys = [sample_key(sample) for sample in samples]
if len(set(keys)) != len(keys):
    raise ValueError('Duplicate physical realization/grid/node across input runs.')
physics = validate_physics(samples)

def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numpy.generic):
        return value.item()
    raise TypeError(f'Unsupported manifest value: {type(value)}')

inventory = {
    'schema': 1,
    'input_roots': input_roots,
    'teacher_arm': {'subcells': 4, 'angles': 768, 'stride': 4},
    'scope': 'Only durable checkpoints discovered at preparation; later outputs are excluded',
    'physics': physics,
    'samples': [asdict(sample) for sample in samples],
}
(run / 'sample_manifest.json').write_text(
    json.dumps(inventory, indent=2, default=json_value, allow_nan=False) + '\n'
)
arguments = [part for root in input_roots for part in ('--input', root)] + [
    '--sample-manifest', str(run / 'sample_manifest.json'),
    '--output', str(run / 'results'),
    '--workers', '4', '--model-seeds', '11', '23', '47', '83',
    '--epochs', str(epochs), '--samples-per-snapshot', '8192',
    '--max-snapshots-per-seed', '24', '--subcells', '4',
    '--angles', '768', '--stride', '4',
    '--model-mode', 'residual', '--radii-cmpc', '6.25', '12.5', '25', '50',
    '--front-weight', '4', '--neutral-weight', '2',
]
with Path(sys.executable).resolve().open('rb') as stream:
    python_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
manifest = {
    'schema': 2,
    'experiment': 'pc-field-emulator',
    'scope': 'Mixed-grid residual field emulator of existing fixed-gas PC checkpoints',
    'commit': (run / 'commit.txt').read_text().strip(),
    'run_directory': str(run),
    'input_roots': input_roots,
    'frozen_sample_manifest': str(run / 'sample_manifest.json'),
    'frozen_sample_count': len(samples),
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
    'commit.txt', 'input_roots.json', 'sample_manifest.json', 'runtime_parent.txt',
    'python_path.txt', 'runtime.env', 'run_manifest.json', 'submit.pbs',
))
with (run / 'prepared.sha256').open('w') as stream:
    for path in tracked:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        stream.write(f'{digest}  {path.relative_to(run)}\n')
(run / 'PREPARED').touch()
print('Training inputs:', manifest['input_roots'])
print('Frozen committed snapshots:', len(samples))
for box_len, dimension in sorted({(row.box_len, row.dimension) for row in samples}):
    group = [row for row in samples if row.box_len == box_len and row.dimension == dimension]
    print(f'  L={box_len:g} cMpc; N={dimension}; snapshots={len(group)}; '
          f'IC seeds={sorted({row.seed for row in group})}')
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
