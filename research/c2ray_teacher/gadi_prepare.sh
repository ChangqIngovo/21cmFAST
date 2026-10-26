#!/usr/bin/env bash
# Freeze the 21cmFAST adapter, untouched C2-Ray source and Gadi runtime.
set -euo pipefail
submit=false
for argument in "$@"; do
    case "$argument" in
        --submit) submit=true ;;
        --help)
            echo 'Usage: bash research/c2ray_teacher/gadi_prepare.sh [--submit]'
            echo 'Environment: C2_RUNTIME_PARENT, C2_PYTHON, C2_UPSTREAM_REPO, C2_CACHE_ROOT, C2_RUN_ROOT,'
            echo 'C2_WORKERS, C2_QUEUE, C2_MEM, C2_WALLTIME, C2_DIMENSION, C2_HIGHRES_DIMENSION,'
            echo 'C2_BOX_LEN, C2_SEED, C2_Z_START, C2_Z_END, C2_INTERVALS, C2_REFERENCE_Z.'
            exit 0 ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/c2ray_teacher/runner.py
git -C "$repo" cat-file -e HEAD:research/c2ray_teacher/gadi_run.pbs
git -C "$repo" diff --exit-code HEAD --
revision=$(git -C "$repo" rev-parse HEAD)
upstream_revision=a363656ff1c3b4b4c7334f398578d8c469154515
parent=${C2_RUNTIME_PARENT:-/g/data/qp00/sz4640/21cmfast_runs/large_pc_5e40d8ab21_20261024T042855_XhUc}
parent=$(realpath "$parent")
test -f "$parent/runtime.env"
c2_python=${C2_PYTHON:-$parent/venv/bin/python}
test -x "$c2_python"
case "$c2_python" in /*) ;; *) echo 'C2_PYTHON must be absolute.' >&2; exit 2 ;; esac
workers=${C2_WORKERS:-32}
if [[ ! $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
    echo 'C2_WORKERS must be an integer from 1 through 32.' >&2
    exit 2
fi

module --force purge
source "$parent/runtime.env"
module load "${PC_MODULES[@]}"
module load gcc/12.2.0 openmpi/4.1.4
command -v gfortran >/dev/null
command -v mpifort >/dev/null
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

upstream=${C2_UPSTREAM_REPO:-}
if [[ -z $upstream ]]; then
    cache=${C2_CACHE_ROOT:-/g/data/qp00/sz4640/c2ray_cache}
    mkdir -p "$cache"
    upstream=$cache/C2-Ray3Dm1D_Helium.git
    if [[ ! -d $upstream ]]; then
        git clone --mirror https://github.com/garrelt/C2-Ray3Dm1D_Helium.git "$upstream"
    fi
    if ! git -C "$upstream" cat-file -e "$upstream_revision^{commit}"; then
        git -C "$upstream" fetch origin "$upstream_revision"
    fi
fi
git -C "$upstream" cat-file -e "$upstream_revision^{commit}"
upstream_tree=$(git -C "$upstream" rev-parse "$upstream_revision^{tree}")
root=${C2_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
mkdir -p "$root"
run=$(mktemp -d "$root/c2ray_teacher_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
mkdir "$run/source" "$run/upstream"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
git -C "$upstream" archive "$upstream_revision" | tar -x -C "$run/upstream"
cp "$parent/runtime.env" "$run/runtime.env"
cp "$run/source/research/c2ray_teacher/gadi_run.pbs" "$run/submit.pbs"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$upstream_revision" > "$run/upstream_commit.txt"
printf '%s\n' "$c2_python" > "$run/python_path.txt"
export C2_PREPARED_RUN="$run" C2_PREPARED_COMMIT="$revision"
export C2_PREPARED_UPSTREAM_REVISION="$upstream_revision" C2_PREPARED_UPSTREAM_TREE="$upstream_tree"
export C2_PREPARED_PYTHON="$c2_python" C2_PREPARED_WORKERS="$workers"
"$c2_python" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import h5py
import numpy
import py21cmfast
import scipy

run = Path(os.environ['C2_PREPARED_RUN'])
if any(character in str(run) for character in ',\n\r'):
    raise ValueError('RUN_DIR must not contain commas or newlines (PBS variable syntax).')

def integer(name, default):
    text = os.environ.get(name, str(default))
    if not re.fullmatch(r'[0-9]+', text):
        raise ValueError(name + ' must be a positive integer.')
    value = int(text)
    if value <= 0:
        raise ValueError(name + ' must be positive.')
    return value

def number(name, default):
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite.')
    return value

config = {
    'dimension': integer('C2_DIMENSION', 16),
    'highres_dimension': integer('C2_HIGHRES_DIMENSION', 48),
    'box_len': number('C2_BOX_LEN', 50),
    'seed': integer('C2_SEED', 1234),
    'z_start': number('C2_Z_START', 20),
    'z_end': number('C2_Z_END', 5),
    'intervals': integer('C2_INTERVALS', 30),
    'reference_z': number('C2_REFERENCE_Z', 8),
}
if (config['dimension'] < 16 or config['highres_dimension'] < config['dimension']
        or config['box_len'] <= 0):
    raise ValueError('dimension >= 16, highres_dimension >= dimension and positive box length required.')
if not config['z_start'] > config['z_end'] >= 0 or config['reference_z'] < 0:
    raise ValueError('Descending nonnegative redshifts and nonnegative reference_z required.')
queue = os.environ.get('C2_QUEUE', 'normal')
memory = os.environ.get('C2_MEM', '128GB')
walltime = os.environ.get('C2_WALLTIME', '24:00:00')
if not re.fullmatch(r'[a-z][a-z0-9_-]*', queue):
    raise ValueError('Invalid PBS queue name.')
if not re.fullmatch(r'[1-9][0-9]*(?:[KMGTP]B)', memory, re.I):
    raise ValueError('Invalid PBS memory request.')
if not re.fullmatch(r'[0-9]+:[0-5][0-9]:[0-5][0-9]', walltime):
    raise ValueError('Invalid PBS walltime.')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

native = Path(py21cmfast.__file__).resolve().parent
native_files = {
    str(path.relative_to(native)): digest(path)
    for path in sorted(native.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts
    and path.suffix not in ('.pyc', '.pyo')
}
if not any(name.endswith(('.so', '.pyd', '.dylib')) for name in native_files):
    raise RuntimeError('The selected runtime has no compiled 21cmFAST extension.')
compilers = {}
for compiler in ('gfortran', 'mpifort'):
    path = Path(shutil.which(compiler)).resolve()
    compilers[compiler] = {
        'path': str(path),
        'sha256': digest(path),
        'version': subprocess.check_output([compiler, '--version'], text=True),
    }
manifest = {
    'schema_version': 1,
    'experiment': 'c2ray-native-teacher',
    'commit': os.environ['C2_PREPARED_COMMIT'],
    'upstream_commit': os.environ['C2_PREPARED_UPSTREAM_REVISION'],
    'upstream_tree': os.environ['C2_PREPARED_UPSTREAM_TREE'],
    'upstream_url': 'https://github.com/garrelt/C2-Ray3Dm1D_Helium',
    'upstream_algorithms': 'unmodified; adapter and build configuration only',
    'run_directory': str(run),
    'python': os.environ['C2_PREPARED_PYTHON'],
    'runtime_python': sys.executable,
    'python_version': platform.python_version(),
    'package_versions': {
        'numpy': numpy.__version__, 'scipy': scipy.__version__,
        'h5py': h5py.__version__, 'py21cmfast': py21cmfast.__version__,
    },
    'native_package': str(native),
    'native_files_sha256': native_files,
    'compilers': compilers,
    'requested_workers': int(os.environ['C2_PREPARED_WORKERS']),
    'queue': queue, 'memory': memory, 'walltime': walltime,
    'scope': 'Prescribed 21cmFAST sources and frozen density contrast; C2-Ray RT teacher pilot.',
    'config': config,
}
(run/'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
(run/'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with (run/'prepared.sha256').open('w') as stream:
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name != 'prepared.sha256':
            stream.write(digest(path) + '  ' + str(path.relative_to(run)) + '\n')
PY
touch "$run/PREPARED"
printf 'Run directory: %s\nResults: %s/results\n' "$run" "$run"
if $submit; then
    qsub -q "${C2_QUEUE:-normal}" \
        -l "ncpus=$workers,mem=${C2_MEM:-128GB},walltime=${C2_WALLTIME:-24:00:00},jobfs=4GB" \
        -v "RUN_DIR=$run" "$run/submit.pbs" | tee "$run/job_id.txt"
fi
