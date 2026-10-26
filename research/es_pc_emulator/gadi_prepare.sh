#!/usr/bin/env bash
# Build and freeze an isolated native runtime, then prepare a paired dataset job.
set -euo pipefail
submit=false
config=
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --config)
            [[ $# -ge 2 ]] || { echo '--config needs a JSON path.' >&2; exit 2; }
            config=$(realpath "$2"); shift 2 ;;
        --help)
            echo 'Usage: bash research/es_pc_emulator/gadi_prepare.sh [--config FILE] [--submit]'
            echo 'ES_PC_WORKERS/MEM/WALLTIME/QUEUE override 32/128GB/48h/normal.'
            echo 'ES_PC_RUNTIME_PARENT, ES_PC_PYTHON, ES_PC_UPSTREAM_REPO, ES_PC_RUN_ROOT select existing resources.'
            echo 'Preparation builds a new native extension on the login node; existing runtimes stay unchanged.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/generate.py
git -C "$repo" cat-file -e HEAD:research/es_pc_emulator/gadi_dataset.pbs
git -C "$repo" diff --exit-code HEAD --
revision=$(git -C "$repo" rev-parse HEAD)
workers=${ES_PC_WORKERS:-32}
queue=${ES_PC_QUEUE:-normal}
memory=${ES_PC_MEM:-128GB}
walltime=${ES_PC_WALLTIME:-48:00:00}
[[ $workers =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo 'ES_PC_WORKERS must be 1 through 32.' >&2; exit 2; }
[[ $queue =~ ^[a-z][a-z0-9_-]*$ ]] || { echo 'Invalid queue.' >&2; exit 2; }
[[ $memory =~ ^[1-9][0-9]*[KMGT]B$ ]] || { echo 'Invalid memory.' >&2; exit 2; }
[[ $walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo 'Invalid walltime.' >&2; exit 2; }

parent=${ES_PC_RUNTIME_PARENT:-${C2_RUNTIME_PARENT:-/g/data/qp00/sz4640/21cmfast_runs/large_pc_5e40d8ab21_20261024T042855_XhUc}}
parent=$(realpath "$parent")
test -f "$parent/runtime.env"
base_python=${ES_PC_PYTHON:-$parent/venv/bin/python}
case "$base_python" in /*) ;; *) echo 'ES_PC_PYTHON must be absolute.' >&2; exit 2 ;; esac
test -x "$base_python"
module --force purge
source "$parent/runtime.env"
PC_MODULES+=(gcc/12.2.0 fftw3/3.3.10-nci1 gsl/2.7.1 openmpi/4.1.4)
module load "${PC_MODULES[@]}"
command -v gfortran >/dev/null
command -v mpifort >/dev/null
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$base_python" -c 'import sys; assert sys.version_info >= (3,12), "Python >=3.12 required"'

# A real worktree retains setuptools_scm release information for native caches.
release_tag=v4.3.1
release_commit=72d8a2d23702f1b69fbc85abd25d4592a7bcd3f5
if ! git -C "$repo" rev-parse --verify --quiet "refs/tags/$release_tag^{commit}" >/dev/null; then
    git -C "$repo" fetch --no-tags https://github.com/21cmfast/21cmFAST.git \
        "refs/tags/$release_tag:refs/tags/$release_tag"
fi
test "$(git -C "$repo" rev-parse "refs/tags/$release_tag^{commit}")" = "$release_commit"
git -C "$repo" merge-base --is-ancestor "$release_commit" "$revision"
root=${ES_PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
mkdir -p "$root"
run=$(mktemp -d "$root/es_pc_dataset_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" worktree add --detach "$run/source" "$revision"
printf '%s\n' "$revision" > "$run/commit.txt"
printf 'PC_MODULES=(' > "$run/runtime.env"
printf ' %q' "${PC_MODULES[@]}" >> "$run/runtime.env"
printf ' )\n' >> "$run/runtime.env"
module -t list > "$run/modules.txt" 2>&1

upstream_revision=a363656ff1c3b4b4c7334f398578d8c469154515
upstream=${ES_PC_UPSTREAM_REPO:-${C2_UPSTREAM_REPO:-}}
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
mkdir "$run/upstream"
git -C "$upstream" archive "$upstream_revision" | tar -x -C "$run/upstream"
printf '%s\n' "$upstream_revision" > "$run/upstream_commit.txt"

# Build only on the login node. No running teacher runtime is modified.
export CC=gcc CXX=g++ FC=gfortran MPIFC=mpifort
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2
if [[ -f /apps/fftw3/3.3.10-nci1/include/fftw3.h ]]; then export PC_FFTW_INCLUDE=/apps/fftw3/3.3.10-nci1/include; fi
if [[ -d /apps/fftw3/3.3.10-nci1/lib ]]; then export PC_FFTW_LIBRARY=/apps/fftw3/3.3.10-nci1/lib; fi
if [[ -f /apps/gsl/2.7.1/include/gsl/gsl_rng.h ]]; then export PC_GSL_INCLUDE=/apps/gsl/2.7.1/include; fi
if [[ -d /apps/gsl/2.7.1/lib ]]; then export PC_GSL_LIBRARY=/apps/gsl/2.7.1/lib; fi
"$base_python" -m venv "$run/venv"
python=$run/venv/bin/python
"$python" -m pip install --upgrade pip setuptools wheel
"$python" -m pip install -e "$run/source" 'pytest>=9'
"$python" -m pip freeze > "$run/python_packages.txt"
cp "$run/source/research/es_pc_emulator/gadi_dataset.pbs" "$run/submit.pbs"
if [[ -z $config ]]; then config=$run/source/research/es_pc_emulator/default_dataset.json; fi
cp "$config" "$run/requested_config.json"
export ES_PC_PREPARED_RUN="$run" ES_PC_PREPARED_WORKERS="$workers"
export ES_PC_PREPARED_QUEUE="$queue" ES_PC_PREPARED_MEMORY="$memory" ES_PC_PREPARED_WALLTIME="$walltime"
cd "$run/source"
"$python" - <<'PY'
import json
import os
from pathlib import Path
import subprocess

from research.es_pc_emulator.generate import digest, runtime_fingerprint, validate_config, write_json
from research.pc_evolution.native_history import require_es_radius_capability

run = Path(os.environ['ES_PC_PREPARED_RUN'])
if any(c in str(run) for c in ',\n\r'):
    raise ValueError('Prepared path must not contain PBS variable separators.')
require_es_radius_capability()
config = validate_config(json.loads((run/'requested_config.json').read_text()))
runtime = runtime_fingerprint()
if Path(runtime['native_package']) != run/'source/src/py21cmfast':
    raise RuntimeError('Native extension is not from the fresh source checkout.')
write_json(run/'dataset_config.json', config)
manifest = {
    'schema_version': 1, 'experiment': 'es-to-c2ray-training-dataset',
    'commit': (run/'commit.txt').read_text().strip(),
    'upstream_commit': (run/'upstream_commit.txt').read_text().strip(),
    'run_directory': str(run), 'source_checkout': str(run/'source'),
    'requested_workers': int(os.environ['ES_PC_PREPARED_WORKERS']),
    'queue': os.environ['ES_PC_PREPARED_QUEUE'], 'memory': os.environ['ES_PC_PREPARED_MEMORY'],
    'walltime': os.environ['ES_PC_PREPARED_WALLTIME'], 'runtime': runtime, 'config': config,
}
write_json(run/'run_manifest.json', manifest)
tracked = subprocess.check_output(['git','-C',str(run/'source'),'ls-files','-z']).split(b'\0')
paths = [run/'source'/p.decode() for p in tracked if p]
paths += [run/name for name in ('runtime.env','modules.txt','python_packages.txt','submit.pbs',
                              'requested_config.json','dataset_config.json','commit.txt',
                              'upstream_commit.txt','run_manifest.json')]
paths += [p for p in (run/'upstream').rglob('*') if p.is_file()]
with (run/'prepared.sha256').open('w') as stream:
    for path in sorted(paths):
        stream.write(digest(path)+'  '+str(path.relative_to(run))+'\n')
PY
touch "$run/PREPARED"
if [[ -n ${ES_PC_PREPARED_PATH_FILE:-} ]]; then
    (set -o noclobber; printf '%s\n' "$run" > "$ES_PC_PREPARED_PATH_FILE")
fi
printf '\nRun directory: %s\nDataset: %s/results/dataset_manifest.json\n' "$run" "$run"
if $submit; then
    job=$(qsub -P qp00 -q "$queue" -l "ncpus=$workers,mem=$memory,walltime=$walltime,jobfs=4GB" \
        -v "RUN_DIR=$run" "$run/submit.pbs")
    [[ $job =~ ^[0-9]+([.][A-Za-z0-9._-]+)?$ ]] || { echo "Unexpected job ID: $job" >&2; exit 1; }
    printf '%s\n' "$job" > "$run/job_id.txt"
    printf 'Submitted: %s\n' "$job"
else
    printf 'Submit: qsub -P qp00 -q %q -l %q -v RUN_DIR=%q %q\n' \
        "$queue" "ncpus=$workers,mem=$memory,walltime=$walltime,jobfs=4GB" "$run" "$run/submit.pbs"
fi
