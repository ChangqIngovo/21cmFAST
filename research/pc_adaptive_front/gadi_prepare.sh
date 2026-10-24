#!/usr/bin/env bash
# Prepare a pinned, editable native build on a Gadi login node.
# Usage: bash research/pc_adaptive_front/gadi_prepare.sh [--submit]
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --submit ) ]]; then
    echo 'Usage: bash research/pc_adaptive_front/gadi_prepare.sh [--submit]' >&2
    exit 2
fi
pc_threads=${PC_THREADS:-4}
if [[ ! $pc_threads =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
    echo 'PC_THREADS must be an integer from 1 to 32.' >&2
    exit 2
fi
if [[ -n ${PC_PYTHON:-} && $PC_PYTHON != /* ]]; then
    echo 'PC_PYTHON must be an absolute path to a Python >=3.12 interpreter.' >&2
    exit 2
fi
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --show-toplevel)
rev=$(git -C "$repo" rev-parse "${PC_REV:-HEAD}^{commit}")
git -C "$repo" cat-file -e "$rev:research/pc_adaptive_front/gadi_compare.pbs"

# Preserve valid setuptools_scm package/cache versions in a tag-less fork.
release_tag=v4.3.1
release_commit=72d8a2d23702f1b69fbc85abd25d4592a7bcd3f5
if ! git -C "$repo" rev-parse --verify --quiet "refs/tags/$release_tag^{commit}" >/dev/null; then
    git -C "$repo" fetch --no-tags https://github.com/21cmfast/21cmFAST.git \
        "refs/tags/$release_tag:refs/tags/$release_tag"
fi
test "$(git -C "$repo" rev-parse "refs/tags/$release_tag^{commit}")" = "$release_commit"
git -C "$repo" merge-base --is-ancestor "$release_commit" "$rev"

run_root=${PC_RUN_ROOT:-/g/data/qp00/${USER:?}/21cmfast_runs}
mkdir -p "$run_root"
run_root=$(cd "$run_root" && pwd -P)
run_dir=$(mktemp -d "$run_root/adaptive_${rev:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
exec > >(tee "$run_dir/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run_dir/FAILED_PREPARE"; fi' EXIT
printf 'Preparing: %s\nCommit: %s\n' "$run_dir" "$rev"
git -C "$repo" worktree add --detach "$run_dir/source" "$rev"
printf '%s\n' "$rev" > "$run_dir/commit.txt"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
module --force purge
pc_modules=(gcc/12.2.0 fftw3/3.3.10-nci1 gsl/2.7.1)
module load "${pc_modules[@]}"
if [[ -n ${PC_PYTHON_MODULE:-} ]]; then
    module load "$PC_PYTHON_MODULE"
    pc_modules+=("$PC_PYTHON_MODULE")
elif [[ -z ${PC_PYTHON:-} ]]; then
    mapfile -t python_modules < <(
        module -t avail python3 2>&1 |
        sed -nE 's@^[[:space:]]*(python3/[0-9]+\.[0-9]+\.[0-9]+)(\(.*\))?[[:space:]]*$@\1@p' |
        sort -V
    )
    python_module=''
    for candidate in "${python_modules[@]}"; do
        candidate_version=${candidate#python3/}
        candidate_major=${candidate_version%%.*}
        candidate_minor=${candidate_version#*.}
        candidate_minor=${candidate_minor%%.*}
        if (( candidate_major > 3 || ( candidate_major == 3 && candidate_minor >= 12 ) )); then
            python_module=$candidate
            break
        fi
    done
    if [[ -z $python_module ]]; then
        echo 'No Python >=3.12 module found; supply PC_PYTHON_MODULE or an absolute PC_PYTHON.' >&2
        exit 2
    fi
    module load "$python_module"
    pc_modules+=("$python_module")
fi
pc_python=${PC_PYTHON:-$(command -v python3)}
"$pc_python" -c 'import sys; assert sys.version_info >= (3,12), "Python >=3.12 is required"'
printf 'PC_MODULES=(' > "$run_dir/runtime.env"
printf ' %q' "${pc_modules[@]}" >> "$run_dir/runtime.env"
printf ' )\n' >> "$run_dir/runtime.env"
module -t list > "$run_dir/modules.txt" 2>&1

# Download/build here; compute nodes never run pip install.
export CC=gcc CXX=g++
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2
if [[ -f /apps/fftw3/3.3.10-nci1/include/fftw3.h ]]; then
    export PC_FFTW_INCLUDE=/apps/fftw3/3.3.10-nci1/include
fi
if [[ -d /apps/fftw3/3.3.10-nci1/lib ]]; then
    export PC_FFTW_LIBRARY=/apps/fftw3/3.3.10-nci1/lib
fi
if [[ -f /apps/gsl/2.7.1/include/gsl/gsl_rng.h ]]; then
    export PC_GSL_INCLUDE=/apps/gsl/2.7.1/include
fi
if [[ -d /apps/gsl/2.7.1/lib ]]; then
    export PC_GSL_LIBRARY=/apps/gsl/2.7.1/lib
fi
"$pc_python" -m venv "$run_dir/venv"
"$run_dir/venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$run_dir/venv/bin/python" -m pip install -e "$run_dir/source" 'pytest>=9'
"$run_dir/venv/bin/python" -m pip freeze > "$run_dir/python_packages.txt"
cp "$run_dir/source/research/pc_adaptive_front/gadi_compare.pbs" "$run_dir/submit.pbs"

export PC_PREPARED_RUN="$run_dir"
"$run_dir/venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import matplotlib
import numpy
import py21cmfast
import pytest

run = Path(os.environ['PC_PREPARED_RUN']).resolve()
source = run / 'source'
package = Path(py21cmfast.__file__).resolve().parent
if package != source / 'src/py21cmfast':
    raise RuntimeError(f'Expected editable package in prepared checkout, got {package}')
if int(numpy.__version__.split('.')[0]) < 2:
    raise RuntimeError('NumPy >=2 is required')
binaries = sorted(package.glob('*.so'))
if not binaries:
    raise RuntimeError('Editable installation did not produce a native extension')
commit = (run / 'commit.txt').read_text().strip()
subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', commit, '--'], check=True)
files = [run / name for name in ('runtime.env', 'python_packages.txt', 'modules.txt', 'submit.pbs', 'commit.txt')]
files += binaries
version_file = package / '_version.py'
if version_file.exists():
    files.append(version_file)
digests = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest = {
    'schema': 1, 'commit': commit, 'python': str(run / 'venv/bin/python'),
    'python_version': platform.python_version(), 'py21cmfast_version': py21cmfast.__version__,
    'numpy_version': numpy.__version__, 'matplotlib_version': matplotlib.__version__,
    'pytest_version': pytest.__version__, 'source_checkout': str(source),
    'files_sha256': digests,
    'scope': 'Adaptive/full-eight-subcell/balanced comparison on identical restricted native fields; no full coupled Qin20 evolution',
    'fixtures': {'grids': [32, 64], 'box_cMpc': 200.0, 'redshift': 8.0, 'seed': 1234},
    'fixture_note': 'The newly generated L-INTEGRAL 32^3 fixture differs from the earlier published E-INTEGRAL 32^3 fixture.',
}
(run / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
digests['run_manifest.json'] = hashlib.sha256((run / 'run_manifest.json').read_bytes()).hexdigest()
(run / 'prepared.sha256').write_text(''.join(f'{digest}  {name}\n' for name, digest in digests.items()))
print('Python:', sys.executable, platform.python_version())
print('21cmFAST:', py21cmfast.__version__, package)
PY
touch "$run_dir/PREPARED"
printf '\nRun directory: %s\nResults: %s/results\n' "$run_dir" "$run_dir"
pc_memory=${PC_MEM:-16GB}
pc_walltime=${PC_WALLTIME:-04:00:00}
pc_queue=${PC_QUEUE:-normal}
if [[ ${1:-} == --submit ]]; then
    job=$(qsub -P qp00 -q "$pc_queue" \
        -l "ncpus=$pc_threads,mem=$pc_memory,walltime=$pc_walltime" \
        -v "RUN_DIR=$run_dir" "$run_dir/submit.pbs")
    printf '%s\n' "$job" > "$run_dir/jobid.txt"
    printf 'Submitted: %s\n' "$job"
else
    printf 'Submit: qsub -P qp00 -q %q -l %q -v RUN_DIR=%q %q\n' \
        "$pc_queue" "ncpus=$pc_threads,mem=$pc_memory,walltime=$pc_walltime" \
        "$run_dir" "$run_dir/submit.pbs"
fi
