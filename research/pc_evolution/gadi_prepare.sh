#!/usr/bin/env bash
# Prepare a pinned, prescribed-source temporal validation; do not run native ICs here.
set -euo pipefail
submit=false
parent=${PC_RUNTIME_PARENT:-}
for argument in "$@"; do
    case "$argument" in
        --submit) submit=true ;;
        --help)
            echo 'Usage: bash research/pc_evolution/gadi_prepare.sh [RUNTIME_PARENT_RUN] [--submit]'
            echo 'Environment: PC_PYTHON, PC_THREADS, PC_MEM, PC_MEMORY_GB, PC_WALLTIME, PC_QUEUE, PC_EVOLUTION_RUN_ROOT,'
            echo 'PC_DIMENSION, PC_BOX_LEN, PC_SEEDS, PC_Z_START, PC_Z_END, PC_BASE_INTERVALS, PC_REFERENCE_Z, PC_ANGLES.'
            echo 'PC_MEMORY_GB is the total history-worker budget in GiB; default is 80% of PC_MEM.'
            exit 0 ;;
        *)
            if [[ -n $parent ]]; then echo 'Specify only one runtime parent.' >&2; exit 2; fi
            parent=$argument ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/pc_evolution/workflow.py
git -C "$repo" cat-file -e HEAD:research/pc_evolution/gadi_compare.pbs
# The archive uses HEAD, so reject tracked edits rather than silently ignoring them.
git -C "$repo" diff --exit-code HEAD --
if [[ -z $parent ]]; then
    parent=$(python3 - <<'PY'
import json
import os
from pathlib import Path
base = Path(os.environ.get('PC_PARENT_ROOT', '/g/data/qp00/sz4640/21cmfast_runs'))
valid = []
for pattern in ('angular_pc_*', 'refined_pc_*', 'large_pc_*', 'evolution_pc_*'):
    for candidate in base.glob(pattern):
        if not (candidate/'COMPLETED').is_file() or not (candidate/'runtime.env').is_file():
            continue
        try:
            manifest = json.loads((candidate/'run_manifest.json').read_text())
            if manifest.get('experiment') in ('angular-front-followup', 'refined-front-followup',
                                               'large-scale', 'prescribed-source-evolution'):
                valid.append(candidate)
        except (OSError, ValueError):
            continue
if not valid:
    raise SystemExit('No completed compatible runtime found; supply PC_RUNTIME_PARENT or a parent path.')
print(max(valid, key=lambda p: (p/'COMPLETED').stat().st_mtime).resolve())
PY
)
fi
parent=$(realpath "$parent")
test -f "$parent/COMPLETED"
test -f "$parent/runtime.env"
test -f "$parent/prepared.sha256"
(cd "$parent" && sha256sum --quiet -c prepared.sha256)
pc_python=$(python3 - "$parent" <<'PY'
import json
import os
from pathlib import Path
import sys
parent = Path(sys.argv[1])
manifest = json.loads((parent/'run_manifest.json').read_text())
python = Path(os.environ.get('PC_PYTHON') or manifest.get('python') or parent/'venv/bin/python')
if not python.is_absolute():
    raise SystemExit('PC_PYTHON must be an absolute executable path.')
print(python)
PY
)
test -x "$pc_python"
pc_threads=${PC_THREADS:-32}
if [[ ! $pc_threads =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
    echo 'PC_THREADS must be an integer from 1 to 32.' >&2
    exit 2
fi
revision=$(git -C "$repo" rev-parse HEAD)
root=${PC_EVOLUTION_RUN_ROOT:-$(dirname "$parent")}
mkdir -p "$root"
run=$(mktemp -d "$root/evolution_pc_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
mkdir "$run/source"
# Include the native source/constants used to verify the installed native runtime.
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/runtime_parent.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$parent/runtime.env" "$run/runtime.env"
cp "$run/source/research/pc_evolution/gadi_compare.pbs" "$run/submit.pbs"
module --force purge
source "$run/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PC_EVOLUTION_PREPARED_RUN="$run" PC_EVOLUTION_PREPARED_THREADS="$pc_threads"
"$pc_python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import h5py
import matplotlib
import numpy
import py21cmfast
import pytest
import scipy

run = Path(os.environ['PC_EVOLUTION_PREPARED_RUN'])
parent = Path((run/'runtime_parent.txt').read_text().strip())
python = (run/'python_path.txt').read_text().strip()
if str(Path(sys.executable).absolute()) != python:
    raise RuntimeError('The running interpreter is not the selected Python executable.')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

compiler = shutil.which('g++')
if compiler is None:
    raise RuntimeError('g++ is unavailable after loading runtime modules.')
native = Path(py21cmfast.__file__).resolve().parent
archived_native = run/'source/src/py21cmfast'
if not archived_native.is_dir():
    raise RuntimeError('Archived native sources are missing.')
native_source_hashes = {}
for archived in sorted(archived_native.rglob('*')):
    if not archived.is_file():
        continue
    relative = archived.relative_to(archived_native)
    installed = native/relative
    expected = digest(archived)
    if not installed.is_file() or digest(installed) != expected:
        raise RuntimeError('Installed native source differs from the selected commit: '+str(relative))
    native_source_hashes[str(relative)] = expected
# Extra generated C/version/extension files need not exist in git, but their
# exact installed contents are pinned and checked again on the compute node.
native_footprint = {str(path.relative_to(native)): digest(path)
    for path in sorted(native.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}
if not any(name.endswith(('.so', '.pyd', '.dylib')) for name in native_footprint):
    raise RuntimeError('No compiled native extension found in the selected package.')

def integers(name, default):
    values = [int(value) for value in os.environ.get(name, default).split()]
    if not values or len(values) != len(set(values)):
        raise ValueError(name+' must contain a nonempty unique integer list.')
    return values

allocation = os.environ.get('PC_MEM', '320GB').strip()
match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(B|KB|KIB|MB|MIB|GB|GIB|TB|TIB)', allocation, re.I)
if match is None:
    raise ValueError('PC_MEM must be a positive number with a byte unit, e.g. 320GB or 64GiB.')
unit_bytes = {'B': 1, 'KB': 10**3, 'KIB': 2**10, 'MB': 10**6, 'MIB': 2**20,
              'GB': 10**9, 'GIB': 2**30, 'TB': 10**12, 'TIB': 2**40}
allocation_gib = float(match[1])*unit_bytes[match[2].upper()]/2**30
memory_gib = float(os.environ.get('PC_MEMORY_GB', str(.8*allocation_gib)))
if not (numpy.isfinite([allocation_gib, memory_gib]).all()
        and allocation_gib > 0 and 0 < memory_gib <= allocation_gib):
    raise ValueError('PC_MEMORY_GB must be finite, positive, and no greater than PC_MEM converted to GiB.')

config = dict(
    dimension=int(os.environ.get('PC_DIMENSION', '32')),
    box_len=float(os.environ.get('PC_BOX_LEN', '200')),
    seeds=integers('PC_SEEDS', '1234 2345 3456 4567'),
    z_start=float(os.environ.get('PC_Z_START', '25')),
    z_end=float(os.environ.get('PC_Z_END', '5')),
    base_intervals=int(os.environ.get('PC_BASE_INTERVALS', '80')),
    reference_z=float(os.environ.get('PC_REFERENCE_Z', '8')),
    angles=integers('PC_ANGLES', '384 768'),
    memory_gb=memory_gib,
)
if config['dimension'] < 4 or config['base_intervals'] < 1:
    raise ValueError('Use dimension >= 4 and at least one base time interval.')
if not numpy.isfinite(config['box_len']) or config['box_len'] <= 0:
    raise ValueError('BOX_LEN must be a positive finite length in cMpc.')
if not (numpy.isfinite([config['z_start'], config['z_end'], config['reference_z']]).all()
        and config['z_start'] > config['z_end'] >= 0
        and config['z_end'] <= config['reference_z'] <= config['z_start']):
    raise ValueError('Require z_start > z_end >= 0 and reference_z in that interval.')
if any(seed < 0 or seed > 2**32-1 for seed in config['seeds']):
    raise ValueError('Native seeds must be unsigned 32-bit integers.')
if any(value < 24 or value % 24 for value in config['angles']):
    raise ValueError('Angles must be positive multiples of 24.')

manifest = dict(
    schema=1, experiment='prescribed-source-evolution',
    commit=(run/'commit.txt').read_text().strip(), runtime_parent=str(parent),
    scope='Native redshift-dependent source rates and carried ionization in fixed gas; no density advection, recombination, thermal or UV-feedback evolution.',
    source_checkout=str(run/'source'), python=python,
    python_binary=str(Path(sys.executable).resolve()),
    python_binary_sha256=digest(Path(sys.executable).resolve()),
    python_version=platform.python_version(), numpy_version=numpy.__version__,
    scipy_version=scipy.__version__, h5py_version=h5py.__version__,
    matplotlib_version=matplotlib.__version__, pytest_version=pytest.__version__,
    py21cmfast_version=py21cmfast.__version__,
    compiler=str(Path(compiler).resolve()), compiler_sha256=digest(Path(compiler).resolve()),
    compiler_version=subprocess.check_output(['g++', '-dumpfullversion', '-dumpversion'], text=True).strip(),
    native_package=str(native), native_source_hashes=native_source_hashes,
    native_footprint_sha256=native_footprint,
    requested_threads=int(os.environ['PC_EVOLUTION_PREPARED_THREADS']),
    requested_memory=allocation, requested_memory_gib=allocation_gib,
    history_worker_memory_gib=memory_gib,
    memory_budget_note='Worker planner budget in GiB; default 80% of requested allocation. GB/MB use decimal bytes, GiB/MiB binary bytes. This is a planning allowance, not measured RSS.',
    execution_note='Native fixtures generated sequentially with allocated native threads; independent serial temporal histories use up to that many worker processes.',
)
(run/'experiment_config.json').write_text(json.dumps(config, indent=2)+'\n')
(run/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(run/'python_packages.txt').write_text(subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True))
files = [run/name for name in ('commit.txt', 'runtime_parent.txt', 'python_path.txt',
         'runtime.env', 'submit.pbs', 'run_manifest.json', 'python_packages.txt', 'experiment_config.json')]
files += sorted(path for path in (run/'source').rglob('*') if path.is_file())
(run/'prepared.sha256').write_text(''.join(digest(path)+'  '+str(path.relative_to(run))+'\n' for path in files))
print('Scope:', manifest['scope'])
print('Native package:', native, py21cmfast.__version__)
print('Native archived source files matched:', len(native_source_hashes))
print('Native package files pinned:', len(native_footprint))
print('Configuration:', json.dumps(config, sort_keys=True))
PY
touch "$run/PREPARED"
printf '\nRuntime parent: %s\nRun directory: %s\nResults: %s/results\n' "$parent" "$run" "$run"
if $submit; then
    qsub -q "${PC_QUEUE:-hugemem}" \
        -l "ncpus=$pc_threads,mem=${PC_MEM:-320GB},walltime=${PC_WALLTIME:-12:00:00},jobfs=4GB" \
        -v "RUN_DIR=$run" "$run/submit.pbs" | tee "$run/jobid.txt"
else
    printf 'Submit: qsub -q %s -l ncpus=%s,mem=%s,walltime=%s,jobfs=4GB -v RUN_DIR=%s %s/submit.pbs\n' \
        "${PC_QUEUE:-hugemem}" "$pc_threads" "${PC_MEM:-320GB}" "${PC_WALLTIME:-12:00:00}" "$run" "$run"
fi
