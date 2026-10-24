#!/usr/bin/env bash
# Continue a completed refined-front experiment with exact angular multipliers.
set -euo pipefail
submit=false
parent=${PC_ANGULAR_PARENT_RUN:-}
for argument in "$@"; do
    case "$argument" in
        --submit) submit=true ;;
        --help) echo 'Usage: bash research/pc_refined_front/gadi_angular_prepare.sh [REFINED_PARENT_RUN] [--submit]'; exit 0 ;;
        *) if [[ -n $parent ]]; then echo 'Specify only one refined parent run.' >&2; exit 2; fi; parent=$argument ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/pc_refined_front/angular_followup.py
git -C "$repo" cat-file -e HEAD:research/pc_refined_front/gadi_angular_compare.pbs
git -C "$repo" diff --exit-code HEAD -- research/pc_large_scale research/pc_adaptive_front research/pc_refinement research/pc_reconsidered research/pc_alternatives research/pc_scale_origin research/pc_refined_front
if [[ -z $parent ]]; then
    parent=$(python3 - <<'PY'
import json
import os
from pathlib import Path
base = Path(os.environ.get('PC_PARENT_ROOT', '/g/data/qp00/sz4640/21cmfast_runs'))
valid = []
for candidate in base.glob('refined_pc_*'):
    if not (candidate/'COMPLETED').is_file() or not (candidate/'results/COMPLETED').is_file():
        continue
    try:
        manifest = json.loads((candidate/'run_manifest.json').read_text())
        if manifest.get('experiment') == 'refined-front-followup':
            valid.append(candidate)
    except (OSError, ValueError):
        continue
if not valid:
    raise SystemExit('No completed refined-front parent found; supply REFINED_PARENT_RUN explicitly.')
print(max(valid, key=lambda p: (p/'COMPLETED').stat().st_mtime).resolve())
PY
)
fi
parent=$(realpath "$parent")
test -f "$parent/COMPLETED"
test -f "$parent/results/COMPLETED"
pc_python=$(python3 - "$parent" <<'PY'
import json
from pathlib import Path
import sys
parent = Path(sys.argv[1])
manifest = json.loads((parent/'run_manifest.json').read_text())
if manifest.get('experiment') != 'refined-front-followup':
    raise SystemExit('The parent must be a completed refined-front-followup run.')
summary = json.loads((parent/'results/summary.json').read_text())
if not summary.get('all_ledgers_passed') or not summary.get('all_f2_implementation_checks_passed'):
    raise SystemExit('Parent numerical or implementation checks did not pass.')
python = Path(manifest['python'])
if not python.is_absolute():
    raise SystemExit('Parent Python executable must be an absolute path.')
print(python)
PY
)
test -x "$pc_python"
(cd "$parent" && sha256sum -c prepared.sha256)
pc_threads=${PC_THREADS:-32}
if [[ ! $pc_threads =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
    echo 'PC_THREADS must be an integer from 1 to 32.' >&2
    exit 2
fi
revision=$(git -C "$repo" rev-parse HEAD)
root=${PC_ANGULAR_RUN_ROOT:-$(dirname "$parent")}
mkdir -p "$root"
run=$(mktemp -d "$root/angular_pc_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
mkdir "$run/source"
git -C "$repo" archive "$revision" research/pc_large_scale research/pc_adaptive_front research/pc_refinement research/pc_reconsidered research/pc_alternatives research/pc_scale_origin research/pc_refined_front | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$parent/runtime.env" "$run/runtime.env"
cp "$run/source/research/pc_refined_front/gadi_angular_compare.pbs" "$run/submit.pbs"
module --force purge
source "$run/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PC_ANGULAR_PREPARED_RUN="$run"
read -r -a pc_grids <<< "${PC_GRIDS:-25 50}"
read -r -a pc_refinements <<< "${PC_REFINEMENTS:-2 4}"
read -r -a pc_doses <<< "${PC_DOSES:-0.8 0.9}"
read -r -a pc_multipliers <<< "${PC_ANGULAR_MULTIPLIERS:-1 2 4 8 16}"
"$pc_python" "$run/source/research/pc_refined_front/angular_followup.py" prepare \
    --root "$run/results" --parent "$parent" --grids "${pc_grids[@]}" \
    --refinements "${pc_refinements[@]}" --doses "${pc_doses[@]}" \
    --multipliers "${pc_multipliers[@]}" --contiguous-ray-gb "${PC_CONTIGUOUS_RAY_GB:-4}" \
    --power-tolerance "${PC_ANGULAR_POWER_TOLERANCE:-0.01}" \
    --field-tolerance "${PC_ANGULAR_FIELD_TOLERANCE:-0.01}"
"$pc_python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy
import scipy
import pytest
run = Path(os.environ['PC_ANGULAR_PREPARED_RUN'])
parent = Path((run/'parent_run.txt').read_text().strip())
python = (run/'python_path.txt').read_text().strip()
previous = json.loads((parent/'run_manifest.json').read_text())
compiler = shutil.which('g++')
if compiler is None:
    raise RuntimeError('g++ is unavailable after loading the parent runtime modules')
manifest = dict(schema=1, experiment='angular-front-followup', parent=str(parent), python=python,
                native_parent=previous['parent'],
                python_version=platform.python_version(), numpy_version=numpy.__version__,
                scipy_version=scipy.__version__, pytest_version=pytest.__version__,
                compiler=str(Path(compiler).resolve()),
                compiler_version=subprocess.check_output(['g++', '-dumpfullversion', '-dumpversion'], text=True).strip(),
                python_binary=str(Path(sys.executable).resolve()),
                python_binary_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest())
manifest['compiler_sha256'] = hashlib.sha256(Path(manifest['compiler']).read_bytes()).hexdigest()
if str(Path(sys.executable).absolute()) != python or python != previous['python']:
    raise RuntimeError('The interpreter differs from the completed refined parent')
for key in ('python_version', 'numpy_version', 'scipy_version', 'pytest_version',
            'compiler', 'compiler_version', 'compiler_sha256', 'python_binary', 'python_binary_sha256'):
    if manifest[key] != previous[key]:
        raise RuntimeError('Parent runtime changed: '+key)
packages = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
if packages != (parent/'python_packages.txt').read_text():
    raise RuntimeError('Parent Python package set changed')
manifest['parent_files_sha256'] = {str(parent/name): hashlib.sha256((parent/name).read_bytes()).hexdigest()
    for name in ('run_manifest.json', 'prepared.sha256', 'runtime.env', 'python_packages.txt',
                 'results/experiment.json', 'results/summary.json')}
(run/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(run/'python_packages.txt').write_text(packages)
files = ['commit.txt', 'parent_run.txt', 'python_path.txt', 'runtime.env', 'submit.pbs',
         'run_manifest.json', 'python_packages.txt', 'results/experiment.json']
(run/'prepared.sha256').write_text(''.join(hashlib.sha256((run/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in files))
PY
touch "$run/PREPARED"
printf '\nRefined parent: %s\nRun directory: %s\nResults: %s/results\n' "$parent" "$run" "$run"
if $submit; then
    qsub -q "${PC_QUEUE:-hugemem}" -l "ncpus=$pc_threads,mem=${PC_MEM:-320GB},walltime=${PC_WALLTIME:-12:00:00},jobfs=4GB" \
        -v "RUN_DIR=$run" "$run/submit.pbs" | tee "$run/jobid.txt"
else
    printf 'Submit: qsub -q %s -l ncpus=%s,mem=%s,walltime=%s,jobfs=4GB -v RUN_DIR=%s %s/submit.pbs\n' \
        "${PC_QUEUE:-hugemem}" "$pc_threads" "${PC_MEM:-320GB}" "${PC_WALLTIME:-12:00:00}" "$run" "$run"
fi
