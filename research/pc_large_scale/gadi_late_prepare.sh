#!/usr/bin/env bash
# Reuse a completed large-scale run; archive only the new diagnostic code.
set -euo pipefail
submit=false
parent=${PC_PARENT_RUN:-}
for argument in "$@"; do
    case "$argument" in
        --submit) submit=true ;;
        --help) echo 'Usage: bash research/pc_large_scale/gadi_late_prepare.sh [PARENT_RUN] [--submit]'; exit 0 ;;
        *) if [[ -n $parent ]]; then echo 'Specify only one parent run.' >&2; exit 2; fi; parent=$argument ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --exit-code HEAD -- research/pc_large_scale research/pc_adaptive_front research/pc_refinement research/pc_reconsidered research/pc_alternatives
if [[ -z $parent ]]; then
    parent=$(python3 - <<'PY'
import json
import os
from pathlib import Path
base = Path(os.environ.get('PC_PARENT_ROOT', '/g/data/qp00/sz4640/21cmfast_runs'))
valid = []
for candidate in base.glob('large_pc_*'):
    if not (candidate/'COMPLETED').is_file():
        continue
    try:
        manifest = json.loads((candidate/'run_manifest.json').read_text())
        if manifest.get('experiment') == 'large-scale':
            valid.append(candidate)
    except (OSError, ValueError):
        continue
if not valid:
    raise SystemExit('No completed large-scale parent found; supply PARENT_RUN explicitly.')
print(max(valid, key=lambda p: (p/'COMPLETED').stat().st_mtime).resolve())
PY
)
fi
parent=$(realpath "$parent")
test -f "$parent/COMPLETED"
test -x "$parent/venv/bin/python"
revision=$(git -C "$repo" rev-parse HEAD)
root=${PC_LATE_RUN_ROOT:-$(dirname "$parent")}
mkdir -p "$root"
run=$(mktemp -d "$root/late_pc_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
mkdir "$run/source"
git -C "$repo" archive "$revision" research/pc_large_scale research/pc_adaptive_front research/pc_refinement research/pc_reconsidered research/pc_alternatives | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
cp "$parent/runtime.env" "$run/runtime.env"
cp "$run/source/research/pc_large_scale/gadi_late.pbs" "$run/submit.pbs"
module --force purge
source "$run/runtime.env"
module load "${PC_MODULES[@]}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PC_LATE_PREPARED_RUN="$run"
"$parent/venv/bin/python" "$run/source/research/pc_large_scale/late_followup.py" prepare --root "$run/results" --parent "$parent"
"$parent/venv/bin/python" - <<'PY'
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
run = Path(os.environ['PC_LATE_PREPARED_RUN'])
parent = Path((run/'parent_run.txt').read_text().strip())
python = parent/'venv/bin/python'
manifest = dict(schema=1, experiment='late-pc-followup', parent=str(parent), python=str(python),
                python_version=platform.python_version(), numpy_version=numpy.__version__,
                scipy_version=scipy.__version__, pytest_version=pytest.__version__,
                compiler=str(Path(shutil.which('g++')).resolve()),
                compiler_version=subprocess.check_output(['g++', '-dumpfullversion', '-dumpversion'], text=True).strip(),
                python_binary=str(Path(sys.executable).resolve()),
                python_binary_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest())
manifest['compiler_sha256'] = hashlib.sha256(Path(manifest['compiler']).read_bytes()).hexdigest()
(run/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(run/'python_packages.txt').write_text(subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True))
files = ['commit.txt', 'parent_run.txt', 'runtime.env', 'submit.pbs', 'run_manifest.json',
         'python_packages.txt', 'results/experiment.json']
(run/'prepared.sha256').write_text(''.join(hashlib.sha256((run/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in files))
PY
touch "$run/PREPARED"
printf '\nParent: %s\nRun directory: %s\nResults: %s/results\n' "$parent" "$run" "$run"
if $submit; then
    qsub -q "${PC_QUEUE:-hugemem}" -l "ncpus=${PC_THREADS:-32},mem=${PC_MEM:-320GB},walltime=${PC_WALLTIME:-06:00:00},jobfs=4GB" -v "RUN_DIR=$run" "$run/submit.pbs" | tee "$run/jobid.txt"
else
    printf 'Submit: qsub -v RUN_DIR=%s %s/submit.pbs\n' "$run" "$run"
fi
