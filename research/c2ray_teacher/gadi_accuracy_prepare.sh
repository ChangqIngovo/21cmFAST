#!/usr/bin/env bash
# Prepare all controlled cases before submitting a sequential PBS job chain.
set -euo pipefail
submit=false
stage=all
history=${C2_HISTORY:-/g/data/qp00/sz4640/21cmfast_runs/c2ray_teacher_3b07aff671_20261026T082902_ApQo/results/native_history.npz}
history_argument=false
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --stage)
            [[ $# -ge 2 ]] || { echo '--stage needs validation, expanded or all.' >&2; exit 2; }
            stage=$2; shift 2 ;;
        --history)
            [[ $# -ge 2 ]] || { echo '--history needs a native_history.npz path.' >&2; exit 2; }
            history=$2; history_argument=true; shift 2 ;;
        --help)
            echo 'Usage: bash research/c2ray_teacher/gadi_accuracy_prepare.sh [--stage validation|expanded|all] [--history FILE] [--submit]'
            echo 'Default all: four controlled time/RT-grid cases, then four native-grid/volume cases.'
            echo 'All cases are prepared first. Submission uses afterok dependencies, with at most one case running.'
            echo 'Validation histories must match L50/N16/DIM48, z20 to 5, 30 source intervals and density reference z8.'
            echo 'Allocation defaults: 32 CPUs, 128GB, 48h, normal. Override C2_WORKERS/C2_MEM/C2_WALLTIME/C2_QUEUE.'
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if $history_argument; then echo 'Only one history path is accepted.' >&2; exit 2; fi
            history=$1; history_argument=true; shift ;;
    esac
done
case "$stage" in validation|expanded|all) ;; *) echo 'Invalid --stage.' >&2; exit 2 ;; esac
if [[ $stage != expanded ]]; then
    history=$(realpath "$history")
    test -f "$history"
fi
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" cat-file -e HEAD:research/c2ray_teacher/gadi_accuracy_prepare.sh
git -C "$repo" diff --exit-code HEAD --
revision=$(git -C "$repo" rev-parse HEAD)
root=${C2_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
mkdir -p "$root"
suite=$(mktemp -d "$root/c2ray_accuracy_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
suite=$(realpath "$suite")
mkdir "$suite/receipts" "$suite/runs" "$suite/preparation_logs"
workers=${C2_WORKERS:-32}
queue=${C2_QUEUE:-normal}
memory=${C2_MEM:-128GB}
walltime=${C2_WALLTIME:-48:00:00}
export C2_SUITE_DIRECTORY="$suite" C2_SUITE_STAGE="$stage" C2_SUITE_HISTORY="$history"
export C2_SUITE_REVISION="$revision" C2_SUITE_WORKERS="$workers" C2_SUITE_QUEUE="$queue"
export C2_SUITE_MEMORY="$memory" C2_SUITE_WALLTIME="$walltime"
python3 - <<'PY'
import json
import os
from pathlib import Path

suite = Path(os.environ['C2_SUITE_DIRECTORY'])
history = os.environ['C2_SUITE_HISTORY']
if any(c in history for c in '\t\r\n'):
    raise ValueError('History path must not contain tabs or newlines.')
stage = os.environ['C2_SUITE_STAGE']
base = {
    'dimension': 16, 'highres_dimension': 48, 'box_len': 50.0,
    'seed': 1234, 'z_start': 20.0, 'z_end': 5.0, 'reference_z': 8.0,
    'intervals': 30, 'stride': 1, 'refinement': 1, 'time_substeps': 1,
}
cases = []
if stage in ('validation', 'all'):
    for name, refinement, substeps in (
        ('time1', 1, 1), ('time2', 1, 2), ('time4', 1, 4), ('space48_dt2', 3, 2),
    ):
        config = base | {'history': history, 'refinement': refinement, 'time_substeps': substeps}
        cases.append({'name': name, 'config': config, 'family': 'fixed_history'})
if stage in ('expanded', 'all'):
    for name, length, dimension, highres in (
        ('bridge_L50_N16_DIM128', 50.0, 16, 128),
        ('box_L75_N24_DIM192', 75.0, 24, 192),
        ('box_L100_N32_DIM256', 100.0, 32, 256),
        ('finer_L75_N32_DIM192', 75.0, 32, 192),
    ):
        config = base | {
            'box_len': length, 'dimension': dimension, 'highres_dimension': highres,
            'intervals': 60, 'time_substeps': 2,
        }
        cases.append({'name': name, 'config': config, 'family': 'new_native_history'})
for index, case in enumerate(cases):
    case['config']['benchmark'] = index == 0
    case.update(run_directory=None, job_id=None, dependency_job_id=None)
manifest = {
    'schema_version': 1, 'experiment': 'c2ray-accuracy-volume-suite',
    'commit': os.environ['C2_SUITE_REVISION'], 'stage': stage,
    'submission_policy': 'sequential_afterok',
    'requested_workers_per_case': int(os.environ['C2_SUITE_WORKERS']),
    'queue': os.environ['C2_SUITE_QUEUE'], 'memory': os.environ['C2_SUITE_MEMORY'],
    'walltime_per_case': os.environ['C2_SUITE_WALLTIME'],
    'cases': cases,
    'interpretation': [
        'time1/time2/time4 share exactly one frozen source and gas history.',
        'space48_dt2 subdivides the old gas grid while preserving original source centres; it adds no density modes.',
        'Expanded boxes have identical IC cell width 0.390625 cMpc; the first three also share RT cell width 3.125 cMpc.',
        'Different box sizes are different realizations even with the same seed; box comparisons are statistical.',
        'L75 N24/N32 share box, DIM and seed; verify recorded high-resolution IC hashes before matched-phase analysis.',
        'Expanded histories use 60 source intervals and 120 RT steps; they are not a time-only comparison to the old history.',
        'No convergence result is assumed by preparing this suite.',
    ],
}
(suite/'suite_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with (suite/'cases.tsv').open('w') as stream:
    for case in cases:
        c = case['config']
        values = [case['name'], c.get('history', '-'), c['dimension'], c['highres_dimension'],
                  c['box_len'], c['intervals'], c['refinement'], c['time_substeps'],
                  str(c['benchmark']).lower()]
        stream.write('\t'.join(map(str, values)) + '\n')
PY

while IFS=$'\t' read -r name source_history dimension highres box intervals refinement substeps benchmark; do
    if [[ $source_history == - ]]; then source_history=; fi
    printf 'Preparing: %s\n' "$name"
    C2_RUN_ROOT="$suite/runs" C2_HISTORY="$source_history" \
    C2_WORKERS="$workers" C2_QUEUE="$queue" C2_MEM="$memory" C2_WALLTIME="$walltime" \
    C2_DIMENSION="$dimension" C2_HIGHRES_DIMENSION="$highres" C2_BOX_LEN="$box" \
    C2_INTERVALS="$intervals" C2_REFINEMENT="$refinement" C2_TIME_SUBSTEPS="$substeps" \
    C2_SEED=1234 C2_Z_START=20 C2_Z_END=5 C2_REFERENCE_Z=8 C2_STRIDE=1 \
    C2_BENCHMARK="$benchmark" C2_PREPARED_PATH_FILE="$suite/receipts/$name.txt" \
        bash "$repo/research/c2ray_teacher/gadi_prepare.sh" \
        > "$suite/preparation_logs/$name.log" 2>&1 || {
            cat "$suite/preparation_logs/$name.log" >&2
            echo 'Preparation failed before suite submission.' >&2
            exit 1
        }
done < "$suite/cases.tsv"

python3 - <<'PY'
import json
import os
from pathlib import Path

suite = Path(os.environ['C2_SUITE_DIRECTORY'])
path = suite/'suite_manifest.json'
manifest = json.loads(path.read_text())
history_hashes = set()
for case in manifest['cases']:
    run = Path((suite/'receipts'/f"{case['name']}.txt").read_text().strip())
    if not (run/'PREPARED').is_file():
        raise RuntimeError('A case did not finish preparation: ' + case['name'])
    case['run_directory'] = str(run)
    case['config'] = json.loads((run/'run_config.json').read_text())
    if case['family'] == 'fixed_history':
        prepared = json.loads((run/'run_manifest.json').read_text())
        history_hashes.add(prepared['reused_history']['sha256'])
if len(history_hashes) > 1:
    raise RuntimeError('Validation histories changed between case preparations; no jobs were submitted.')
manifest['validation_history_sha256'] = next(iter(history_hashes), None)
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(manifest, indent=2) + '\n')
temporary.replace(path)
PY

if [[ $stage != expanded ]]; then
    # Use the selected prepared runtime to read only the small NPZ metadata.
    # Do not infer a potentially expensive RT grid from an arbitrary history.
    reference_run=$(cat "$suite/receipts/time1.txt")
    (
        module --force purge
        source "$reference_run/runtime.env"
        module load "${PC_MODULES[@]}"
        unset PYTHONPATH PYTHONHOME
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        c2_python=$(cat "$reference_run/python_path.txt")
        "$c2_python" - "$reference_run/input/native_history.npz" <<'PY'
import json
import os
from pathlib import Path
import sys

import numpy as np

expected = {
    'dimension': 16, 'highres_dimension': 48, 'box_len_cMpc': 50.0,
    'redshift_start': 20.0, 'redshift_end': 5.0, 'intervals': 30,
    'reference_redshift': 8.0,
}
with np.load(sys.argv[1], allow_pickle=False) as archive:
    raw = archive['metadata']
    if raw.shape != () or raw.dtype.kind not in 'US':
        raise ValueError('Validation history metadata must be a scalar JSON string.')
    metadata = json.loads(str(raw.item()))
for key, value in expected.items():
    if metadata.get(key) != value:
        raise ValueError(f'Accuracy validation history requires {key}={value}; got {metadata.get(key)!r}.')
path = Path(os.environ['C2_SUITE_DIRECTORY'])/'suite_manifest.json'
manifest = json.loads(path.read_text())
manifest['validation_history_geometry'] = expected
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(manifest, indent=2) + '\n')
temporary.replace(path)
PY
    )
fi
touch "$suite/PREPARED"
printf 'Suite directory: %s\nManifest: %s/suite_manifest.json\n' "$suite" "$suite"

if $submit; then
    previous=
    index=0
    while IFS=$'\t' read -r name rest; do
        run=$(cat "$suite/receipts/$name.txt")
        dependency=()
        if [[ -n $previous ]]; then dependency=(-W "depend=afterok:$previous"); fi
        job=$(qsub -N "c2_${index}_${name:0:8}" -q "$queue" \
            -l "ncpus=$workers,mem=$memory,walltime=$walltime,jobfs=4GB" \
            "${dependency[@]}" -v "RUN_DIR=$run" "$run/submit.pbs")
        if [[ ! $job =~ ^[0-9]+([.][A-Za-z0-9._-]+)?$ ]]; then
            echo "Unexpected qsub job ID: $job" >&2
            exit 1
        fi
        printf '%s\n' "$job" > "$run/job_id.txt"
        export C2_SUITE_CASE="$name" C2_SUITE_JOB="$job" C2_SUITE_PREVIOUS="$previous"
        python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ['C2_SUITE_DIRECTORY'])/'suite_manifest.json'
manifest = json.loads(path.read_text())
case = next(item for item in manifest['cases'] if item['name'] == os.environ['C2_SUITE_CASE'])
case['job_id'] = os.environ['C2_SUITE_JOB']
case['dependency_job_id'] = os.environ['C2_SUITE_PREVIOUS'] or None
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(manifest, indent=2) + '\n')
temporary.replace(path)
PY
        printf '%s: %s\n' "$name" "$job"
        previous=$job
        index=$((index + 1))
    done < "$suite/cases.tsv"
    touch "$suite/SUBMITTED"
fi
