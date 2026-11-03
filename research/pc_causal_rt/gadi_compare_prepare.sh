#!/usr/bin/env bash
# Freeze one native ES / continuous PC RT / original SCRIPT comparison.
set -euo pipefail
submit=false
parent=${PC_RUNTIME_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply an existing native runtime run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_causal_rt/gadi_compare_prepare.sh [--parent RUN] [--submit]'
            echo 'Defaults: 256 cMpc, native 128^3, initial 384^3, seed 222, 32 threads.'
            echo 'PC RT: D96/f1/CFL0.4/t4/s1; z=25..5.75, 154 intervals, every node saved.'
            echo 'Native ES and original SCRIPT use the same archived native inputs.'
            echo 'PC_CONFIG_JSON overrides defaults; existing scientific PC_* overrides remain supported.'
            echo 'Unset old PC_CONFIG_JSON and numerical-axis variables to use the default single case.'
            echo 'PC_COMPARE_RUN_ROOT falls back to PC_CAUSAL_RT_RUN_ROOT and the usual Gadi run root.'
            echo 'PC_PARENT_ROOT selects the existing runtime-discovery root.'
            echo 'PC_MEM/MEMORY_GB/WALLTIME/JOBFS default to 128GB/100GiB/48:00:00/10GB.'
            echo 'PC_SCRIPT_TIMEOUT is the per-snapshot SCRIPT limit in seconds (default 14400).'
            echo 'PC_PROJECT/STORAGE/QUEUE override PBS allocation; preparation runs no simulation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

export PC_MEM=${PC_MEM:-128GB}
export PC_MEMORY_GB=${PC_MEMORY_GB:-100}
export PC_WALLTIME=${PC_WALLTIME:-48:00:00}
pc_jobfs=${PC_JOBFS:-10GB}
script_timeout=${PC_SCRIPT_TIMEOUT:-14400}
[[ $PC_WALLTIME =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || {
    echo 'PC_WALLTIME must be a positive HH:MM:SS duration.' >&2; exit 2;
}
[[ $pc_jobfs =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || {
    echo 'PC_JOBFS must be a positive integer MB/GB/TB allocation.' >&2; exit 2;
}
python3 - "$script_timeout" "$PC_WALLTIME" <<'PY'
import math
import sys
timeout = float(sys.argv[1])
if not math.isfinite(timeout) or timeout <= 0:
    raise SystemExit("PC_SCRIPT_TIMEOUT must be finite and positive")
if not any(int(value) for value in sys.argv[2].split(":")):
    raise SystemExit("PC_WALLTIME must be positive")
PY

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_causal_rt research/pc_script_reference)
[[ -z $untracked ]] || { echo 'Commit new causal RT and SCRIPT files before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/pc_causal_rt/gadi_compare_run.pbs
revision=$(git -C "$repo" rev-parse HEAD)
previous_root=${PC_CAUSAL_RT_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
root=${PC_COMPARE_RUN_ROOT:-$previous_root}
root=$(realpath -m "$root")
export PC_PARENT_ROOT=${PC_PARENT_ROOT:-$previous_root}
export PC_PARENT_ROOT=$(realpath -m "$PC_PARENT_ROOT")
custom_config=${PC_CONFIG_JSON:-}
if [[ -n $custom_config ]]; then custom_config=$(realpath "$custom_config"); fi
mkdir -p "$root"
campaign=$(mktemp -d "$root/pc_compare_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
campaign=$(realpath "$campaign")
[[ $campaign != *,* && $campaign != *$'\n'* && $campaign != *$'\r'* ]] || {
    echo 'Invalid PBS campaign path.' >&2; exit 2;
}
exec > >(tee "$campaign/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$campaign/FAILED_PREPARE"; fi' EXIT

# Build explicit defaults rather than inheriting the eleven-case convergence
# ladder. An optional JSON and scientific environment overrides are deliberate
# user inputs; the child preparation validates and freezes their final values.
python3 - "$campaign/comparison_defaults.json" "$custom_config" <<'PY'
import json
import sys
from pathlib import Path
cfg = {
    "dimension": 128, "highres_dimension": 384, "box_len": 256.0,
    "threads": 32, "seeds": [222], "z_start": 25.0, "z_end": 5.75,
    "source_intervals": 154, "checkpoint_stride": 1,
    "source_mode": "native_cumulative",
    "settings": [{"directions": 96, "refinement": 1, "cfl": 0.4,
                  "time_substeps": 4, "source_stride": 1}],
}
if sys.argv[2]:
    path = Path(sys.argv[2])
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict):
        raise SystemExit("PC_CONFIG_JSON must contain a dictionary")
    native = supplied.get("native_inputs_json")
    if isinstance(native, str) and not Path(native).is_absolute():
        supplied["native_inputs_json"] = str((path.parent / native).resolve())
    cfg.update(supplied)
Path(sys.argv[1]).write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
PY
export PC_CONFIG_JSON="$campaign/comparison_defaults.json"
export PC_CAUSAL_RT_RUN_ROOT="$campaign"
prepare_arguments=()
[[ -z $parent ]] || prepare_arguments+=(--parent "$parent")
bash "$repo/research/pc_causal_rt/gadi_prepare.sh" "${prepare_arguments[@]}"
shopt -s nullglob
children=("$campaign"/causal_rt_*)
[[ ${#children[@]} == 1 && -f ${children[0]}/PREPARED ]] || {
    echo 'Expected exactly one prepared native RT run.' >&2; exit 2;
}
run=${children[0]}
workers=$(python3 - "$run/experiment_config.json" <<'PY'
import json
import sys
cfg = json.load(open(sys.argv[1]))
if cfg["source_mode"] != "native_cumulative":
    raise SystemExit("The three-model comparison requires native_cumulative sources")
print(cfg["threads"])
PY
)

# The child submit.pbs is part of its immutable preparation receipt. Freeze a
# separate wrapper, leaving that file and all child configuration untouched.
cp "$run/source/research/pc_causal_rt/gadi_compare_run.pbs" "$campaign/submit.pbs"
printf '%s\n' "$run" > "$campaign/run_path.txt"
printf '%s\n' "$campaign/script" > "$campaign/script_output_path.txt"
printf '%s\n' "$script_timeout" > "$campaign/script_timeout.txt"
printf '%s\n' "$revision" > "$campaign/commit.txt"
python3 - "$campaign/allocation.json" "$workers" "$PC_MEM" "$PC_WALLTIME" "$pc_jobfs" "${PC_PROJECT:-qp00}" "${PC_QUEUE:-normal}" "${PC_STORAGE:-gdata/qp00+scratch/qp00}" <<'PY'
import json
import sys
keys = ("ncpus", "mem", "walltime", "jobfs", "project", "queue", "storage")
allocation = dict(zip(keys, sys.argv[2:]))
allocation["ncpus"] = int(allocation["ncpus"])
with open(sys.argv[1], "w") as stream:
    json.dump(allocation, stream, indent=2)
    stream.write("\n")
PY
(
    cd "$campaign"
    sha256sum submit.pbs run_path.txt script_output_path.txt script_timeout.txt commit.txt \
        comparison_defaults.json allocation.json > prepared.sha256
)
printf 'Frozen native ES / PC RT / original SCRIPT comparison.\n' > "$campaign/PREPARED"
command=(qsub -N pc_compare -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$workers,mem=$PC_MEM,walltime=$PC_WALLTIME,jobfs=$pc_jobfs,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_COMPARE_RUN=$campaign" "$campaign/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$campaign/resubmit.sh"
printf '%q ' "${command[@]}" >> "$campaign/resubmit.sh"
printf '\n' >> "$campaign/resubmit.sh"
printf 'Prepared three-model comparison: %s\nNative/PC run: %s\n' "$campaign" "$run"
printf '%q ' "${command[@]}"
printf '\nSubmit/resume the complete comparison: bash %q\n' "$campaign/resubmit.sh"
if $submit; then "${command[@]}" | tee "$campaign/jobid.txt"; fi
