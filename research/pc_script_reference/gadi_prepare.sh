#!/usr/bin/env bash
# Freeze the original SCRIPT comparison; never rerun native sources or RT here.
set -euo pipefail
submit=false
input_run=${PC_INPUT_RUN:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --input-run) input_run=${2:?Supply an existing causal RT run}; shift 2 ;;
        --help)
            echo 'Usage: PC_INPUT_RUN=/path/to/causal_rt_run bash research/pc_script_reference/gadi_prepare.sh [--submit]'
            echo 'Runs the pinned original SCRIPT kernel on archived native source/density fields.'
            echo 'No native simulation or RT refinement matrix is submitted.'
            echo 'PC_RUN_ROOT defaults to /g/data/qp00/sz4640/21cmfast_runs.'
            echo 'PC_MEM/WALLTIME default to 4GB/01:00:00; CPU count matches the archived native configuration.'
            echo 'PC_PROJECT/STORAGE/QUEUE override the PBS allocation.'
            echo 'PC_PYTHON overrides the input run Python; PC_GFORTRAN_MODULE defaults to gcc/12.2.0.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n $input_run ]] || { echo 'Set PC_INPUT_RUN to the existing causal RT run.' >&2; exit 2; }
input_run=$(realpath "$input_run")
[[ $input_run != *$'\n'* && $input_run != *$'\r'* ]] || { echo 'Invalid input path.' >&2; exit 2; }
for required in experiment_config.json runtime.env python_path.txt; do
    [[ -f $input_run/$required ]] || { echo "Missing input run file: $required" >&2; exit 2; }
done
[[ -d $input_run/results/histories ]] || { echo 'Input run has no archived source histories.' >&2; exit 2; }
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_script_reference)
[[ -z $untracked ]] || { echo 'Commit new pc_script_reference files before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/pc_script_reference/run.py
git -C "$repo" cat-file -e HEAD:research/pc_script_reference/gadi_run.pbs
revision=$(git -C "$repo" rev-parse HEAD)
pc_python=${PC_PYTHON:-$(cat "$input_run/python_path.txt")}
[[ $pc_python = /* && -x $pc_python ]] || { echo 'Selected Python must be an absolute executable path.' >&2; exit 2; }
pc_mem=${PC_MEM:-4GB}
pc_walltime=${PC_WALLTIME:-01:00:00}
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'PC_MEM must be a positive integer MB/GB/TB value.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'PC_WALLTIME must be a positive HH:MM:SS duration.' >&2; exit 2; }
root=${PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
root=$(realpath -m "$root")
case "$root/" in
    "$input_run/"*) echo 'PC_RUN_ROOT must be outside the readonly input run.' >&2; exit 2 ;;
esac
mkdir -p "$root"
run=$(mktemp -d "$root/script_reference_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* && $run != *$'\r'* ]] || { echo 'Invalid PBS run path.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
mkdir "$run/source"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$input_run" > "$run/input_run.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$input_run/runtime.env" "$run/runtime.env"
PC_MODULES=()
source "$run/runtime.env"
if (( ${#PC_MODULES[@]} )); then
    pc_gcc_present=false
    for pc_existing_module in "${PC_MODULES[@]}"; do
        if [[ ${pc_existing_module%%/*} == gcc ]]; then pc_gcc_present=true; fi
    done
    if ! $pc_gcc_present; then PC_MODULES+=("${PC_GFORTRAN_MODULE:-gcc/12.2.0}"); fi
    printf '\n' >> "$run/runtime.env"
    declare -p PC_MODULES >> "$run/runtime.env"
    type module >/dev/null 2>&1 || { echo 'Runtime requires unavailable environment modules.' >&2; exit 2; }
    module --force purge
    module load "${PC_MODULES[@]}"
    module -t list > "$run/modules.txt" 2>&1
else
    printf 'No environment modules requested.\n' > "$run/modules.txt"
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
cp "$run/source/research/pc_script_reference/gadi_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_script_reference.prepare_run prepare --run "$run"
workers=$("$pc_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ncpus"])' "$run/run_manifest.json")
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$workers,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_SCRIPT_REFERENCE_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared original SCRIPT reference: %s\nExisting input: %s\nPython: %s\n' "$run" "$input_run" "$pc_python"
printf '%q ' "${command[@]}"
printf '\nSubmit/resubmit this frozen run: bash %q\n' "$run/resubmit.sh"
if $submit; then "${command[@]}" | tee "$run/jobid.txt"; fi
