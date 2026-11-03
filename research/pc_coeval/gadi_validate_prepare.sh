#!/usr/bin/env bash
# Postprocess existing coeval PC/SCRIPT/ES fields without rerunning simulations.
set -euo pipefail
submit=false
latest=false
input_run=${PC_INPUT_RUN:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --latest) latest=true; shift ;;
        --input-run) input_run=${2:?Supply the completed coeval comparison run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_coeval/gadi_validate_prepare.sh (--input-run COEVAL_RUN | --latest) [--submit]'
            echo 'Also accepts COEVAL_RUN/results. Reuses saved fields; no PC, SCRIPT or native simulation.'
            echo '--latest explicitly selects the latest completed coeval_pc_* under PC_RUN_ROOT and prints its path.'
            echo 'Defaults: 8 CPUs, 16GB, 4h; 100000 rays per model, phase and threshold.'
            echo 'PC_MFP_THRESHOLDS=0.1,0.5,0.9; PC_MFP_SEED=20261103; PC_MFP_MAX_DISTANCE_BOXES=1.'
            echo 'PC_THREADS/PC_MEM/PC_WALLTIME/PC_QUEUE/PC_PROJECT/PC_STORAGE set resources.'
            echo 'Runtime defaults to the coeval input; override PC_PYTHON and optionally PC_RUNTIME_ENV.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
root=$(realpath -m "${PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}")
if $latest; then
    [[ -z $input_run ]] || { echo 'Choose --latest or --input-run/PC_INPUT_RUN, not both.' >&2; exit 2; }
    latest_marker=
    shopt -s nullglob
    for candidate in "$root"/coeval_pc_*/results/COMPLETED; do
        candidate_root=$(dirname "$candidate")
        [[ -f $candidate_root/report_receipt.json && -f $candidate_root/coeval_report.json ]] || continue
        if [[ -z $latest_marker || $candidate -nt $latest_marker ]]; then latest_marker=$candidate; fi
    done
    shopt -u nullglob
    [[ -n $latest_marker ]] || { echo "No completed coeval_pc_* run found under $root." >&2; exit 2; }
    input_run=$(dirname "$(dirname "$latest_marker")")
    printf 'Selected latest completed coeval comparison: %s\n' "$input_run"
fi
[[ -n $input_run ]] || { echo 'Set PC_INPUT_RUN or pass --input-run.' >&2; exit 2; }
input_run=$(realpath "$input_run")
if [[ -f $input_run/coeval_report.json ]]; then
    input_results=$input_run
    [[ ${input_run##*/} != results ]] || input_run=$(dirname "$input_run")
else
    input_results=$input_run/results
fi
for path in "$input_run" "$input_results" "$root"; do
    [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'PBS paths must be absolute without spaces, commas or shell metacharacters.' >&2; exit 2; }
done
case "$root/" in "$input_run/"*) echo 'PC_RUN_ROOT must be outside the coeval input run.' >&2; exit 2 ;; esac
for name in identity.json coeval_report.json report_receipt.json COMPLETED; do
    [[ -f $input_results/$name ]] || { echo "Missing completed coeval input: $input_results/$name" >&2; exit 2; }
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_coeval)
[[ -z $untracked ]] || { echo 'Commit new coeval validation files before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/pc_coeval/validate.py
revision=$(git -C "$repo" rev-parse HEAD)
pc_python=${PC_PYTHON:-}
if [[ -z $pc_python && -f $input_run/python_path.txt ]]; then pc_python=$(cat "$input_run/python_path.txt"); fi
[[ $pc_python = /* && -x $pc_python ]] || { echo 'Set PC_PYTHON to an absolute executable, or use a coeval run with python_path.txt.' >&2; exit 2; }
pc_runtime=${PC_RUNTIME_ENV:-$input_run/runtime.env}
[[ -f $pc_runtime || -n ${PC_PYTHON:-} ]] || { echo 'Input runtime.env is missing; supply PC_PYTHON and optionally PC_RUNTIME_ENV.' >&2; exit 2; }
pc_mem=${PC_MEM:-16GB}
pc_walltime=${PC_WALLTIME:-04:00:00}
pc_threads=${PC_THREADS:-8}
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'Invalid PC_MEM.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'Invalid PC_WALLTIME.' >&2; exit 2; }
[[ $pc_threads =~ ^[1-9][0-9]*$ ]] && (( pc_threads <= 32 )) || { echo 'PC_THREADS must be in [1,32].' >&2; exit 2; }
export PC_THREADS="$pc_threads"
mkdir -p "$root"
run=$(mktemp -d "$root/coeval_validation_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
mkdir "$run/source" "$run/pbs_logs"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$input_results" > "$run/input_run.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
if [[ -f $pc_runtime ]]; then cp "$pc_runtime" "$run/runtime.env"; else printf 'PC_MODULES=()\n' > "$run/runtime.env"; fi
PC_MODULES=()
source "$run/runtime.env"
if (( ${#PC_MODULES[@]} )); then
    pc_gcc_present=false
    for pc_module in "${PC_MODULES[@]}"; do
        if [[ ${pc_module%%/*} == gcc ]]; then pc_gcc_present=true; fi
    done
    if ! $pc_gcc_present; then PC_MODULES+=("${PC_GCC_MODULE:-gcc/12.2.0}"); fi
    printf '\n' >> "$run/runtime.env"
    declare -p PC_MODULES >> "$run/runtime.env"
    module --force purge
    module load "${PC_MODULES[@]}"
    module -t list > "$run/modules.txt" 2>&1
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cp "$run/source/research/pc_coeval/gadi_validate_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_coeval.validate_prepare prepare --run "$run"
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$pc_threads,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_COEVAL_VALIDATION_RUN=$run" "$run/submit.pbs")
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'pc_run=%q\ncd "$pc_run"\n' "$run"
    printf 'pc_stamp=$(date -u +%%Y%%m%%dT%%H%%M%%S)_$$\n'
    printf '%q ' "${command[0]}"
    printf '%s ' '-o "$pc_run/pbs_logs/output_${pc_stamp}.log" -e "$pc_run/pbs_logs/error_${pc_stamp}.log"'
    printf '%q ' "${command[@]:1}"
    printf '\n'
} > "$run/resubmit.sh"
cd "$run"
printf 'Prepared photon/MFP validation: %s\nRead-only coeval input: %s\n' "$run" "$input_results"
printf 'Submit/resubmit: bash %q\n' "$run/resubmit.sh"
if $submit; then bash "$run/resubmit.sh" | tee "$run/jobid.txt"; fi
