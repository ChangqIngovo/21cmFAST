#!/usr/bin/env bash
# Freeze a new coeval comparison; reuse original 21cmFAST snapshots unchanged.
set -euo pipefail
submit=false
input_run=${PC_INPUT_RUN:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --input-run) input_run=${2:?Supply the archived native input run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_coeval/gadi_prepare.sh --input-run RUN [--submit]'
            echo 'Compares independent coeval PC and original SCRIPT on identical native inputs.'
            echo 'Default: 8 CPUs, 32GB, 4h, ten archived snapshots from z=10 to z=6.'
            echo 'PC_THREADS/PC_MEM/PC_WALLTIME/PC_QUEUE/PC_PROJECT/PC_STORAGE set resources.'
            echo 'PC_REDSHIFTS selects exact archived snapshots; PC_TOLERANCE defaults to 1e-8.'
            echo 'PC_MAX_ITERATIONS=1000; PC_MAX_INTERACTIONS=1000000000000; PC_SCRIPT_TIMEOUT=900.'
            echo 'No finite light speed or continuous gas evolution in this coeval approximation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n $input_run ]] || { echo 'Set PC_INPUT_RUN or pass --input-run.' >&2; exit 2; }
input_run=$(realpath "$input_run")
root=$(realpath -m "${PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}")
for path in "$input_run" "$root"; do
    [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'PBS paths must be absolute and contain no spaces, commas or shell metacharacters.' >&2; exit 2; }
done
case "$root/" in "$input_run/"*) echo 'PC_RUN_ROOT must be outside the archived input run.' >&2; exit 2 ;; esac
for name in experiment_config.json runtime.env python_path.txt results/manifest.json \
            native_runtime/manifest.json native_runtime/manifest.sha256; do
    [[ -f $input_run/$name ]] || { echo "Missing archived input: $input_run/$name" >&2; exit 2; }
done
[[ -d $input_run/results/histories && -d $input_run/native_build/source/src ]] || {
    echo 'Input run requires native histories and its original native build.' >&2; exit 2;
}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_coeval research/pc_script_reference)
[[ -z $untracked ]] || { echo 'Commit new coeval and SCRIPT files before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/pc_coeval/run.py
revision=$(git -C "$repo" rev-parse HEAD)
pc_python=${PC_PYTHON:-$(cat "$input_run/python_path.txt")}
[[ $pc_python = /* && -x $pc_python ]] || { echo 'Python must be an absolute executable path.' >&2; exit 2; }
pc_mem=${PC_MEM:-32GB}
pc_walltime=${PC_WALLTIME:-04:00:00}
pc_threads=${PC_THREADS:-8}
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'Invalid PC_MEM.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'Invalid PC_WALLTIME.' >&2; exit 2; }
[[ $pc_threads =~ ^[1-9][0-9]*$ ]] && (( pc_threads <= 32 )) || { echo 'PC_THREADS must be in [1,32].' >&2; exit 2; }
export PC_MEM="$pc_mem" PC_THREADS="$pc_threads"
mkdir -p "$root"
run=$(mktemp -d "$root/coeval_pc_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
mkdir "$run/source" "$run/pbs_logs"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$input_run" > "$run/input_run.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$input_run/runtime.env" "$run/runtime.env"
PC_MODULES=()
source "$run/runtime.env"
if (( ${#PC_MODULES[@]} )); then
    pc_gcc_present=false
    for pc_module in "${PC_MODULES[@]}"; do
        if [[ ${pc_module%%/*} == gcc ]]; then pc_gcc_present=true; fi
    done
    if ! $pc_gcc_present; then PC_MODULES+=("${PC_GFORTRAN_MODULE:-gcc/12.2.0}"); fi
    printf '\n' >> "$run/runtime.env"
    declare -p PC_MODULES >> "$run/runtime.env"
    module --force purge
    module load "${PC_MODULES[@]}"
    module -t list > "$run/modules.txt" 2>&1
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cp "$run/source/research/pc_coeval/gadi_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_coeval.prepare prepare --run "$run"

# PBS output must never appear inside the frozen source tree, including on retry.
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$pc_threads,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_COEVAL_RUN=$run" "$run/submit.pbs")
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
printf 'Prepared coeval PC comparison: %s\nOriginal native input: %s\n' "$run" "$input_run"
printf 'Submit/resubmit: bash %q\n' "$run/resubmit.sh"
if $submit; then bash "$run/resubmit.sh" | tee "$run/jobid.txt"; fi
