#!/usr/bin/env bash
# Freeze independent native realizations and parallel analytic teacher generation.
set -euo pipefail
submit=false
parent=${PC_RUNTIME_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply the original native-source run}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_analytic/gadi_teacher_prepare.sh --parent RUN [--submit]'
            echo 'Default: 8 fresh seeds, native N=32, high-resolution N=256, L=256 cMpc, 7 coeval redshifts.'
            echo 'The default density grid is max(3*PC_DIMENSION, ceil(PC_BOX_LEN)), keeping its cell size <=1 cMpc.'
            echo 'PC_WORKERS=4 independent seeds run concurrently; PC_THREADS=4 threads per worker.'
            echo 'Workers share a single node; this launcher does not distribute MPI ranks across nodes.'
            echo 'PC_NCPUS defaults to workers*threads; 64GB, 48GiB working budget and 12h by default.'
            echo 'PC_SEEDS, PC_REDSHIFTS, PC_DIMENSION, PC_HIGHRES_DIMENSION and PC_BOX_LEN select the dataset.'
            echo 'PC_DIRECTIONS=96 and PC_EXPOSURE_STEPS=32 select the analytic solver accuracy.'
            echo 'The parent supplies verified full physical inputs and its existing native runtime.'
            echo 'PC_TEMPLATE_SEED=222 and PC_TEMPLATE_REDSHIFT=7.5 select the physical template.'
            echo 'Completed seed/snapshot receipts are reused by the generated resubmit.sh.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n $parent ]] || { echo 'Set PC_RUNTIME_PARENT or pass --parent.' >&2; exit 2; }
parent=$(realpath "$parent")
root=$(realpath -m "${PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}")
for path in "$parent" "$root"; do
    [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'PBS paths must be absolute without spaces, commas or shell metacharacters.' >&2; exit 2; }
done
case "$root/" in "$parent/"*) echo 'PC_RUN_ROOT must be outside the archived parent.' >&2; exit 2 ;; esac
for name in experiment_config.json runtime.env python_path.txt results/manifest.json; do
    [[ -f $parent/$name ]] || { echo "Missing archived runtime input: $parent/$name" >&2; exit 2; }
done
[[ -d $parent/results/histories && -d $parent/native_build/source/src ]] || {
    echo 'Use the original native-source run containing histories and native_build/source/src.' >&2; exit 2;
}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_analytic)
[[ -z $untracked ]] || { echo 'Commit analytic teacher files before preparing.' >&2; exit 2; }
for pc_code in campaign.py teacher_prepare.py solver.py transport_kernel.cpp gadi_teacher_run.pbs; do
    git -C "$repo" cat-file -e "HEAD:research/pc_analytic/$pc_code" || {
        echo "Commit missing teacher source file: $pc_code" >&2; exit 2;
    }
done
revision=$(git -C "$repo" rev-parse HEAD)
pc_python=${PC_PYTHON:-$(cat "$parent/python_path.txt")}
[[ $pc_python = /* && -x $pc_python ]] || { echo 'PC_PYTHON must identify an absolute executable.' >&2; exit 2; }
pc_workers=${PC_WORKERS:-4}
pc_threads=${PC_THREADS:-4}
for value in "$pc_workers" "$pc_threads"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'PC_WORKERS and PC_THREADS must be positive integers.' >&2; exit 2; }
done
pc_cpus=${PC_NCPUS:-$((pc_workers * pc_threads))}
[[ $pc_cpus =~ ^[1-9][0-9]*$ ]] && (( pc_threads <= 32 && pc_cpus >= pc_workers * pc_threads )) || {
    echo 'PC_NCPUS must cover PC_WORKERS*PC_THREADS; PC_THREADS must be <=32.' >&2; exit 2;
}
pc_mem=${PC_MEM:-64GB}
pc_walltime=${PC_WALLTIME:-12:00:00}
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'Invalid PC_MEM.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'Invalid PC_WALLTIME.' >&2; exit 2; }
export PC_WORKERS="$pc_workers" PC_THREADS="$pc_threads" PC_NCPUS="$pc_cpus"
export PC_MEM="$pc_mem" PC_MEMORY_GB="${PC_MEMORY_GB:-48}"
mkdir -p "$root"
run=$(mktemp -d "$root/analytic_teacher_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
mkdir "$run/source" "$run/pbs_logs"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/parent_run.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$parent/runtime.env" "$run/runtime.env"
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
export PYTHONPATH="$parent/native_build/source/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cp "$run/source/research/pc_analytic/gadi_teacher_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_analytic.teacher_prepare prepare --run "$run"
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$pc_cpus,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_TEACHER_RUN=$run" "$run/submit.pbs")
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
printf 'Prepared analytic teacher dataset: %s\nNative runtime parent: %s\n' "$run" "$parent"
printf 'Submit/resubmit: bash %q\n' "$run/resubmit.sh"
if $submit; then bash "$run/resubmit.sh" | tee "$run/jobid.txt"; fi
