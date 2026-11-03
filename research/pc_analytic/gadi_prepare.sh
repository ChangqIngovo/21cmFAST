#!/usr/bin/env bash
# Freeze a small same-input analytic-PC / SCRIPT / C2-Ray comparison.
set -euo pipefail
submit=false
input_run=${PC_INPUT_RUN:-}
c2_source=${PC_C2RAY_SOURCE:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --input-run) input_run=${2:?Supply the original native input run}; shift 2 ;;
        --c2ray-source) c2_source=${2:?Supply a pinned pyC2Ray source directory}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_analytic/gadi_prepare.sh --input-run RUN [--c2ray-source DIR] [--submit]'
            echo 'Default pilot: z=8,7.5,7; 8 CPUs, 32GB, 4h; actual C2-Ray is required.'
            echo 'N>16 is rejected unless PC_COARSEN_TO=16 explicitly requests conservative block averages.'
            echo 'This changes source/density resolution for all three methods; it is not a high-resolution certification.'
            echo 'PC_SEEDS, PC_REDSHIFTS, PC_DIRECTIONS, PC_EXPOSURE_STEPS, PC_C2_EXPOSURE_STEPS select the pilot.'
            echo 'PC_C2_TOLERANCE=1e-6 and PC_C2_MAX_ITERATIONS=2000 control the declared C2-Ray opacity iteration.'
            echo 'PC_BOUNDARY=source_periodic_cube (default) matches the C2-Ray domain; periodic/open are explicit alternatives.'
            echo 'Characteristic transport uses full ray chords, without interpolated fractional-cell translations.'
            echo 'PC_C2_SUBBOX_SIZE=2 and PC_C2_LOSS_FRACTION=1e-10 control native subbox termination; use size>=N and loss=0 for a full-cube control.'
            echo 'PC_SIGMA_CM2 sets the common cross-section; default is the declared monochromatic 18-eV approximation.'
            echo 'Pinned upstream files download on the login node if --c2ray-source is omitted. Compute jobs need no network.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n $input_run ]] || { echo 'Set PC_INPUT_RUN or pass --input-run.' >&2; exit 2; }
input_run=$(realpath "$input_run")
root=$(realpath -m "${PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}")
for path in "$input_run" "$root"; do
    [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'PBS paths must be absolute without spaces, commas or shell metacharacters.' >&2; exit 2; }
done
case "$root/" in "$input_run/"*) echo 'PC_RUN_ROOT must be outside the archived input.' >&2; exit 2 ;; esac
for name in experiment_config.json runtime.env python_path.txt results/manifest.json; do
    [[ -f $input_run/$name ]] || { echo "Missing archived input: $input_run/$name" >&2; exit 2; }
done
[[ -d $input_run/results/histories && -d $input_run/native_build/source/src ]] || {
    echo 'Use the original native-source archive, containing histories and native_build/source/src.' >&2; exit 2;
}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_analytic)
[[ -z $untracked ]] || { echo 'Commit analytic comparison files before preparing.' >&2; exit 2; }
for pc_code in compare.py c2ray_reference.py c2ray_driver.F90; do
    git -C "$repo" cat-file -e "HEAD:research/pc_analytic/$pc_code" || {
        echo "Commit missing analytic source file: $pc_code" >&2; exit 2;
    }
done
revision=$(git -C "$repo" rev-parse HEAD)
pc_python=${PC_PYTHON:-$(cat "$input_run/python_path.txt")}
[[ $pc_python = /* && -x $pc_python ]] || { echo 'PC_PYTHON must identify an absolute executable.' >&2; exit 2; }
pc_threads=${PC_THREADS:-8}
pc_mem=${PC_MEM:-32GB}
pc_walltime=${PC_WALLTIME:-04:00:00}
[[ $pc_threads =~ ^[1-9][0-9]*$ ]] && (( pc_threads <= 32 )) || { echo 'Invalid PC_THREADS.' >&2; exit 2; }
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'Invalid PC_MEM.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'Invalid PC_WALLTIME.' >&2; exit 2; }
export PC_THREADS="$pc_threads"
if [[ -n $c2_source ]]; then export PC_C2RAY_SOURCE="$c2_source"; fi
mkdir -p "$root"
run=$(mktemp -d "$root/analytic_pc_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
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
cp "$run/source/research/pc_analytic/gadi_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_analytic.prepare prepare --run "$run"
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$pc_threads,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_ANALYTIC_RUN=$run" "$run/submit.pbs")
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
printf 'Prepared analytic comparison: %s\nOriginal input: %s\n' "$run" "$input_run"
printf 'Submit/resubmit: bash %q\n' "$run/resubmit.sh"
if $submit; then bash "$run/resubmit.sh" | tee "$run/jobid.txt"; fi
