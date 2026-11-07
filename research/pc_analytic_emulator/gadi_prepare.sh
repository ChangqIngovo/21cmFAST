#!/usr/bin/env bash
# Freeze completed analytic teachers and submit a seed-separated training job.
set -euo pipefail
submit=false
latest=false
teacher=${PC_TEACHER_RUN:-}
external_test=${PC_EXTERNAL_TEST_ROOT:-}
training_method=${PC_TRAINING_METHOD:-legacy}
root=$(realpath -m "${PC_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}")
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --teacher-run) teacher=${2:?Supply a completed analytic_teacher run}; shift 2 ;;
        --latest-teacher) latest=true; shift ;;
        --improved) training_method=projected_multiscale_v2; shift ;;
        --external-test-root) external_test=${2:?Supply a separate completed teacher archive}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_analytic_emulator/gadi_prepare.sh (--teacher-run RUN | --latest-teacher) [--improved] [--external-test-root RUN] [--submit]'
            echo 'Latest mode selects the newest completed N32/L256 campaign with seeds 1001..1008 and all 7 requested redshifts.'
            echo 'Defaults: seeds 1001..1006 train, 1007 validates, 1008 tests; all redshifts stay with their seed.'
            echo 'Legacy: four initializations (11,23,47,83), 200 epochs maximum, 8192 sampled cells per training snapshot.'
            echo '--improved selects projected_multiscale_v2; PC_V2_EPOCHS=80 controls its refinement stage.'
            echo 'V2 uses every training cell; PC_EMULATOR_EPOCHS controls budget fitting and the retained legacy competitors.'
            echo 'Seed 1008 is a previously inspected regression test; an optional external archive must contain new initial conditions.'
            echo 'The external archive remains separate and never selects model parameters or checkpoints.'
            echo 'Resources: 4 workers x 4 BLAS threads, 16 CPUs on one node, 64GB, 4 hours.'
            echo 'PC_TRAIN_SEEDS, PC_VALIDATION_SEEDS, PC_TEST_SEEDS, PC_MODEL_SEEDS select splits/trials.'
            echo 'PC_EMULATOR_EPOCHS, PC_SAMPLES_PER_SNAPSHOT, PC_WORKERS, PC_THREADS and PC_NCPUS control fitting.'
            echo 'PC_PYTHON overrides the completed teacher interpreter; its modules are reused.'
            echo 'This workflow reads existing teacher data and runs no simulations or reference solvers.'
            echo 'The generated resubmit.sh reuses verified completed trial checkpoints.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ $training_method == legacy || $training_method == projected_multiscale_v2 ]] || {
    echo 'PC_TRAINING_METHOD must be legacy or projected_multiscale_v2.' >&2; exit 2;
}
export PC_TRAINING_METHOD="$training_method"
if $latest && [[ -n $teacher ]]; then
    echo 'Select exactly one of --teacher-run (or PC_TEACHER_RUN) and --latest-teacher.' >&2
    exit 2
fi
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if $latest; then
    pc_bootstrap_python=${PC_PYTHON:-python3}
    if ! "$pc_bootstrap_python" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        echo 'Teacher discovery requires Python >=3.11. On Gadi run: module load python3/3.12.1; or set PC_PYTHON to that interpreter.' >&2
        exit 2
    fi
    teacher=$("$pc_bootstrap_python" "$repo/research/pc_analytic_emulator/prepare.py" locate --root "$root")
fi
[[ -n $teacher ]] || { echo 'Pass --teacher-run or --latest-teacher.' >&2; exit 2; }
teacher=$(realpath "$teacher")
if [[ $(basename "$teacher") == results ]]; then teacher=$(dirname "$teacher"); fi
for path in "$teacher" "$root"; do
    [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'PBS paths must be absolute without spaces, commas or shell metacharacters.' >&2; exit 2; }
done
case "$root/" in "$teacher/"*) echo 'PC_RUN_ROOT must be outside the teacher archive.' >&2; exit 2 ;; esac
if [[ -n $external_test ]]; then
    external_test=$(realpath "$external_test")
    [[ $external_test =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'External test paths must be absolute without spaces, commas or shell metacharacters.' >&2; exit 2; }
    case "$root/" in "$external_test/"*) echo 'PC_RUN_ROOT must be outside the external test archive.' >&2; exit 2 ;; esac
    export PC_EXTERNAL_TEST_ROOT="$external_test"
else
    unset PC_EXTERNAL_TEST_ROOT
fi
for name in runtime.env python_path.txt results/COMPLETED results/dataset_manifest.json results/completion.json; do
    [[ -f $teacher/$name ]] || { echo "Missing completed teacher input: $teacher/$name" >&2; exit 2; }
done
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_analytic_emulator)
[[ -z $untracked ]] || { echo 'Commit analytic emulator files before preparing.' >&2; exit 2; }
for pc_code in data.py train.py prepare.py gadi_train.pbs; do
    git -C "$repo" cat-file -e "HEAD:research/pc_analytic_emulator/$pc_code" || {
        echo "Missing committed training source: $pc_code" >&2; exit 2;
    }
done
revision=$(git -C "$repo" rev-parse HEAD)
pc_python=${PC_PYTHON:-$(cat "$teacher/python_path.txt")}
[[ $pc_python = /* && -x $pc_python ]] || { echo 'PC_PYTHON must identify an absolute executable.' >&2; exit 2; }
pc_workers=${PC_WORKERS:-4}
pc_threads=${PC_THREADS:-4}
for value in "$pc_workers" "$pc_threads"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'PC_WORKERS and PC_THREADS must be positive integers.' >&2; exit 2; }
done
pc_cpus=${PC_NCPUS:-$((pc_workers * pc_threads))}
[[ $pc_cpus =~ ^[1-9][0-9]*$ ]] && (( pc_cpus >= pc_workers * pc_threads )) || {
    echo 'PC_NCPUS must cover PC_WORKERS*PC_THREADS.' >&2; exit 2;
}
pc_mem=${PC_MEM:-64GB}
pc_walltime=${PC_WALLTIME:-04:00:00}
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'Invalid PC_MEM.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'Invalid PC_WALLTIME.' >&2; exit 2; }
export PC_WORKERS="$pc_workers" PC_THREADS="$pc_threads" PC_NCPUS="$pc_cpus"
mkdir -p "$root"
run=$(mktemp -d "$root/analytic_train_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
mkdir "$run/source" "$run/pbs_logs"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$teacher" > "$run/teacher_run.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
cp "$teacher/runtime.env" "$run/runtime.env"
cp "$run/source/research/pc_analytic_emulator/gadi_train.pbs" "$run/submit.pbs"
PC_MODULES=()
source "$run/runtime.env"
if (( ${#PC_MODULES[@]} )); then
    module --force purge
    module load "${PC_MODULES[@]}"
    module -t list > "$run/modules.txt" 2>&1
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1
cd "$run/source"
"$pc_python" -m research.pc_analytic_emulator.prepare prepare --run "$run"
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$pc_cpus,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_ANALYTIC_TRAIN_RUN=$run" "$run/submit.pbs")
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
printf 'Selected teacher: %s\nPrepared training run: %s\n' "$teacher" "$run"
printf 'Training method: %s\n' "$training_method"
if [[ -n $external_test ]]; then printf 'Separate external test: %s\n' "$external_test"; fi
printf 'Submit/resubmit: bash %q\n' "$run/resubmit.sh"
if $submit; then bash "$run/resubmit.sh" | tee "$run/jobid.txt"; fi
