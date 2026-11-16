#!/usr/bin/env bash
# Freeze the paired collection workflow and optionally submit it on Gadi.
set -euo pipefail
campaign=
previous=
run=
submit=false
relative=research/pc_analytic_emulator/completion_20261116/reproduction
while (( $# )); do
    case "$1" in
        --campaign-root) campaign=${2:?Supply the completed campaign root}; shift 2 ;;
        --previous-model) previous=${2:?Supply the previous model directory}; shift 2 ;;
        --output) run=${2:?Supply a new prepared run directory}; shift 2 ;;
        --submit) submit=true; shift ;;
        --help)
            echo 'Usage: bash research/pc_analytic_emulator/completion_20261116/reproduction/gadi_train_and_validate.sh --campaign-root ROOT --previous-model MODEL --output NEW_RUN [--submit]'
            echo 'ROOT contains training_collection and the completed blind/transfer campaigns from generate_teachers.py.'
            echo 'PC_PYTHON must identify an existing absolute Python >=3.11 interpreter with the native simulation dependencies.'
            echo 'PC_RUNTIME_ENV optionally names a shell file defining PC_MODULES=(...) for this interpreter; no modules are guessed.'
            echo 'Defaults: PC_WORKERS=2 PC_THREADS=2 PC_NCPUS=4 PC_MEM=16GB PC_WALLTIME=24:00:00 PC_RAYS=4096.'
            echo 'PC_FRAME_CACHE_MIB=512 caps cached arrays per worker; set 1024 for this expanded collection, or 0 to disable.'
            echo 'This freezes committed source, trains two predeclared candidates, freezes validation selection, then evaluates blind/transfer data.'
            echo 'Use the generated resubmit.sh to resume the same frozen run.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n $campaign && -n $previous && -n $run ]] || {
    echo 'Supply --campaign-root, --previous-model and --output.' >&2; exit 2;
}
campaign=$(realpath "$campaign")
previous=$(realpath "$previous")
run=$(realpath -m "$run")
for path in "$campaign" "$previous" "$run"; do
    [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || {
        echo 'PBS paths must be absolute without spaces, commas or shell metacharacters.' >&2; exit 2;
    }
done
[[ ! -e $run ]] || { echo 'Output exists; use its resubmit.sh or a new --output.' >&2; exit 2; }
case "$run/" in "$campaign/"*|"$previous/"*) echo 'Output must be outside the campaign and model archives.' >&2; exit 2 ;; esac
[[ -f $campaign/training_collection/collection_manifest.json && -f $previous/model.json ]] || {
    echo 'Missing training collection manifest or previous model.json.' >&2; exit 2;
}
pc_python=${PC_PYTHON:?Set PC_PYTHON to the existing native environment interpreter}
[[ $pc_python = /* && -x $pc_python ]] || { echo 'PC_PYTHON must be an absolute executable.' >&2; exit 2; }
pc_workers=${PC_WORKERS:-2}
pc_threads=${PC_THREADS:-2}
for value in "$pc_workers" "$pc_threads"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'Workers and threads must be positive integers.' >&2; exit 2; }
done
pc_cpus=${PC_NCPUS:-$((pc_workers * pc_threads))}
pc_rays=${PC_RAYS:-4096}
pc_cache_mib=${PC_FRAME_CACHE_MIB:-512}
[[ $pc_cache_mib =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'PC_FRAME_CACHE_MIB must be a nonnegative integer.' >&2; exit 2; }
[[ $pc_cpus =~ ^[1-9][0-9]*$ && $pc_rays =~ ^[1-9][0-9]*$ ]] && (( pc_cpus >= pc_workers * pc_threads && pc_rays >= 2 )) || {
    echo 'PC_NCPUS must cover workers*threads; PC_RAYS must be an integer >=2.' >&2; exit 2;
}
pc_mem=${PC_MEM:-16GB}
pc_walltime=${PC_WALLTIME:-24:00:00}
[[ $pc_mem =~ ^[1-9][0-9]*(MB|GB|TB)$ ]] || { echo 'Invalid PC_MEM.' >&2; exit 2; }
[[ $pc_walltime =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && $pc_walltime != 00:00:00 ]] || { echo 'Invalid PC_WALLTIME.' >&2; exit 2; }
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
[[ -z $(git -C "$repo" ls-files --others --exclude-standard -- research) ]] || {
    echo 'Commit research files before preparing.' >&2; exit 2;
}
for name in train_and_validate.py gadi_train_and_validate.pbs gadi_train_and_validate.sh; do
    git -C "$repo" cat-file -e "HEAD:$relative/$name" || { echo "Missing committed source: $name" >&2; exit 2; }
done
revision=$(git -C "$repo" rev-parse HEAD)
mkdir -p "$run/source" "$run/pbs_logs"
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
if [[ -n ${PC_RUNTIME_ENV:-} ]]; then
    cp "$PC_RUNTIME_ENV" "$run/runtime.env"
else
    printf 'PC_MODULES=()\n' > "$run/runtime.env"
fi
cp "$run/source/$relative/gadi_train_and_validate.pbs" "$run/submit.pbs"
{
    printf 'pc_campaign=%q\npc_previous=%q\n' "$campaign" "$previous"
    printf 'pc_workers=%q\npc_threads=%q\npc_cpus=%q\npc_rays=%q\n' "$pc_workers" "$pc_threads" "$pc_cpus" "$pc_rays"
    printf 'pc_cache_mib=%q\n' "$pc_cache_mib"
} > "$run/settings.env"
PC_MODULES=()
source "$run/runtime.env"
if (( ${#PC_MODULES[@]} )); then
    module --force purge
    module load "${PC_MODULES[@]}"
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$pc_python" -c 'import sys; print(sys.version); sys.exit(sys.version_info < (3, 11))' > "$run/python_version.txt"
cd "$run"
# Guard every archived byte, runtime shell configuration and interpreter binary.
find source -type f -print0 | sort -z | xargs -0 sha256sum > source.sha256
sha256sum "$pc_python" > interpreter.sha256
sha256sum commit.txt python_path.txt python_version.txt runtime.env settings.env submit.pbs source.sha256 interpreter.sha256 > prepared.sha256
printf 'Frozen committed training and evaluation workflow.\n' > PREPARED
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$pc_cpus,mem=$pc_mem,walltime=$pc_walltime,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_COMPLETION_RUN=$run" "$run/submit.pbs")
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'pc_run=%q\ncd "$pc_run"\n' "$run"
    printf 'pc_stamp=$(date -u +%%Y%%m%%dT%%H%%M%%S)_$$\n'
    printf '%q ' "${command[0]}"
    printf '%s ' '-o "$pc_run/pbs_logs/output_${pc_stamp}.log" -e "$pc_run/pbs_logs/error_${pc_stamp}.log"'
    printf '%q ' "${command[@]:1}"
    printf '\n'
} > resubmit.sh
printf 'Prepared: %s\nResults: %s/results\nResubmit: bash %s/resubmit.sh\n' "$run" "$run" "$run"
if $submit; then bash "$run/resubmit.sh" | tee "$run/jobid.txt"; fi
