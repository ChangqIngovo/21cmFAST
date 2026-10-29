#!/usr/bin/env bash
# Archive continuous native-source RT; no simulations run on the login node.
set -euo pipefail
submit=false
parent=${PC_RUNTIME_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply a completed runtime parent}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_causal_rt/gadi_prepare.sh [--parent RUN] [--submit]'
            echo 'Continuous native-source RT with persistent radiation and ionization state.'
            echo 'PC_CONFIG_JSON supplies any validated scientific configuration.'
            echo 'PC_DIMENSION/BOX_LEN/SEEDS/THREADS control native generation.'
            echo 'Scalar/list workflow defaults accept matching PC_ environment names.'
            echo 'PC_DIRECTIONS/REFINEMENTS/CFL_VALUES/TIME_SUBSTEPS/SOURCE_STRIDES select a Cartesian RT matrix.'
            echo 'PC_NATIVE_INPUTS_JSON supplies full native InputParameters JSON if supported by the workflow.'
            echo 'PC_PYTHON/PC_RUNTIME_ENV or PC_RUNTIME_PARENT select an existing native runtime.'
            echo 'PC_MEM/MEMORY_GB/WALLTIME default to 32GB/24GiB/12:00:00.'
            echo 'PC_CAUSAL_RT_RUN_ROOT/PARENT_ROOT/PROJECT/STORAGE/QUEUE override paths and PBS allocation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
export PC_MEM=${PC_MEM:-32GB}
export PC_MEMORY_GB=${PC_MEMORY_GB:-24}
export PC_WALLTIME=${PC_WALLTIME:-12:00:00}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_causal_rt)
[[ -z $untracked ]] || { echo 'Commit new pc_causal_rt files before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/pc_causal_rt/workflow.py
git -C "$repo" cat-file -e HEAD:research/pc_causal_rt/gadi_run.pbs
revision=$(git -C "$repo" rev-parse HEAD)
root=${PC_CAUSAL_RT_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
selection_arguments=(select --root "${PC_PARENT_ROOT:-$root}")
[[ -z $parent ]] || selection_arguments+=(--parent "$parent")
[[ -z ${PC_PYTHON:-} ]] || selection_arguments+=(--python "$PC_PYTHON")
[[ -z ${PC_RUNTIME_ENV:-} ]] || selection_arguments+=(--runtime-env "$PC_RUNTIME_ENV")
# The discovery script is intentionally Python 3.6 compatible. Load the selected
# environment before importing the Python >=3.12 scientific preparation module.
selection=$(python3 "$repo/research/pc_matched/runtime_select.py" "${selection_arguments[@]}")
mapfile -t selected <<< "$selection"
pc_python=${selected[0]}
parent=${selected[1]:-}
runtime_env=${selected[2]:-}
if [[ -n ${PC_CONFIG_JSON:-} ]]; then
    export PC_CONFIG_JSON=$(realpath "$PC_CONFIG_JSON")
fi
if [[ -n ${PC_NATIVE_INPUTS_JSON:-} ]]; then
    export PC_NATIVE_INPUTS_JSON=$(realpath "$PC_NATIVE_INPUTS_JSON")
fi
mkdir -p "$root"
run=$(mktemp -d "$root/causal_rt_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
run=$(realpath "$run")
[[ $run != *,* && $run != *$'\n'* && $run != *$'\r'* ]] || { echo 'Invalid PBS run path.' >&2; exit 2; }
exec > >(tee "$run/prepare.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run/FAILED_PREPARE"; fi' EXIT
mkdir "$run/source"
git -C "$repo" archive "$revision" | tar -x -C "$run/source"
printf '%s\n' "$revision" > "$run/commit.txt"
printf '%s\n' "$parent" > "$run/runtime_parent.txt"
printf '%s\n' "$pc_python" > "$run/python_path.txt"
if [[ -n $runtime_env ]]; then
    cp "$runtime_env" "$run/runtime.env"
else
    printf 'PC_MODULES=()\n' > "$run/runtime.env"
fi
PC_MODULES=()
source "$run/runtime.env"
if (( ${#PC_MODULES[@]} )); then
    for pc_build_module in gcc/12.2.0 fftw3/3.3.10-nci1 gsl/2.7.1; do
        pc_module_present=false
        for pc_existing_module in "${PC_MODULES[@]}"; do
            if [[ ${pc_existing_module%%/*} == "${pc_build_module%%/*}" ]]; then
                pc_module_present=true
                break
            fi
        done
        if ! $pc_module_present; then PC_MODULES+=("$pc_build_module"); fi
    done
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
export CC=gcc CXX=g++ MPLBACKEND=Agg
"$pc_python" -c 'import sys; sys.version_info >= (3, 12) or sys.exit("Selected native Python must be >=3.12; set PC_PYTHON to a compatible existing runtime.")'
"$pc_python" -c 'import cffi, setuptools' > "$run/build_dependencies.log" 2>&1 || {
    cat "$run/build_dependencies.log" >&2
    echo 'The existing runtime needs cffi and setuptools for the frozen native build. No job was submitted.' >&2
    exit 2
}
cp "$run/source/research/pc_causal_rt/gadi_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_causal_rt.prepare_run prepare --run "$run"
workers=$("$pc_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threads"])' "$run/experiment_config.json")
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$workers,mem=$PC_MEM,walltime=$PC_WALLTIME,storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_CAUSAL_RT_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared continuous native-source RT run: %s\nNative Python: %s\n' "$run" "$pc_python"
printf '%q ' "${command[@]}"
printf '\nResume after interruption: bash %q\n' "$run/resubmit.sh"
if $submit; then "${command[@]}" | tee "$run/jobid.txt"; fi
