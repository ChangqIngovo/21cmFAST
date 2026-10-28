#!/usr/bin/env bash
# Freeze a small same-source ES/PC experiment using an existing native runtime.
set -euo pipefail
submit=false
parent=${PC_RUNTIME_PARENT:-}
while (( $# )); do
    case "$1" in
        --submit) submit=true; shift ;;
        --parent) parent=${2:?Supply a completed runtime parent}; shift 2 ;;
        --help)
            echo 'Usage: bash research/pc_matched/gadi_prepare.sh [--parent RUN] [--submit]'
            echo 'Uses an existing native21cmFAST runtime; no C2-Ray or environment installation.'
            echo 'PC_PYTHON/PC_RUNTIME_ENV override interpreter/modules; explicit PC_PYTHON can run without modules.'
            echo 'Direct finite-volume photon-number RT with native effective atom counts and a single H-opacity absorber.'
            echo 'Defaults: 8^3 cells, 50 cMpc, seeds111/222, z20->4, 40 source intervals, 18 eV photons.'
            echo 'Five RT histories per seed: 6/26 directions, spatial refinements1/2, CFL0.4/0.2/0.1.'
            echo 'PC_DIMENSION/BOX_LEN/SEEDS/Z_START/Z_END/BASE_INTERVALS/REFERENCE_Z override the experiment.'
            echo 'PC_DIRECTIONS/REFINEMENTS/CFL_VALUES/PHOTON_ENERGY_EV select the RT resolution study.'
            echo 'PC_THREADS/MEM/MEMORY_GB/WALLTIME: 16/64GB/48/12:00:00.'
            echo 'PC_MATCHED_RUN_ROOT/PARENT_ROOT/PROJECT/STORAGE/QUEUE override paths and allocation.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- || { echo 'Commit tracked changes before preparing.' >&2; exit 2; }
untracked=$(git -C "$repo" ls-files --others --exclude-standard -- research/pc_matched)
[[ -z $untracked ]] || { echo 'Commit all new pc_matched files before preparing.' >&2; exit 2; }
git -C "$repo" cat-file -e HEAD:research/pc_matched/workflow.py
git -C "$repo" cat-file -e HEAD:research/pc_matched/gadi_run.pbs
revision=$(git -C "$repo" rev-parse HEAD)
root=${PC_MATCHED_RUN_ROOT:-/g/data/qp00/sz4640/21cmfast_runs}
selection_arguments=(select --root "${PC_PARENT_ROOT:-$root}")
[[ -z $parent ]] || selection_arguments+=(--parent "$parent")
[[ -z ${PC_PYTHON:-} ]] || selection_arguments+=(--python "$PC_PYTHON")
[[ -z ${PC_RUNTIME_ENV:-} ]] || selection_arguments+=(--runtime-env "$PC_RUNTIME_ENV")
selection=$(python3 "$repo/research/pc_matched/runtime_select.py" "${selection_arguments[@]}")
mapfile -t selected <<< "$selection"
pc_python=${selected[0]}
parent=${selected[1]:-}
runtime_env=${selected[2]:-}
mkdir -p "$root"
run=$(mktemp -d "$root/matched_pc_${revision:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
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
    type module >/dev/null 2>&1 || { echo 'Runtime requires environment modules, but module is unavailable.' >&2; exit 2; }
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
"$pc_python" -c 'import sys; sys.version_info >= (3, 11) or sys.exit("Selected native Python must be >=3.11; found " + sys.version.split()[0] + " at " + sys.executable + ". Set PC_PYTHON to a compatible existing native runtime.")'
cp "$run/source/research/pc_matched/gadi_run.pbs" "$run/submit.pbs"
cd "$run/source"
"$pc_python" -m research.pc_matched.prepare_run prepare --run "$run"
workers=$("$pc_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threads"])' "$run/experiment_config.json")
command=(qsub -P "${PC_PROJECT:-qp00}" -q "${PC_QUEUE:-normal}"
         -l "ncpus=$workers,mem=${PC_MEM:-64GB},walltime=${PC_WALLTIME:-12:00:00},storage=${PC_STORAGE:-gdata/qp00+scratch/qp00}"
         -v "PC_MATCHED_RUN=$run" "$run/submit.pbs")
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$run/resubmit.sh"
printf '%q ' "${command[@]}" >> "$run/resubmit.sh"
printf '\n' >> "$run/resubmit.sh"
printf 'Prepared matched ES/PC run: %s\nNative Python: %s\n' "$run" "$pc_python"
printf '%q ' "${command[@]}"
printf '\nResubmit after interruption: bash %q\n' "$run/resubmit.sh"
if $submit; then
    "${command[@]}" | tee "$run/jobid.txt"
fi
