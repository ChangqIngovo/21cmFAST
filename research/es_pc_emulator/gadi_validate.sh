#!/usr/bin/env bash
# Generate audited pairs, then evaluate all neural input arms on held-out seeds.
set -euo pipefail
if (( $# )) && [[ $1 == --help ]]; then
    echo 'Usage: bash research/es_pc_emulator/gadi_validate.sh [--config FILE]'
    echo 'Submits dataset generation and dependent exploratory training.'
    exit 0
fi
options=()
if (( $# )); then
    [[ $# == 2 && $1 == --config ]] || { echo 'Expected --config FILE.' >&2; exit 2; }
    options=(--config "$(realpath "$2")")
fi
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if (( ${#options[@]} == 0 )); then
    options=(--config "$repo/research/es_pc_emulator/validation_dataset.json")
fi
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
export ES_PC_PREPARED_PATH_FILE="$temporary/prepared_path"
bash "$repo/research/es_pc_emulator/gadi_prepare.sh" "${options[@]}" --submit
run=$(cat "$ES_PC_PREPARED_PATH_FILE")
dataset_job=$(cat "$run/job_id.txt")
[[ $dataset_job =~ ^[0-9]+([.][A-Za-z0-9._-]+)?$ ]] || { echo 'Invalid dataset job ID.' >&2; exit 1; }
training_job=$(qsub -W "depend=afterok:$dataset_job" \
    -v "DATASET_RUN=$run,ALLOW_UNCONVERGED_TEACHER=1" \
    "$run/source/research/es_pc_emulator/gadi_train.pbs")
[[ $training_job =~ ^[0-9]+([.][A-Za-z0-9._-]+)?$ ]] || { echo 'Unexpected training job ID.' >&2; exit 1; }
printf '%s\n' "$training_job" > "$run/training_job_id.txt"
printf 'Dataset job: %s\nTraining job: %s\nRun directory: %s\n' "$dataset_job" "$training_job" "$run"
echo 'Training waits for successful dataset generation and residual auditing.'
echo 'This is exploratory accuracy validation; teacher convergence is not certified.'
