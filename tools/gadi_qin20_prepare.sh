#!/usr/bin/env bash
# Prepare an isolated, commit-pinned 21cmFAST environment on a Gadi login node.
# Usage: bash tools/gadi_qin20_prepare.sh [--submit]
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --submit ) ]]; then
    echo 'Usage: bash tools/gadi_qin20_prepare.sh [--submit]' >&2
    exit 2
fi
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
rev=$(git -C "$repo" rev-parse "${PC_REV:-HEAD}^{commit}")
# setuptools_scm requires release tags for valid cache-format version metadata.
# This branch started at the official v4.3.1 release; a tag-less fork/checkout
# otherwise builds as 0.0.1.dev..., which upstream's HDF5 reader cannot handle.
release_tag=v4.3.1
release_commit=72d8a2d23702f1b69fbc85abd25d4592a7bcd3f5
if ! git -C "$repo" rev-parse --verify --quiet "refs/tags/$release_tag^{commit}" >/dev/null; then
    git -C "$repo" fetch --no-tags https://github.com/21cmfast/21cmFAST.git \
        "refs/tags/$release_tag:refs/tags/$release_tag"
fi
if [[ $(git -C "$repo" rev-parse "refs/tags/$release_tag^{commit}") != "$release_commit" ]]; then
    echo 'The v4.3.1 tag does not point at the verified upstream release commit.' >&2
    exit 2
fi
git -C "$repo" merge-base --is-ancestor "$release_commit" "$rev" || {
    echo 'The selected revision does not descend from the verified v4.3.1 base.' >&2
    exit 2
}
project=${PC_PROJECT:-qp00}
run_root=${PC_RUN_ROOT:-/g/data/${project}/${USER}/21cmfast_runs}
mkdir -p "$run_root"
run_dir=$(mktemp -d "$run_root/Qin20_${rev:0:10}_$(date -u +%Y%m%dT%H%M%S)_XXXX")
printf '%s\n' "$rev" > "$run_dir/commit.txt"
printf '%s\n' "$repo" > "$run_dir/repository.txt"
git -C "$repo" worktree add --detach "$run_dir/source" "$rev"
exec > >(tee "$run_dir/prepare.log") 2>&1
printf 'Preparing %s\nCommit: %s\n' "$run_dir" "$rev"

module --force purge
pc_modules=(gcc/12.2.0 fftw3/3.3.10-nci1 gsl/2.7.1)
module load "${pc_modules[@]}"
if [[ -n ${PC_PYTHON_MODULE:-} ]]; then
    module load "$PC_PYTHON_MODULE"
    pc_modules+=("$PC_PYTHON_MODULE")
elif [[ -z ${PC_PYTHON:-} ]]; then
    # The current upstream code requires Python >=3.12. Select from the actual
    # module inventory rather than assuming the old 3.11 environment works.
    mapfile -t python_modules < <(
        module -t avail python3 2>&1 |
        sed -nE 's@^[[:space:]]*(python3/[0-9]+\.[0-9]+\.[0-9]+)(\(.*\))?[[:space:]]*$@\1@p' |
        sort -V
    )
    python_module=''
    for candidate in "${python_modules[@]}"; do
        candidate_version=${candidate#python3/}
        candidate_major=${candidate_version%%.*}
        candidate_minor=${candidate_version#*.}
        candidate_minor=${candidate_minor%%.*}
        if (( candidate_major > 3 || ( candidate_major == 3 && candidate_minor >= 12 ) )); then
            python_module=$candidate
            break
        fi
    done
    if [[ -z $python_module ]]; then
        echo 'No Python >=3.12 module found. Set PC_PYTHON_MODULE or PC_PYTHON to an installed Python >=3.12.' >&2
        exit 2
    fi
    module load "$python_module"
    pc_modules+=("$python_module")
fi
pc_python=${PC_PYTHON:-$(command -v python3)}
"$pc_python" -c 'import sys; assert sys.version_info >= (3,12), "This 21cmFAST branch requires Python >=3.12"'
printf 'PC_MODULES=(' > "$run_dir/runtime.env"
printf ' %q' "${pc_modules[@]}" >> "$run_dir/runtime.env"
printf ' )\n' >> "$run_dir/runtime.env"

# Build on the login node, where pip can reach its package index. Runtime PBS
# jobs then use this complete environment and require no outbound pip access.
export CC=gcc CXX=g++
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2
# CFFI consumes explicit include/library environment variables. Standard Gadi
# module paths are checked on this host; module-provided search paths remain active.
if [[ -f /apps/fftw3/3.3.10-nci1/include/fftw3.h ]]; then
    export PC_FFTW_INCLUDE=/apps/fftw3/3.3.10-nci1/include
fi
if [[ -d /apps/fftw3/3.3.10-nci1/lib ]]; then
    export PC_FFTW_LIBRARY=/apps/fftw3/3.3.10-nci1/lib
fi
if [[ -f /apps/gsl/2.7.1/include/gsl/gsl_rng.h ]]; then
    export PC_GSL_INCLUDE=/apps/gsl/2.7.1/include
fi
if [[ -d /apps/gsl/2.7.1/lib ]]; then
    export PC_GSL_LIBRARY=/apps/gsl/2.7.1/lib
fi
"$pc_python" -m venv "$run_dir/venv"
"$run_dir/venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$run_dir/venv/bin/python" -m pip install "$run_dir/source"
"$run_dir/venv/bin/python" -m pip freeze > "$run_dir/python_packages.txt"
module -t list > "$run_dir/modules.txt" 2>&1
"$run_dir/venv/bin/python" "$run_dir/source/tools/gadi_qin20_compare.py" \
    --prepare --output "$run_dir/results" \
    --threads "${PC_THREADS:-32}" --grid "${PC_GRID:-64}" \
    --box-mpc "${PC_BOX_MPC:-200}" --zmin "${PC_ZMIN:-5.5}" \
    --seed "${PC_SEED:-1234}"
cp "$run_dir/source/tools/gadi_qin20.pbs" "$run_dir/submit.pbs"
(
    cd "$run_dir"
    sha256sum runtime.env python_packages.txt results/run_config.json results/Qin20_ES.toml results/Qin20_PC.toml submit.pbs \
        source/tools/gadi_qin20_compare.py > prepared.sha256
)
touch "$run_dir/PREPARED"
printf '\nPrepared run: %s\nResults: %s/results\n' "$run_dir" "$run_dir"
if [[ ${1:-} == --submit ]]; then
    job=$(qsub -P "$project" -l "ncpus=${PC_THREADS:-32},mem=${PC_MEM:-128GB},walltime=${PC_WALLTIME:-24:00:00}" \
        -v "RUN_DIR=$run_dir" "$run_dir/submit.pbs")
    printf '%s\n' "$job" > "$run_dir/jobid.txt"
    printf 'Submitted: %s\n' "$job"
else
    printf 'Submit: qsub -P %q -v RUN_DIR=%q %q\n' "$project" "$run_dir" "$run_dir/submit.pbs"
fi
