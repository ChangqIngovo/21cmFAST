#!/usr/bin/env bash
# Build and optionally submit the pinned large-volume experiment.
set -euo pipefail
export PC_EXPERIMENT=large-scale
export PC_THREADS=${PC_THREADS:-32}
export PC_MEM=${PC_MEM:-320GB}
export PC_WALLTIME=${PC_WALLTIME:-24:00:00}
export PC_QUEUE=${PC_QUEUE:-hugemem}
exec bash "$(dirname "${BASH_SOURCE[0]}")/../pc_adaptive_front/gadi_prepare.sh" "$@"
