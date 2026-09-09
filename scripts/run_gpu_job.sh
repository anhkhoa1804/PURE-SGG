#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_INDEX=""
BATCH_SIZE=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device-index) DEVICE_INDEX="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${DEVICE_INDEX}" || -z "${BATCH_SIZE}" || -z "${OUTPUT_DIR}" || $# -eq 0 ]]; then
  echo "usage: $0 --device-index N --batch-size N --output-dir PATH -- COMMAND [ARGS...]" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output already exists: ${OUTPUT_DIR}" >&2
  exit 5
fi

"${REPO_DIR}/scripts/gpu_preflight.sh" --device-index "${DEVICE_INDEX}"
mkdir -p "${OUTPUT_DIR}"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"
  echo "git_branch=$(git -C "${REPO_DIR}" branch --show-current)"
  echo "git_dirty_count=$(git -C "${REPO_DIR}" status --porcelain | wc -l)"
  echo "device_index=${DEVICE_INDEX}"
  echo "batch_size=${BATCH_SIZE}"
  printf 'command='; printf '%q ' "$@"; echo
} > "${OUTPUT_DIR}/launch_provenance.txt"

export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
export DEVICE="cuda:0"
export BATCH_SIZE
export OUT_DIR="${OUTPUT_DIR}"
exec "$@"
