#!/usr/bin/env bash
set -euo pipefail

DEVICE_INDEX=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device-index) DEVICE_INDEX="$2"; shift 2 ;;
    *) echo "usage: $0 [--device-index N]" >&2; exit 2 ;;
  esac
done

if ! [[ "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --device-index must be a non-negative integer" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; refusing GPU launch" >&2
  exit 3
fi

# Mandatory visible preflight, not a CUDA workload.
nvidia-smi
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | grep -qx "${DEVICE_INDEX}"; then
  echo "ERROR: GPU index ${DEVICE_INDEX} does not exist" >&2
  exit 3
fi

ACTIVE="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "${ACTIVE//[[:space:]]/}" && "${ACTIVE}" != "No running processes found" ]]; then
  echo "ERROR: active GPU compute process(es) detected; refusing to compete:" >&2
  echo "${ACTIVE}" >&2
  exit 4
fi

echo "GPU preflight PASS: device ${DEVICE_INDEX} exists and no compute process is reported."
