#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

echo "# Environment capture"
echo
echo "Captured UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo '```text'
echo "repository=${REPO_DIR}"
echo "git_commit=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
echo "git_branch=$(git -C "${REPO_DIR}" branch --show-current 2>/dev/null || echo UNKNOWN)"
echo "git_dirty_count=$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l)"
echo "os=$(awk -F= '/^PRETTY_NAME=/{gsub(/\"/,"",$2); print $2}' /etc/os-release 2>/dev/null || uname -s)"
echo "kernel=$(uname -srmo)"
echo "python_bin=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import platform

print(f"python={platform.python_version()}")
for package in ("torch", "transformers", "datasets", "huggingface-hub", "numpy", "pytest"):
    try:
        print(f"{package}={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
try:
    import torch
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_probe_error={type(exc).__name__}: {exc}")
PY
for name in RESEARCH_NO1_DATA_ROOT RESEARCH_NO1_CHECKPOINT_ROOT RESEARCH_NO1_RUN_ROOT RESEARCH_NO1_CACHE_ROOT DATA_ROOT HF_HOME CUDA_VISIBLE_DEVICES; do
  printf '%s=%s\n' "${name}" "${!name-UNSET}"
done
echo
echo "filesystems:"
df -hT "${REPO_DIR}" "${RESEARCH_NO1_DATA_ROOT:-${REPO_DIR}/datasets_vg150_clean}" 2>&1 || true
echo
echo "nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1 || true
else
  echo "NOT_INSTALLED"
fi
echo '```'
