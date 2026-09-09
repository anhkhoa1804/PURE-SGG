#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL=0
if [[ "${1-}" == "--install" ]]; then
  INSTALL=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--install]" >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
echo "git_commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"

if [[ ${INSTALL} -eq 1 ]]; then
  "${PYTHON_BIN}" -m pip install -r "${REPO_DIR}/requirements.txt"
  echo "Core dependencies installed. Optional/dev requirements remain explicit."
else
  echo "Install skipped. Pass --install to install requirements.txt; no data is downloaded."
fi

PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<'PY'
import torch
import transformers
import datasets
import huggingface_hub
import openvocab_rel
print(f"imports=PASS torch={torch.__version__} transformers={transformers.__version__} datasets={datasets.__version__} huggingface_hub={huggingface_hub.__version__}")
print(f"cuda_available={torch.cuda.is_available()} cuda_build={torch.version.cuda}")
PY

echo "RESEARCH_NO1_DATA_ROOT=${RESEARCH_NO1_DATA_ROOT:-${REPO_DIR}/datasets_vg150_clean}"
echo "RESEARCH_NO1_CHECKPOINT_ROOT=${RESEARCH_NO1_CHECKPOINT_ROOT:-${REPO_DIR}/checkpoints}"
echo "RESEARCH_NO1_RUN_ROOT=${RESEARCH_NO1_RUN_ROOT:-${REPO_DIR}/runs}"
echo "RESEARCH_NO1_CACHE_ROOT=${RESEARCH_NO1_CACHE_ROOT:-${REPO_DIR}/caches}"

if "${PYTHON_BIN}" -c 'import pytest' >/dev/null 2>&1; then
  PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m pytest -q \
    "${REPO_DIR}/tests/test_imports.py" "${REPO_DIR}/tests/test_portable_paths.py"
else
  echo "pytest not installed; lightweight smoke tests skipped. Install requirements-dev.txt explicitly to enable them."
fi
