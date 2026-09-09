#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="${PYTHON:-python3}"; fi

echo "git_commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"
echo "git_branch=$(git -C "${REPO_DIR}" branch --show-current)"
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
  echo "git_state=DIRTY"
  git -C "${REPO_DIR}" status --short
else
  echo "git_state=CLEAN"
fi
"${PYTHON_BIN}" --version
"${REPO_DIR}/scripts/collect_environment.sh"

for item in \
  "${RESEARCH_NO1_DATA_ROOT:-${REPO_DIR}/datasets_vg150_clean}" \
  "${RESEARCH_NO1_CHECKPOINT_ROOT:-${REPO_DIR}/checkpoints}" \
  "${RESEARCH_NO1_RUN_ROOT:-${REPO_DIR}/runs}"; do
  [[ -e "${item}" ]] && echo "root=PASS ${item}" || echo "root=MISSING ${item}"
done

exec "${PYTHON_BIN}" "${REPO_DIR}/scripts/verify_data_manifest.py" "$@"
