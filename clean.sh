#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash clean.sh
# Optional env vars:
#   CONCORD_ENV_NAME     Conda environment name to remove (default: concord)
#   CONCORD_DOWNLOAD_DIR Download directory to delete (default: <repo>/downloads/zenodo_8393793)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONCORD_ENV_NAME:-concord}"
DOWNLOAD_ROOT="${CONCORD_DOWNLOAD_DIR:-${PROJECT_ROOT}/downloads/zenodo_8393793}"
APEX_DIR="${PROJECT_ROOT}/apex"
TS_DIR="${PROJECT_ROOT}/data_processing"
TS_PACKAGE_DIR="${TS_DIR}/ts_package"
TS_BUILD_DIR="${TS_DIR}/build"

safe_rm() {
  local target="$1"
  local label="$2"
  if [[ -z "${target}" || "${target}" == "/" ]]; then
    echo "Refusing to remove empty or root path for ${label}" >&2
    return
  fi
  if [[ -e "${target}" ]]; then
    rm -rf "${target}"
    echo "Removed ${label}: ${target}"
  else
    echo "No ${label} found at ${target}; skipping."
  fi
}

remove_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found on PATH; skipping environment removal."
    return
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"

  if conda info --envs | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Removing conda environment '${ENV_NAME}'..."
    conda env remove -y -n "${ENV_NAME}"
  else
    echo "Conda environment '${ENV_NAME}' not found; skipping."
  fi
}

echo "[1/4] Removing conda environment (if present)..."
remove_conda_env

echo "[2/4] Cleaning cloned NVIDIA Apex repository..."
safe_rm "${APEX_DIR}" "Apex repository"

echo "[3/4] Cleaning tree-sitter build artifacts..."
safe_rm "${TS_PACKAGE_DIR}" "tree-sitter sources"
safe_rm "${TS_BUILD_DIR}" "tree-sitter build outputs"

echo "[4/4] Removing downloaded artifacts..."
safe_rm "${DOWNLOAD_ROOT}" "download directory"

echo "Cleanup complete."
