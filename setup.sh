#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup.sh
# Optional env vars:
#   CONCORD_ENV_NAME        Conda environment name (default: concord)
#   CONCORD_PYTHON_VERSION  Python version for the environment (default: 3.8.12)
#   CONCORD_DOWNLOAD_DIR    Where to store downloaded artifacts (default: <repo>/downloads/zenodo_8393793)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONCORD_ENV_NAME:-concord}"
PYTHON_VERSION="${CONCORD_PYTHON_VERSION:-3.8.12}"
DOWNLOAD_ROOT="${CONCORD_DOWNLOAD_DIR:-${PROJECT_ROOT}/downloads/zenodo_8393793}"
ENV_PREFIX=""
APEX_REPO_URL="https://github.com/NVIDIA/apex.git"
APEX_COMMIT="feae3851a5449e092202a1c692d01e0124f977e4"

echo "[1/6] Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found on PATH. Please install Miniconda/Anaconda and re-run." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[2/6] Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
if conda info --envs | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
echo "[2a/6] Setting LD_PRELOAD for '${ENV_NAME}'..."
conda env config vars set --name "${ENV_NAME}" "LD_PRELOAD=${ENV_PREFIX}/lib/libstdc++.so.6"
conda env config vars set --name "${ENV_NAME}" "PYTHONPATH=${PROJECT_ROOT}"

echo "[3/6] Installing Python dependencies..."
conda install -y -n "${ENV_NAME}" pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y -n "${ENV_NAME}" numpy=1.22.4 scipy scikit-learn

if [ ! -d "${PROJECT_ROOT}/apex" ]; then
  echo "[3a/6] Cloning NVIDIA Apex..."
  git clone "${APEX_REPO_URL}" "${PROJECT_ROOT}/apex"
fi
pushd "${PROJECT_ROOT}/apex" >/dev/null
git checkout "${APEX_COMMIT}"
conda run -n "${ENV_NAME}" pip install -v --disable-pip-version-check --no-cache-dir ./
popd >/dev/null

echo "[4/6] Installing project requirements..."
conda run -n "${ENV_NAME}" pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "[5/6] Building tree-sitter parsers..."
pushd "${PROJECT_ROOT}/data_processing" >/dev/null
conda run -n "${ENV_NAME}" bash build_tree_sitter.sh
popd >/dev/null

echo "[6/6] Downloading pretrained weights and finetuning data..."
mkdir -p "${DOWNLOAD_ROOT}"

FINETUNE_ZIP="${DOWNLOAD_ROOT}/finetune_data.zip"
PRETRAIN_ZIP="${DOWNLOAD_ROOT}/concord_pretrained_ckpt.zip"

download_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -f "${dest}" ]; then
    echo "File already exists, skipping download: ${dest}"
  else
    curl -L "${url}" -o "${dest}"
  fi
}

download_if_missing "https://zenodo.org/records/8393793/files/finetune_data.zip?download=1" "${FINETUNE_ZIP}"
download_if_missing "https://zenodo.org/records/8393793/files/concord_pretrained_ckpt.zip?download=1" "${PRETRAIN_ZIP}"

unzip -o "${FINETUNE_ZIP}" -d "${DOWNLOAD_ROOT}/finetune_data"
unzip -o "${PRETRAIN_ZIP}" -d "${DOWNLOAD_ROOT}/concord_pretrained_ckpt"

echo "Setup complete. Activate the environment with: conda activate ${ENV_NAME}"
