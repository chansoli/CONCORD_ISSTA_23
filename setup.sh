#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup.sh
# Optional env vars:
#   CONCORD_ENV_NAME        Conda environment name (default: concord)
#   CONCORD_PYTHON_VERSION  Python version for the environment (default: 3.8.12)
#   CONCORD_DOWNLOAD_DIR    Where to store downloaded artifacts (default: <repo>/downloads/zenodo_8393793)
#   CONCORD_CODEBERT_DIR    Where to store the CodeBERT base model (default: <repo>/downloads/huggingface/codebert-base)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONCORD_ENV_NAME:-concord}"
PYTHON_VERSION="${CONCORD_PYTHON_VERSION:-3.8.12}"
DOWNLOAD_ROOT="${CONCORD_DOWNLOAD_DIR:-${PROJECT_ROOT}/downloads/zenodo_8393793}"
CODEBERT_DIR="${CONCORD_CODEBERT_DIR:-${PROJECT_ROOT}/downloads/huggingface/codebert-base}"
APEX_REPO_URL="https://github.com/NVIDIA/apex.git"
APEX_COMMIT="feae3851a5449e092202a1c692d01e0124f977e4"
export CONDA_OVERRIDE_CUDA=""

echo "[1/7] Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found on PATH. Please install Miniconda/Anaconda and re-run." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
DEFAULT_ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
ENV_PREFIX="${CONCORD_ENV_PREFIX:-${DEFAULT_ENV_PREFIX}}"
if ! mkdir -p "$(dirname "${ENV_PREFIX}")" 2>/dev/null; then
  ENV_PREFIX="${PROJECT_ROOT}/.conda/envs/${ENV_NAME}"
  mkdir -p "$(dirname "${ENV_PREFIX}")"
fi
CONDA_TARGET=(--prefix "${ENV_PREFIX}")
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[2/7] Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION}) at ${ENV_PREFIX}..."
if [ -d "${ENV_PREFIX}" ]; then
  echo "Environment path '${ENV_PREFIX}' already exists; reusing it."
else
  conda create -y "${CONDA_TARGET[@]}" "python=${PYTHON_VERSION}"
fi
echo "[2a/7] Setting LD_PRELOAD for '${ENV_NAME}'..."
conda env config vars set "${CONDA_TARGET[@]}" "LD_PRELOAD=${ENV_PREFIX}/lib/libittnotify.so ${ENV_PREFIX}/lib/libstdc++.so.6"
conda env config vars set "${CONDA_TARGET[@]}" "PYTHONPATH=${PROJECT_ROOT}"

echo "[3/7] Installing Python dependencies..."
conda install -y "${CONDA_TARGET[@]}" pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y "${CONDA_TARGET[@]}" ittapi intel-openmp -c conda-forge
conda install -y "${CONDA_TARGET[@]}" numpy=1.22.4 scipy scikit-learn -c conda-forge

if [ ! -d "${PROJECT_ROOT}/apex" ]; then
  echo "[3a/7] Cloning NVIDIA Apex..."
  git clone "${APEX_REPO_URL}" "${PROJECT_ROOT}/apex"
fi
pushd "${PROJECT_ROOT}/apex" >/dev/null
git checkout "${APEX_COMMIT}"
conda run "${CONDA_TARGET[@]}" pip install -v --disable-pip-version-check --no-cache-dir ./
popd >/dev/null

echo "[4/7] Installing project requirements..."
conda run "${CONDA_TARGET[@]}" pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "[5/7] Building tree-sitter parsers..."
pushd "${PROJECT_ROOT}/data_processing" >/dev/null
conda run "${CONDA_TARGET[@]}" bash build_tree_sitter.sh
popd >/dev/null

echo "[6/7] Downloading pretrained weights and finetuning data..."
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

echo "[7/7] Downloading CodeBERT base model from Hugging Face..."
mkdir -p "${CODEBERT_DIR}"
CODEBERT_MARKER="${CODEBERT_DIR}/config.json"
if [ -f "${CODEBERT_MARKER}" ]; then
  echo "CodeBERT already present at ${CODEBERT_DIR}; skipping download."
else
  DOWNLOAD_SCRIPT="
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path('${CODEBERT_DIR}').resolve()
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id='microsoft/codebert-base',
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f'CodeBERT downloaded to {target}')
"
  conda run "${CONDA_TARGET[@]}" python -c "$DOWNLOAD_SCRIPT"
fi

echo "Setup complete. Activate the environment with: conda activate ${ENV_NAME}"
