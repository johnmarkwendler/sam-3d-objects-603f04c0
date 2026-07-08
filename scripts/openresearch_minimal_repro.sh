#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACT_DIR="$ROOT/.openresearch/artifacts"
mkdir -p "$ARTIFACT_DIR"

exec > >(tee "$ARTIFACT_DIR/run.log") 2>&1

echo "== SAM 3D Objects minimal reproduction =="
date -u +"started_at_utc=%Y-%m-%dT%H:%M:%SZ"
uname -a
command -v nvidia-smi >/dev/null && nvidia-smi || true

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for gated facebook/sam-3d-objects checkpoints." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_DIR="$ROOT/.orx_env"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Creating Python venv at $ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121}"
export PIP_FIND_LINKS="${PIP_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"

python - <<'PY'
import sys
print("python", sys.version)
PY

python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

python -m pip install "huggingface-hub[cli]<1.0"

# Install the official package and its optional inference dependencies. The
# p3d extra is installed separately because the upstream setup guide notes that
# PyTorch3D's torch dependency resolution is fragile.
python -m pip install -e '.[dev]'
python -m pip install -e '.[p3d]'
python -m pip install -e '.[inference]'

python patching/hydra

if [[ ! -f checkpoints/hf/pipeline.yaml ]]; then
  echo "Downloading SAM 3D Objects checkpoints from Hugging Face"
  rm -rf checkpoints/hf-download
  hf download \
    --repo-type model \
    --local-dir checkpoints/hf-download \
    --max-workers 1 \
    facebook/sam-3d-objects
  rm -rf checkpoints/hf
  mv checkpoints/hf-download/checkpoints checkpoints/hf
  rm -rf checkpoints/hf-download
fi

python scripts/run_minimal_repro.py

date -u +"finished_at_utc=%Y-%m-%dT%H:%M:%SZ"
