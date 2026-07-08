#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACT_DIR="$ROOT/.openresearch/artifacts"
mkdir -p "$ARTIFACT_DIR"

exec > >(tee "$ARTIFACT_DIR/run.log") 2>&1

echo "== SAM 3D Body VAR Pipeline =="
date -u +"started_at_utc=%Y-%m-%dT%H:%M:%SZ"
uname -a
command -v nvidia-smi >/dev/null && nvidia-smi || true

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for gated facebook/sam-3d-body checkpoints." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
ENV_DIR="$ROOT/.orx_env"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Creating Python venv at $ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# pkg_resources is used by detectron2; newer setuptools (>=81) removed it.
# Pin setuptools<81 to keep pkg_resources available.
python -m pip install "setuptools<81"

python - <<'PY'
import sys
print("python", sys.version)
PY

# Install PyTorch (CUDA 12.1)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

python -m pip install "huggingface-hub[cli]<1.0"

# Install numpy + cython first (needed for building xtcocotools from source)
python -m pip install numpy cython

# Install SAM 3D Body dependencies (packages that build from source use --no-build-isolation)
python -m pip install \
  pytorch-lightning pyrender opencv-python-headless yacs scikit-image \
  einops timm dill pandas rich hydra-core hydra-submitit-launcher \
  hydra-colorlog pyrootutils webdataset chump "networkx==3.2.1" roma \
  joblib seaborn appdirs jsonlines loguru optree \
  fvcore trimesh plotly kaleido

# Build-from-source packages (need numpy at build time, so --no-build-isolation)
python -m pip install --no-build-isolation xtcocotools pycocotools

# Install detectron2 (pinned, built from source against installed torch)
# The host driver (CUDA 13.2) is backward-compatible with CUDA 12.1 toolkit, but
# torch's _check_cuda_version rejects the mismatch. Patch it to a no-op before building.
export TORCH_CUDA_ARCH_LIST="8.0"
export FORCE_CUDA=1
TORCH_CPP_EXT=$(python -c "import torch.utils.cpp_extension as ce; print(ce.__file__)")
sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE.*/pass  # CUDA version check disabled/' "$TORCH_CPP_EXT"
python -m pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" \
  --no-build-isolation --no-deps

# Install MoGe (for FOV estimation)
python -m pip install "git+https://github.com/microsoft/MoGe.git" || \
  echo "MoGe install failed — continuing without FOV estimator"

# Clone the SAM 3D Body repo
SAM3D_DIR="$ROOT/sam-3d-body"
if [[ ! -d "$SAM3D_DIR" ]]; then
  echo "Cloning sam-3d-body repo..."
  git clone https://github.com/facebookresearch/sam-3d-body.git "$SAM3D_DIR"
fi

# Download checkpoints
CKPT_DIR="$ROOT/checkpoints/sam-3d-body-dinov3"
if [[ ! -f "$CKPT_DIR/model.ckpt" ]]; then
  echo "Downloading SAM 3D Body checkpoints from Hugging Face..."
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam-3d-body-dinov3', local_dir='$CKPT_DIR')
"
fi

# Run the VAR pipeline
export SAM3D_DIR="$SAM3D_DIR"
export SAM3D_CKPT="$CKPT_DIR/model.ckpt"
export SAM3D_MHR_PATH="$CKPT_DIR/assets/mhr_model.pt"

python scripts/run_var_pipeline.py

date -u +"finished_at_utc=%Y-%m-%dT%H:%M:%SZ"
