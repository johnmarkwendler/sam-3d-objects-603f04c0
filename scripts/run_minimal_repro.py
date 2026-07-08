#!/usr/bin/env python
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / ".openresearch" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def file_info(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> None:
    sys.path.append(str(ROOT / "notebook"))
    from inference import Inference, load_image, load_single_mask

    config_path = ROOT / "checkpoints" / "hf" / "pipeline.yaml"
    image_path = ROOT / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567" / "image.png"
    mask_dir = ROOT / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567"
    mask_index = 14
    output_ply = ARTIFACT_DIR / "splat.ply"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    started = time.time()
    inference = Inference(str(config_path), compile=False)
    image = load_image(str(image_path))
    mask = load_single_mask(str(mask_dir), index=mask_index)
    output = inference(image, mask, seed=42)
    output["gs"].save_ply(str(output_ply))
    elapsed = time.time() - started

    gs = output["gs"]
    vertex_count = int(gs.get_xyz.shape[0])
    active_count = int((gs.get_opacity > 0.9).sum().item())

    metadata = {
        "paper": "arXiv:2511.16624",
        "repo": "facebookresearch/sam-3d-objects",
        "task": "README single-object SAM 3D Objects inference",
        "image": str(image_path.relative_to(ROOT)),
        "mask_dir": str(mask_dir.relative_to(ROOT)),
        "mask_index": mask_index,
        "seed": 42,
        "elapsed_seconds": elapsed,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "output": file_info(output_ply),
        "gaussian_vertex_count": vertex_count,
        "gaussian_active_opacity_gt_0_9": active_count,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with output_ply.open("rb") as f:
        header = f.read(4096).decode("utf-8", errors="replace")
    (ARTIFACT_DIR / "splat_header.txt").write_text(header)

    eval_md = f"""# SAM 3D Objects Minimal Reproduction

Result: PASS

Ran the official README single-object example end to end:

- Paper: arXiv:2511.16624
- Code path: `demo.py` / `notebook/inference.py`
- Image: `{metadata["image"]}`
- Mask index: `{mask_index}`
- Seed: `42`
- Output: `{metadata["output"]["path"]}`
- Output size: `{metadata["output"]["size_bytes"]}` bytes
- Gaussian vertices: `{vertex_count}`
- Active opacity > 0.9: `{active_count}`
- Runtime: `{elapsed:.2f}` seconds
- CUDA device: `{metadata["cuda_device_name"]}`

Artifacts:

- `splat.ply`: exported Gaussian splat reconstruction
- `metadata.json`: run metadata and output counts
- `splat_header.txt`: beginning of the PLY file for text inspection
- `run.log`: setup and execution log
"""
    (ARTIFACT_DIR / "EVAL.md").write_text(eval_md)
    print(eval_md)


if __name__ == "__main__":
    main()
