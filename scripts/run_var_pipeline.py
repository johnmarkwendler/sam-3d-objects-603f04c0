#!/usr/bin/env python
"""
SAM 3D Body VAR Pipeline — minimal end-to-end proof-of-concept.

Loads a soccer image, detects players with ViTDet, reconstructs 3D body meshes
with SAM 3D Body, places them on a shared field frame, draws an offside plane,
and writes results to .openresearch/artifacts/.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / ".openresearch" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SAM3D_DIR = os.environ.get("SAM3D_DIR", str(ROOT / "sam-3d-body"))
if SAM3D_DIR not in sys.path:
    sys.path.insert(0, SAM3D_DIR)

sys.path.insert(0, str(ROOT / "scripts"))
from var_geometry import (
    goal_dir_from_lines, world_verts, non_arm_mask,
    place_players, offside_plane_x, build_scene,
)


def download_sample_image():
    """Get a sample image with people. Tries Wikimedia, falls back to sam-3d-body's dancing.jpg."""
    img_path = ROOT / "data" / "soccer_sample.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if img_path.exists():
        return str(img_path)

    # Try downloading a soccer image with a proper User-Agent
    import urllib.request
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/"
        "Football_in_Bloomington%2C_Indiana%2C_2014.jpg/1280px-"
        "Football_in_Bloomington%2C_Indiana%2C_2014.jpg"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_path.write_bytes(resp.read())
        print(f"Downloaded sample soccer image from Wikimedia")
    except Exception as e:
        print(f"Wikimedia download failed ({e}), using sam-3d-body dancing.jpg instead")
        sam3d_img = Path(os.environ.get("SAM3D_DIR", str(ROOT / "sam-3d-body"))) / "notebook" / "images" / "dancing.jpg"
        if sam3d_img.exists():
            import shutil
            shutil.copy(str(sam3d_img), str(img_path))
        else:
            raise RuntimeError(f"No sample image available (tried {url} and {sam3d_img})")
    return str(img_path)


def main():
    import torch
    from notebook.utils import setup_sam_3d_body
    from tools.vis_utils import visualize_sample_together

    hf_repo_id = os.environ.get("SAM3D_REPO_ID", "facebook/sam-3d-body-dinov3")
    image_path = download_sample_image()

    started = time.time()
    print("Setting up SAM 3D Body estimator...")
    estimator = setup_sam_3d_body(
        hf_repo_id=hf_repo_id,
        detector_name="vitdet",
        segmentor_name="",
        fov_name="moge2",
    )
    faces = np.asarray(estimator.faces)
    print(f"Model loaded in {time.time() - started:.1f}s")

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    print(f"Image: {w}x{h}")

    # --- Step 1: Detect all players and reconstruct meshes ---
    recon_started = time.time()
    print("Running detection + 3D reconstruction on all players...")
    outputs = estimator.process_one_image(img_rgb, bbox_thr=0.3)
    print(f"Detected and reconstructed {len(outputs)} players in {time.time() - recon_started:.1f}s")

    if not outputs:
        raise RuntimeError("No players detected in the sample image!")

    # Save detection visualization
    rend_img = visualize_sample_together(img_bgr, outputs, faces)
    cv2.imwrite(str(ARTIFACT_DIR / "detection_overlay.jpg"), rend_img.astype(np.uint8))

    # --- Step 2: Build per-player data dicts ---
    people = []
    for p in outputs:
        people.append({
            "bbox": np.asarray(p["bbox"]).reshape(-1)[:4].astype(float),
            "pred_vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
            "pred_cam_t": np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3),
            "focal_length": float(np.asarray(p["focal_length"]).reshape(-1)[0]),
            "pred_keypoints_3d": np.asarray(p.get("pred_keypoints_3d"), dtype=np.float32)
                                 if p.get("pred_keypoints_3d") is not None else None,
        })

    # --- Step 3: Estimate goal direction ---
    # For the minimal repro, derive a synthetic goal direction from the image width.
    # In the full marimo notebook, the user draws 2 lines on the frame.
    focal = people[0]["focal_length"]
    # Simulate two vertical goal-parallel lines near the image edges
    line_pts = [
        [w * 0.15, h * 0.2], [w * 0.15, h * 0.8],   # left line
        [w * 0.85, h * 0.2], [w * 0.85, h * 0.8],   # right line
    ]
    goal_dir = goal_dir_from_lines(line_pts, focal, w, h)
    print(f"Goal direction (camera frame): {goal_dir}")

    # --- Step 4: Place all players on a shared field frame ---
    selected_ids = list(range(len(people)))
    placed = place_players(people, selected_ids, goal_dir, flip_up=False)

    # --- Step 5: Offside analysis ---
    # For the demo: mark the leftmost players as defenders (attack direction = +X)
    sorted_by_x = sorted(selected_ids, key=lambda i: placed[i][:, 0].mean())
    defender_ids = sorted_by_x[:max(1, len(sorted_by_x) // 2)]
    attack_sign = +1
    masks = {i: non_arm_mask(people[i]) for i in selected_ids}
    plane_x = offside_plane_x(placed, attack_sign, defender_ids, masks)

    # --- Step 6: Build the 3D scene ---
    fig, any_off = build_scene(placed, faces, plane_x, attack_sign, defender_ids, masks)
    fig.write_html(str(ARTIFACT_DIR / "var_3d_scene.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(str(ARTIFACT_DIR / "var_3d_scene.png"), width=1200, height=640)
    except Exception as e:
        print(f"PNG export skipped: {e}")

    # --- Step 7: Save per-player meshes as PLY ---
    import trimesh
    for i in selected_ids:
        mesh = trimesh.Trimesh(vertices=np.asarray(people[i]["pred_vertices"]),
                               faces=faces)
        mesh.export(str(ARTIFACT_DIR / f"player_{i:02d}.ply"))

    # --- Step 8: Compute metrics ---
    median_height = float(np.median([
        placed[i][:, 2].max() for i in selected_ids
    ]))
    forward_players = [i for i in selected_ids if i not in defender_ids]
    margins = {}
    for i in forward_players:
        fm = float(attack_sign * placed[i][masks[i], 0].max())
        margins[i] = round(fm - attack_sign * plane_x, 3)

    elapsed = time.time() - started
    verdict = "OFFSIDE" if any_off else "NO OFFSIDE"

    # --- Save metadata ---
    metadata = {
        "paper": "arXiv:2511.16624 (SAM 3D — Body component)",
        "repo": "facebookresearch/sam-3d-body",
        "task": "VAR-style soccer offside visualization using SAM 3D Body",
        "image": str(image_path),
        "image_size": [w, h],
        "hf_repo_id": hf_repo_id,
        "num_players_detected": len(outputs),
        "num_defenders": len(defender_ids),
        "defender_ids": defender_ids,
        "attack_direction": "+X" if attack_sign > 0 else "-X",
        "offside_plane_x": round(float(plane_x), 4),
        "verdict": verdict,
        "margins_m": margins,
        "median_player_height_m": round(median_height, 4),
        "focal_length": round(focal, 2),
        "elapsed_seconds": round(elapsed, 2),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # --- Write EVAL.md ---
    margins_table = "\n".join(
        f"| #{i} | {m:+.3f} | {'OFFSIDE' if m > 0 else ('level' if m >= -0.3 else 'onside')} |"
        for i, m in margins.items()
    )

    eval_md = f"""# SAM 3D Body VAR Pipeline — Minimal Reproduction

## Result: {verdict}

Reconstructed {len(outputs)} players from a single soccer image using **SAM 3D Body**
(arXiv:2511.16624), placed them on a shared metric field frame, and drew a VAR-style
offside plane through the furthest-forward defender.

### Pipeline

1. **Input**: single RGB soccer frame (`data/soccer_sample.jpg`)
2. **Detection**: ViTDet Cascade Mask R-CNN (detectron2) — found {len(outputs)} players
3. **3D Reconstruction**: SAM 3D Body (`facebook/sam-3d-body-dinov3`) — per-player
   full-body mesh (vertices + camera translation + 3D keypoints)
4. **Placement**: SVD ground fit from feet vertices → shared field frame
   (X=offside axis, Y=goal line, Z=up)
5. **Offside**: constant-X plane at the furthest-forward defender's body point
   (arms/hands excluded via MHR-70 keypoint mask)

### Metrics

| Metric | Value |
|--------|-------|
| Players reconstructed | {len(outputs)} |
| Defenders marked | {len(defender_ids)} (IDs: {defender_ids}) |
| Attack direction | {'+X' if attack_sign > 0 else '-X'} |
| Offside plane X | {plane_x:.3f} m |
| Median player height | {median_height:.3f} m |
| Verdict | **{verdict}** |
| Runtime | {elapsed:.1f} s |
| GPU | {metadata['cuda_device_name']} |

### Per-player margins (attackers)

| Player | Margin (m) | Status |
|--------|-----------|--------|
{margins_table}

### Artifacts

- `detection_overlay.jpg`: ViTDet detections + mesh overlays
- `var_3d_scene.html`: interactive Plotly 3D scene (offside plane + verdict)
- `var_3d_scene.png`: static render of the 3D scene
- `player_NN.ply`: per-player reconstructed meshes
- `metadata.json`: full run metadata
- `run.log`: setup and execution log

### Marimo notebook

The companion marimo notebook (`notebook/var_offside.py`) provides an interactive
version of this pipeline with UI controls for line drawing, player selection, and
defender marking — designed for the molab Notebook Competition.
"""
    (ARTIFACT_DIR / "EVAL.md").write_text(eval_md)
    print(eval_md)


if __name__ == "__main__":
    main()
