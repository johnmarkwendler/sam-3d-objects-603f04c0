import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return mo,


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🥅 VAR Offside Visualizer — Powered by SAM 3D Body

        **Turn a single soccer photo into a 3D offside verdict.**

        This notebook brings the core contribution of **SAM 3D** (arXiv 2511.16624) to
        life: reconstructing full 3D human body meshes from a single 2D image — and then
        uses those meshes to build a VAR-style offside visualization.

        ## The Paper in 30 Seconds

        SAM 3D is Meta Superintelligence Labs' foundation model for single-image 3D
        reconstruction. This notebook focuses on **SAM 3D Body** — the human mesh
        recovery component — which reconstructs a full-body 3D mesh (pose, shape, and
        camera-space translation) from a single RGB image, using a DINOv3 backbone and the
        Momentum Human Rig (MHR) representation.

        ## The VAR Workflow

        1. **Upload** a soccer photo
        2. **Detect** all players (ViTDet)
        3. **Reconstruct** 3D body meshes (SAM 3D Body)
        4. **Draw goal-parallel lines** on the frame (fixes the offside axis)
        5. **Select players** and **mark defenders**
        6. **Build** a 3D scene with a draggable offside plane → **verdict**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Setup & Configuration

        ### Hugging Face Access

        SAM 3D Body checkpoints are **gated** on Hugging Face. You need:

        1. A Hugging Face account
        2. Request access at [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
        3. A user access token from your [settings page](https://huggingface.co/settings/tokens)

        Paste your token below (it stays in this session and is never stored).
        If your environment already has `HF_TOKEN` set (e.g. a molab secret), you can
        skip this step.
        """
    )
    return


@app.cell
def _():
    import os
    import sys
    import subprocess
    import importlib
    import importlib.util
    import numpy as np

    SAM3D_DIR = os.environ.get("SAM3D_DIR", "/root/sam-3d-body")
    if not os.path.isdir(SAM3D_DIR):
        SAM3D_DIR = "/tmp/sam-3d-body"
    if not os.path.isdir(SAM3D_DIR):
        print("Cloning sam-3d-body repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/facebookresearch/sam-3d-body.git", SAM3D_DIR],
            check=True,
        )
    if SAM3D_DIR not in sys.path:
        sys.path.insert(0, SAM3D_DIR)

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return SAM3D_DIR, importlib, np, os, subprocess, sys


@app.cell
def _(mo, os):
    _env_token = os.environ.get("HF_TOKEN", "")
    hf_token = mo.ui.text(
        kind="password",
        label="Hugging Face token" if not _env_token else "Hugging Face token (already set via HF_TOKEN env — override if needed)",
        placeholder="hf_...",
        value=_env_token,
        full_width=True,
    )
    hf_token
    return hf_token,


@app.cell
def _(hf_token, mo):
    from huggingface_hub import login

    _result = mo.md("⚠️ Enter your Hugging Face token above, then continue.").callout(kind="warn")
    if hf_token.value:
        login(token=hf_token.value, add_to_git_credential=False)
        _result = mo.md("✅ Authenticated with Hugging Face.").callout(kind="success")
    _result
    return login,


@app.cell
def _(mo):
    hf_repo = mo.ui.text(
        value="facebook/sam-3d-body-dinov3",
        label="HuggingFace model repo",
    )
    mo.vstack([mo.md("### Model checkpoint"), hf_repo])
    return hf_repo,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Load SAM 3D Body

        This loads the DINOv3-backbone SAM 3D Body model, ViTDet human detector, and
        MoGe2 FOV estimator. The model stays warm after first load — subsequent
        reconstructions are fast.
        """
    )
    return


@app.cell
def _(SAM3D_DIR, hf_repo, importlib, mo, os):
    import torch

    _utils_path = os.path.join(SAM3D_DIR, "notebook", "utils.py")
    _spec = importlib.util.spec_from_file_location("sam3d_notebook_utils", _utils_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    setup_sam_3d_body = _mod.setup_sam_3d_body

    load_btn = mo.ui.run_button(label="Load SAM 3D Body model")
    load_btn
    return load_btn, setup_sam_3d_body, torch


@app.cell
def _(hf_repo, load_btn, mo, setup_sam_3d_body, torch):
    estimator = None
    faces = None
    if load_btn.value:
        mo.mpl.clear()
        with mo.status.spinner(title="Loading SAM 3D Body..."):
            estimator = setup_sam_3d_body(hf_repo_id=hf_repo.value)
            faces = np.asarray(estimator.faces)
        mo.md(f"✅ Model loaded on **{torch.cuda.get_device_name(0)}** "
              f"({len(faces)} faces in body mesh)").callout(kind="success")
    else:
        mo.md("Click **Load SAM 3D Body model** to begin.").callout(kind="info")
    return estimator, faces


@app.cell
def _(mo):
    mo.md(r"""## 2. Upload a Soccer Image""")
    return


@app.cell
def _(mo):
    import cv2

    image_upload = mo.ui.file(
        label="Upload a soccer photo (JPG/PNG)",
        filetypes=[".jpg", ".jpeg", ".png"],
    )
    image_upload
    return cv2, image_upload


@app.cell
def _(cv2, image_upload, mo, np):
    img_bgr = None
    img_rgb = None
    if image_upload.value:
        raw = image_upload.value[0].contents
        arr = np.frombuffer(raw, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mo.image(img_rgb, caption=f"{img_bgr.shape[1]}×{img_bgr.shape[0]}")
    else:
        mo.md("_Upload an image to continue._")
    return img_bgr, img_rgb


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Detect Players (GPU)

        ViTDet Cascade Mask R-CNN finds every person in the frame. Adjust the confidence
        threshold if you're getting false positives or missing players.
        """
    )
    return


@app.cell
def _(mo):
    conf_slider = mo.ui.slider(0.0, 0.9, value=0.3, step=0.05, label="Detection confidence")
    conf_slider
    return conf_slider,


@app.cell
def _(conf_slider, estimator, img_rgb, mo):
    people = []
    detect_btn = mo.ui.run_button(label="Detect players")
    detect_btn
    if detect_btn.value and estimator is not None and img_rgb is not None:
        with mo.status.spinner(title="Detecting players..."):
            outputs = estimator.process_one_image(img_rgb, bbox_thr=conf_slider.value)
        people = outputs
        mo.md(f"**Detected {len(people)} players**").callout(kind="success")
    else:
        mo.md("_Load the model and upload an image first._").callout()
    return detect_btn, people


@app.cell
def _(SAM3D_DIR, cv2, estimator, faces, img_bgr, importlib, mo, os, people):
    if people and estimator is not None:
        _vis_path = os.path.join(SAM3D_DIR, "tools", "vis_utils.py")
        _spec = importlib.util.spec_from_file_location("sam3d_vis_utils", _vis_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        rend = _mod.visualize_sample_together(img_bgr, people, faces)
        mo.image(cv2.cvtColor(rend, cv2.COLOR_BGR2RGB), caption="Detection + mesh overlay")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Draw Goal-Parallel Lines

        Click **4 points** on the image below: first 2 points define line 1, next 2
        define line 2. Both lines should be parallel to the goal line. This fixes the
        offside axis via the vanishing point.

        Or click **Auto-detect** to find pitch lines automatically.
        """
    )
    return


@app.cell
def _(img_rgb, mo):
    line_pts = mo.ui.point_collection_drawing(
        img_rgb,
        labeling_instructions="Click 4 points: 2 for each goal-parallel line"
    ) if img_rgb is not None else None
    line_pts
    return line_pts,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Select Players & Mark Defenders

        Select which players to reconstruct in 3D (usually the ones near the offside
        decision). Then mark which are defenders — the offside line is drawn at the
        furthest-forward defender's body point (arms excluded).
        """
    )
    return


@app.cell
def _(mo, people):
    player_ids = list(range(len(people))) if people else []
    selected_players = mo.ui.multiselect(
        options=player_ids,
        value=player_ids,
        label="Players to reconstruct"
    )
    defenders = mo.ui.multiselect(
        options=player_ids,
        value=player_ids[:max(1, len(player_ids)//2)],
        label="Defenders (incl. GK)"
    )
    mo.vstack([selected_players, defenders])
    return defenders, selected_players


@app.cell
def _(mo):
    attack_dir = mo.ui.radio(
        ["−X  ←", "+X  →"], value="−X  ←",
        label="Attacking direction"
    )
    flip_up = mo.ui.checkbox(label="Flip up (if players are upside-down)")
    mo.vstack([attack_dir, flip_up])
    return attack_dir, flip_up


@app.cell
def _(mo):
    mo.md(r"""## 6. Build 3D Scene + Offside Verdict""")
    return


@app.cell
def _(attack_dir, estimator, faces, flip_up, img_bgr, line_pts, mo, people, selected_players, defenders):
    build_btn = mo.ui.run_button(label="Build 3D scene", kind="success")
    build_btn
    return build_btn,


@app.cell
def _(
        attack_dir, build_btn, defenders, estimator, faces, flip_up, img_rgb,
        line_pts, mo, np, people, selected_players
    ):
    scene_html = None
    verdict = None

    if build_btn.value and people and line_pts is not None:
        pts = line_pts.value
        if len(pts) >= 4:
            import sys as _sys
            _sys.path.insert(0, _sys.path[0])
            from var_geometry import (
                goal_dir_from_lines, place_players, offside_plane_x,
                non_arm_mask, build_scene
            )

            selected = [int(i) for i in selected_players.value]
            h, w = img_rgb.shape[:2]
            focal = float(people[0]["focal_length"])

            # Build per-player dicts
            p_dict = {}
            for i in selected:
                p = people[i]
                p_dict[i] = {
                    "bbox": np.asarray(p["bbox"]).reshape(-1)[:4].astype(float),
                    "pred_vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
                    "pred_cam_t": np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3),
                    "focal_length": float(np.asarray(p["focal_length"]).reshape(-1)[0]),
                    "pred_keypoints_3d": (
                        np.asarray(p["pred_keypoints_3d"], dtype=np.float32)
                        if p.get("pred_keypoints_3d") is not None else None
                    ),
                }

            goal_dir = goal_dir_from_lines(pts, focal, w, h)
            placed = place_players(p_dict, selected, goal_dir, flip_up=flip_up.value)

            attack_sign = -1 if attack_dir.value.startswith("−") else +1
            dset = [int(d) for d in defenders.value]
            masks = {i: non_arm_mask(p_dict[i]) for i in selected}
            plane_x = offside_plane_x(placed, attack_sign, dset, masks)

            fig, any_off = build_scene(placed, faces, plane_x, attack_sign, dset, masks)
            verdict = "OFFSIDE" if any_off else "NO OFFSIDE"
            scene_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

            med_h = float(np.median([placed[i][:, 2].max() for i in selected]))
            mo.vstack([
                mo.md(f"### Verdict: {verdict}").callout(
                    kind="danger" if any_off else "success"),
                mo.md(f"Median player height: **{med_h:.2f} m**  "
                      f"| Offside plane X: **{plane_x:.2f} m**"),
                mo.Html(scene_html),
            ])
        else:
            mo.md("⚠️ Draw at least 4 points (2 lines) on the image.").callout(kind="warn")
    else:
        mo.md("_Detect players and draw lines first._").callout()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How It Works

        ### SAM 3D Body Architecture

        SAM 3D Body uses an **encoder-decoder** architecture with a DINOv3 backbone
        for image feature extraction. It predicts:

        - **3D body vertices** — a full-body mesh in the MHR (Momentum Human Rig) format
        - **Camera translation** (`pred_cam_t`) — placing the mesh in camera space
        - **3D keypoints** (MHR-70) — joint positions for pose-based analysis

        The model supports auxiliary prompts (2D keypoints, masks) for user-guided
        inference, similar to the SAM family.

        ### Offside Geometry

        The offside verdict is computed in 3 world-space steps:

        1. **World vertices** = `pred_vertices + pred_cam_t` (all players share one
           camera frame)
        2. **Field frame** = SVD ground fit from feet vertices, with the goal direction
           from the vanishing point of two clicked parallel lines →
           X = offside axis, Y = goal line, Z = up
        3. **Offside plane** = constant-X plane at the furthest-forward defender's
           furthest body point (arms/hands excluded per the Laws of the Game)

        Any attacker with any body part past this plane is **offside**.

        ### Extensions & Insights

        - **Arms excluded**: Using MHR-70 keypoints, vertices nearest arm/hand joints
          are masked out — the offside law considers only the body to the second-last
          defender
        - **Per-player height normalization**: each player's feet are dropped to Z=0
          individually (not one global shift), so uneven ground doesn't tilt players
        - **Scale from body height**: metric positions come from reconstructed body
          height (~1.7–1.9m), giving approximate meters without pitch calibration

        > ⚠️ **Honesty**: Scale is approximate — good for relative offside ordering,
        > not sub-10cm calls. A full metric homography was deliberately avoided.
        """
    )
    return


if __name__ == "__main__":
    app.run()
