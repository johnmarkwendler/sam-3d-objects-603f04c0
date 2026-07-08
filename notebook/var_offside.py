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
    import base64
    import json
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
    return SAM3D_DIR, base64, importlib, json, np, os, subprocess, sys


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
def _(mo):
    install_btn = mo.ui.run_button(label="Install all dependencies")
    install_btn
    return install_btn,


@app.cell
def _(install_btn, mo, os, subprocess, sys):
    if install_btn.value:
        import torch as _torch
        with mo.status.spinner(title="Installing SAM 3D Body dependencies..."):
            _pybin = sys.executable

            subprocess.run([_pybin, "-m", "pip", "install", "setuptools<81"], check=True)
            subprocess.run([_pybin, "-m", "pip", "install", "numpy", "cython"], check=True)

            subprocess.run([_pybin, "-m", "pip", "install"] + [
                "pytorch-lightning", "pyrender", "opencv-python-headless",
                "yacs", "scikit-image", "einops", "timm", "dill", "pandas",
                "rich", "hydra-core", "hydra-submitit-launcher",
                "hydra-colorlog", "pyrootutils", "webdataset", "chump",
                "networkx==3.2.1", "roma", "joblib", "seaborn", "appdirs",
                "jsonlines", "loguru", "optree",
                "fvcore", "trimesh", "plotly", "kaleido",
                "braceexpand", "matplotlib",
            ], check=True)

            subprocess.run([_pybin, "-m", "pip", "install", "pycocotools"], check=True)
            subprocess.run([_pybin, "-m", "pip", "install",
                "--no-build-isolation", "xtcocotools"
            ], check=False)

            os.environ.pop("FORCE_CUDA", None)
            os.environ.pop("CUDA_HOME", None)
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)

            subprocess.run([
                _pybin, "-m", "pip", "install",
                "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
                "--no-build-isolation", "--no-deps"
            ], check=True)
            subprocess.run([
                _pybin, "-m", "pip", "install",
                "git+https://github.com/microsoft/MoGe.git"
            ], check=False)
        mo.md("✅ All dependencies installed. Click **Load SAM 3D Body model** "
              "below.").callout(kind="success")
    else:
        mo.md("Click the button above to install all dependencies (SAM 3D Body, "
              "detectron2, MoGe2, and supporting packages).").callout(kind="info")
    return


@app.cell
def _(SAM3D_DIR, hf_repo, importlib, mo, os):
    load_btn = mo.ui.run_button(label="Load SAM 3D Body model")
    load_btn
    return load_btn,


@app.cell
def _(SAM3D_DIR, hf_repo, importlib, load_btn, mo, np, os):
    import torch

    estimator = None
    faces = None
    setup_sam_3d_body = None
    if load_btn.value:
        with mo.status.spinner(title="Loading SAM 3D Body..."):
            _utils_path = os.path.join(SAM3D_DIR, "notebook", "utils.py")
            _spec = importlib.util.spec_from_file_location("sam3d_notebook_utils", _utils_path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            setup_sam_3d_body = _mod.setup_sam_3d_body
            estimator = setup_sam_3d_body(hf_repo_id=hf_repo.value)
            faces = np.asarray(estimator.faces)
        mo.md(f"✅ Model loaded on **{torch.cuda.get_device_name(0)}** "
              f"({len(faces)} faces in body mesh)").callout(kind="success")
    else:
        mo.md("Click **Install dependencies** above, then **Load SAM 3D Body model**.").callout(kind="info")
    return estimator, faces, setup_sam_3d_body, torch


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
def _(conf_slider, mo):
    detect_btn = mo.ui.run_button(label="Detect players")
    detect_btn
    return conf_slider, detect_btn,


@app.cell
def _(conf_slider, detect_btn, estimator, img_rgb, mo):
    people = []
    _msg = mo.md("_Load the model and upload an image first, then click **Detect players**._").callout()
    if detect_btn.value and estimator is not None and img_rgb is not None:
        with mo.status.spinner(title="Detecting players..."):
            outputs = estimator.process_one_image(img_rgb, bbox_thr=conf_slider.value)
        people = outputs
        _msg = mo.md(f"**Detected {len(people)} players** — meshes reconstructed in 3D.").callout(kind="success")
    elif detect_btn.value and (estimator is None or img_rgb is None):
        _msg = mo.md("⚠️ Load the model and upload an image first.").callout(kind="warn")
    _msg
    return people,


@app.cell
def _(SAM3D_DIR, cv2, estimator, faces, img_bgr, importlib, mo, os, people):
    if people and estimator is not None:
        _vis_path = os.path.join(SAM3D_DIR, "tools", "vis_utils.py")
        _spec = importlib.util.spec_from_file_location("sam3d_vis_utils", _vis_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        rend = _mod.visualize_sample_together(img_bgr, people, faces)
        mo.image(cv2.cvtColor(rend, cv2.COLOR_BGR2RGB), caption="Detection + mesh overlay")
    elif people is not None and not people:
        mo.md("_No players detected yet. Click **Detect players** above._")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Set Goal-Parallel Lines

        **Click 4 points** on the image below: points 1-2 define line 1 (yellow),
        points 3-4 define line 2 (red). Both lines should be parallel to the goal
        line. This fixes the offside axis via the vanishing point.

        Click **Clear** to reset. The coordinates are in image pixel space.
        """
    )
    return


@app.cell
def _(base64, cv2, img_bgr, mo):
    _b64 = ""
    if img_bgr is not None:
        _, _buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _b64 = base64.b64encode(_buf).decode('utf-8')

    line_pts = mo.ui.html(f'''<input type="hidden" value="" />
<div id="cc" style="position:relative; display:inline-block; max-width:100%;">
  <img src="data:image/jpeg;base64,{_b64}" id="ccimg"
       style="max-width:100%; display:block; cursor:crosshair;" />
  <canvas id="cccv" style="position:absolute; top:0; left:0; pointer-events:none;"></canvas>
</div>
<div style="margin-top:6px;">
  <button id="ccclr" style="padding:5px 12px; background:#7c3aed; color:#fff; border:none; border-radius:6px; cursor:pointer;">Clear</button>
  <span style="margin-left:8px; color:#666; font-size:0.85rem;">Click 4 points on the image (yellow = line 1, red = line 2)</span>
</div>
<script>
(function() {{
  var img = document.getElementById('ccimg');
  var cv = document.getElementById('cccv');
  var inp = document.querySelector('input[type=hidden]');
  var clr = document.getElementById('ccclr');
  var pts = [];
  function fit() {{ cv.width = img.clientWidth; cv.height = img.clientHeight; draw(); }}
  function draw() {{
    var ctx = cv.getContext('2d');
    ctx.clearRect(0,0,cv.width,cv.height);
    var sx = img.naturalWidth / cv.width || 1;
    var sy = img.naturalHeight / cv.height || 1;
    pts.forEach(function(p,i) {{
      var dx = p[0]/sx, dy = p[1]/sy;
      ctx.fillStyle = i < 2 ? '#ffff00' : '#ff0000';
      ctx.beginPath(); ctx.arc(dx,dy,8,0,2*Math.PI); ctx.fill();
      ctx.strokeStyle='#000'; ctx.lineWidth=2; ctx.stroke();
      ctx.fillStyle='#000'; ctx.font='bold 13px sans-serif';
      ctx.fillText(String(i+1), dx-4, dy+5);
    }});
    if(pts.length>=2){{ ctx.strokeStyle='#ffff00'; ctx.lineWidth=3; ctx.beginPath();
      ctx.moveTo(pts[0][0]/sx,pts[0][1]/sy); ctx.lineTo(pts[1][0]/sx,pts[1][1]/sy); ctx.stroke(); }}
    if(pts.length>=4){{ ctx.strokeStyle='#ff0000'; ctx.beginPath();
      ctx.moveTo(pts[2][0]/sx,pts[2][1]/sy); ctx.lineTo(pts[3][0]/sx,pts[3][1]/sy); ctx.stroke(); }}
  }}
  img.addEventListener('click', function(e) {{
    var r = img.getBoundingClientRect();
    var sx = img.naturalWidth / r.width || 1;
    var sy = img.naturalHeight / r.height || 1;
    var x = Math.round((e.clientX - r.left) * sx);
    var y = Math.round((e.clientY - r.top) * sy);
    pts.push([x,y]);
    if(pts.length>4) pts = pts.slice(-4);
    inp.value = JSON.stringify(pts);
    inp.dispatchEvent(new Event('input', {{bubbles:true}}));
    draw();
  }});
  clr.addEventListener('click', function() {{
    pts = []; inp.value=''; inp.dispatchEvent(new Event('input',{{bubbles:true}})); draw();
  }});
  if(img.complete) fit(); else img.addEventListener('load', fit);
  window.addEventListener('resize', fit);
}})();
</script>''') if img_bgr is not None else None

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
def _(np):
    import plotly.graph_objects as go

    _ARM_KP = frozenset({7, 8, 41, 62} | set(range(21, 41)) | set(range(42, 62)))

    def goal_dir_from_lines(line_pts, focal, w, h):
        def _hl(p, q):
            return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
        lp = np.asarray(line_pts, dtype=float)
        l1, l2 = _hl(lp[0], lp[1]), _hl(lp[2], lp[3])
        vp = np.cross(l1, l2)
        K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.0]])
        d_img = (np.array([vp[0], vp[1], 0.0]) if abs(vp[2]) < 1e-9
                 else np.array([vp[0]/vp[2], vp[1]/vp[2], 1.0]))
        g = np.linalg.inv(K) @ d_img
        return g / np.linalg.norm(g)

    def world_verts(p):
        return np.asarray(p["pred_vertices"]) + np.asarray(p["pred_cam_t"]).reshape(1, 3)

    def non_arm_mask(p):
        V = np.asarray(p["pred_vertices"])
        kp = p.get("pred_keypoints_3d")
        if kp is None:
            return np.ones(len(V), bool)
        kp = np.asarray(kp)
        if kp.ndim != 2 or kp.shape[0] < 15:
            return np.ones(len(V), bool)
        kp = kp[:, :3]
        arm = [i for i in _ARM_KP if i < len(kp)]
        keep = [i for i in range(len(kp)) if i not in _ARM_KP]
        if not arm or not keep:
            return np.ones(len(V), bool)
        d_arm = np.linalg.norm(V[:, None, :] - kp[arm][None], axis=2).min(1)
        d_keep = np.linalg.norm(V[:, None, :] - kp[keep][None], axis=2).min(1)
        return d_keep <= d_arm

    def _fwd(Vf, attack_sign, mask=None):
        sel = Vf[mask] if (mask is not None and mask.any()) else Vf
        return float((attack_sign * sel[:, 0]).max())

    def place_players(people, selected_ids, goal_dir_cam, flip_up=False):
        def feet(Vc, frac=0.03):
            thr = np.quantile(Vc[:, 1], 1 - frac)
            return Vc[Vc[:, 1] >= thr]
        def head(Vc, frac=0.03):
            thr = np.quantile(Vc[:, 1], frac)
            return Vc[Vc[:, 1] <= thr]
        ups = []
        for i in selected_ids:
            Vc = world_verts(people[i])
            u = head(Vc).mean(0) - feet(Vc).mean(0)
            ups.append(u / (np.linalg.norm(u) + 1e-9))
        n = np.mean(ups, 0); n /= np.linalg.norm(n) + 1e-9
        if flip_up:
            n = -n
        feet_all = np.vstack([feet(world_verts(people[i])) for i in selected_ids])
        o = feet_all.mean(0)
        g = goal_dir_cam - (goal_dir_cam @ n) * n; g /= np.linalg.norm(g)
        ez, ey = n, g
        ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        Rwf = np.stack([ex, ey, ez], axis=1)
        placed = {i: (world_verts(people[i]) - o) @ Rwf for i in selected_ids}
        for i in placed:
            placed[i][:, 2] -= placed[i][:, 2].min()
        return placed

    def offside_plane_x(placed, attack_sign, defender_ids, masks=None):
        dset = set(int(d) for d in (defender_ids or []))
        m = masks or {}
        dfwd = [_fwd(placed[i], attack_sign, m.get(i)) for i in placed if i in dset]
        if dfwd:
            return attack_sign * max(dfwd)
        return float(np.median(np.vstack(list(placed.values()))[:, 0]))

    def build_scene(placed, faces, plane_x, attack_sign, defender_ids, masks=None, too_close=0.30):
        allP = np.vstack(list(placed.values()))
        x0, x1 = allP[:, 0].min() - 4, allP[:, 0].max() + 4
        y0, y1 = allP[:, 1].min() - 4, allP[:, 1].max() + 4
        m = masks or {}
        fmost = {i: _fwd(placed[i], attack_sign, m.get(i)) for i in placed}
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[0, 0, 0, 0],
                                i=[0, 0], j=[1, 2], k=[2, 3], color="seagreen", opacity=0.5))
        if plane_x is not None:
            fig.add_trace(go.Mesh3d(x=[plane_x]*4, y=[y0, y1, y1, y0], z=[0, 0, 3, 3],
                                    i=[0, 0], j=[1, 2], k=[2, 3], color="red", opacity=0.3))
        palette = ["crimson", "royalblue", "gold", "darkorange", "mediumpurple",
                   "deepskyblue", "hotpink", "mediumspringgreen", "tomato", "slateblue"]
        dset = set(int(d) for d in (defender_ids or []))
        any_off = False
        line_fwd = attack_sign * plane_x if plane_x is not None else None
        for n_, i in enumerate(placed):
            if i in dset:
                col, lab = "royalblue", "defender"
            elif line_fwd is None:
                col, lab = palette[n_ % len(palette)], ""
            else:
                margin = fmost[i] - line_fwd
                if margin > 0:
                    tight = " (tight)" if margin <= too_close else ""
                    col, lab, any_off = "red", f"OFFSIDE +{margin:.2f}m{tight}", True
                elif margin >= -too_close:
                    col, lab = "orange", f"level {margin:+.2f}m"
                else:
                    col, lab = "seagreen", f"onside {margin:.2f}m"
            V = placed[i]
            fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    color=col, opacity=1.0, name=f"#{i} {lab}"))
        title = "VAR — OFFSIDE" if any_off else "VAR — NO OFFSIDE"
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=22, color="white")),
            paper_bgcolor="#0b1f3a", height=640, margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(aspectmode="data", bgcolor="#0b1f3a",
                       xaxis_title="X offside axis (m)", yaxis_title="Y goal line (m)",
                       zaxis_title="Z (m)",
                       camera=dict(eye=dict(x=0, y=-2.2, z=1.2), up=dict(x=0, y=0, z=1))))
        return fig, any_off
    return build_scene, goal_dir_from_lines, non_arm_mask, offside_plane_x, place_players


@app.cell
def _(
        attack_dir, build_btn, defenders, estimator, faces, flip_up, img_rgb,
        line_pts, mo, np, people, selected_players,
        build_scene, goal_dir_from_lines, non_arm_mask, offside_plane_x, place_players,
        json
    ):
    scene_html = None
    verdict = None

    _pts_raw = line_pts.value if (line_pts is not None and line_pts.value) else "[]"
    _pts_list = json.loads(_pts_raw) if isinstance(_pts_raw, str) else _pts_raw

    if build_btn.value and people and len(_pts_list) >= 4:
        pts = np.array(_pts_list[:4], dtype=float)
        selected = [int(i) for i in selected_players.value]
        h, w = img_rgb.shape[:2]
        focal = float(people[0]["focal_length"])

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
    elif build_btn.value and len(_pts_list) < 4:
        mo.md(f"⚠️ Click 4 points on the image (you have {len(_pts_list)}).").callout(kind="warn")
    else:
        mo.md("_Detect players and click 4 points on the image first._").callout()
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
