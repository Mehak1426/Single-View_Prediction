import streamlit as st
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
 
# --- PAGE CONFIG ---
st.set_page_config(page_title="Depth Prediction Showcase", layout="wide", page_icon="🧊")
 
# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .idx-card {
        padding: 28px;
        border-radius: 16px;
        background-color: #161b22;
        border: 1px solid #30363d;
        margin-bottom: 40px;
    }
    .scene-title {
        color: #58a6ff;
        font-family: 'Segoe UI', sans-serif;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .metric-box {
        background-color: #0d1117;
        padding: 14px 18px;
        border-radius: 10px;
        border-left: 4px solid #238636;
        font-family: monospace;
        font-size: 13px;
        color: #c9d1d9;
        white-space: pre-wrap;
        line-height: 1.7;
    }
    .sparsity-box {
        background-color: #0d1117;
        padding: 14px 18px;
        border-radius: 10px;
        border-left: 4px solid #9e6a03;
        font-family: monospace;
        font-size: 13px;
        color: #c9d1d9;
        white-space: pre-wrap;
        line-height: 1.7;
    }
    .tag {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        color: #8b949e;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 6px;
    }
    h2, h3 { color: #e6edf3 !important; }
    .stDivider { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)
 
# --- CONFIGURATION ---
DEFAULT_ROOTS = ["test_outputs", "test_output", "outputs"]
 
 
# ── PLY PARSER (no open3d needed) ───────────────────────────────────────────
def read_ply_numpy(ply_path: Path) -> np.ndarray | None:
    """
    Pure-Python PLY reader. Returns Nx3 float32 array of (x, y, z).
    Supports ASCII and binary_little_endian formats.
    """
    try:
        with open(ply_path, "rb") as f:
            # Parse header
            header_lines = []
            while True:
                line = f.readline().decode("utf-8", errors="ignore").strip()
                header_lines.append(line)
                if line == "end_header":
                    break
 
            fmt = "ascii"
            n_vertices = 0
            prop_names = []
            for l in header_lines:
                if l.startswith("format"):
                    fmt = l.split()[1]
                if l.startswith("element vertex"):
                    n_vertices = int(l.split()[-1])
                if l.startswith("property"):
                    prop_names.append(l.split()[-1])
 
            x_i = prop_names.index("x") if "x" in prop_names else 0
            y_i = prop_names.index("y") if "y" in prop_names else 1
            z_i = prop_names.index("z") if "z" in prop_names else 2
            n_props = len(prop_names)
 
            if fmt == "ascii":
                rows = []
                for _ in range(n_vertices):
                    vals = f.readline().decode().split()
                    rows.append([float(vals[x_i]), float(vals[y_i]), float(vals[z_i])])
                return np.array(rows, dtype=np.float32)
            else:
                # binary_little_endian — assume all float32 (most common)
                byte_count = n_vertices * n_props * 4
                raw = f.read(byte_count)
                data = np.frombuffer(raw, dtype="<f4").reshape(n_vertices, n_props)
                return data[:, [x_i, y_i, z_i]]
    except Exception as e:
        st.warning(f"PLY parse error: {e}")
        return None
 
 
def make_3d_figure(points: np.ndarray) -> go.Figure:
    """Downsample and render a colourised 3-D scatter via Plotly."""
    if len(points) > 60_000:
        idx = np.random.choice(len(points), 60_000, replace=False)
        points = points[idx]
 
    z = points[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=z,
        mode="markers",
        marker=dict(
            size=1.2,
            color=z,
            colorscale="Viridis",
            opacity=0.85,
            colorbar=dict(thickness=10, title="Depth"),
        ),
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig
 
 
def _safe_rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return p.name


def _list_scene_folders(root: Path) -> list[Path]:
    """Return subfolders that look like per-scene outputs (idx*, scene*, etc.)."""
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    # Prefer folders that look like idx_* or idx123
    def score(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        is_idx = 1 if name.startswith("idx") else 0
        return (-is_idx, name)

    return sorted(dirs, key=score)


def _find_default_root() -> Path:
    for cand in DEFAULT_ROOTS:
        p = Path(cand)
        if p.exists():
            return p
    return Path(DEFAULT_ROOTS[0])


# ── HEADER ───────────────────────────────────────────────────────────────────
st.title(" Depth Prediction Showcase")
st.markdown("Visualising high-fidelity depth estimation and 3D point-cloud reconstruction results.")

with st.expander("How to generate these outputs (run locally, then push to GitHub)", expanded=False):
    st.markdown(
        """
This Streamlit site **only displays files** from your outputs folder. To generate the visualizations locally:

```bash
python scripts/run_pipeline.py --index 0 --anchors 100 --device cpu --output test_outputs
```

If you see `CSV manifest not found: nyu2_train.csv`, you need to download/setup NYU Depth V2 into `nyu_data/` (see `README.md`), or pass a custom path via `--data`.

Then commit/push `test_outputs/` to GitHub so Streamlit Cloud can render it.
"""
    )
 
# ── GLOBAL METRICS (root-level files) ────────────────────────────────────────
with st.sidebar:
    st.title("ℹ Dashboard")
    root_dir = st.text_input("Outputs folder", value=str(_find_default_root()))
    DATA_ROOT = Path(root_dir)
    st.caption("Tip: On Streamlit Cloud this must exist in your GitHub repo.")
    st.divider()

    scene_folders = _list_scene_folders(DATA_ROOT)
    has_scene_folders = len(scene_folders) > 0
    if has_scene_folders:
        options = [p.name for p in scene_folders]
        default_sel = options[: min(8, len(options))]
        selected = st.multiselect("Scenes to show", options=options, default=default_sel)
        show_all = st.toggle("Show all scenes", value=False)
        if show_all:
            selected = options
        per_page = st.selectbox("Cards per page", options=[3, 5, 8, 12, 20], index=2)
        total_pages = max(1, (len(selected) + int(per_page) - 1) // int(per_page))
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    else:
        selected = []

    if st.button("🔄 Refresh"):
        st.rerun()

if not DATA_ROOT.exists():
    st.error(f"❌ Directory `{DATA_ROOT}` not found. Commit an outputs folder (e.g. `test_outputs/`) to your repo.")
    st.stop()

root_eval = DATA_ROOT / "eval_results.txt"
root_sparsity = DATA_ROOT / "sparsity_data.txt"
root_sparsity_img = DATA_ROOT / "sparsity_sensitivity.png"
 
if root_eval.exists() or root_sparsity.exists() or root_sparsity_img.exists():
    with st.expander(" Global Evaluation & Sparsity Summary", expanded=True):
        g1, g2 = st.columns(2)
        with g1:
            if root_eval.exists():
                st.markdown("**Overall Evaluation Metrics**")
                st.markdown(f'<div class="metric-box">{root_eval.read_text()}</div>',
                            unsafe_allow_html=True)
            if root_sparsity.exists():
                st.markdown("**Sparsity Data**")
                st.markdown(f'<div class="sparsity-box">{root_sparsity.read_text()}</div>',
                            unsafe_allow_html=True)
        with g2:
            if root_sparsity_img.exists():
                st.markdown("**Sparsity Sensitivity Plot**")
                st.image(str(root_sparsity_img), use_container_width=True)
    st.divider()
 
# ── CONTENT ─────────────────────────────────────────────────────────────────
if has_scene_folders:
    if not selected:
        st.warning("No scenes selected.")
        st.stop()

    start = (int(page) - 1) * int(per_page)
    end = min(len(selected), start + int(per_page))
    visible = selected[start:end]

    rendered = 0
    for idx_name in visible:
        folder = DATA_ROOT / idx_name
        if not folder.exists():
            st.warning(f"Folder `{folder}` not found — skipping.")
            continue

        num = "".join(filter(str.isdigit, idx_name))  # e.g. "1239" from "idx_1239"

        # Locate files (flexible glob so naming variants work)
        comparison_img = next(folder.glob(f"comparison*{num}*.png"), None) or next(folder.glob("comparison*.png"), None)
        error_img = next(folder.glob(f"error_map*{num}*.png"), None) or next(folder.glob("error_map*.png"), None)
        ply_file = next(folder.glob(f"pointcloud*{num}*.ply"), None) or next(folder.glob("pointcloud*.ply"), None)
        eval_txt = folder / "eval_results.txt"

        st.markdown(f'<div class="idx-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="scene-title">📍 Scene — {idx_name}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="tag">folder: {_safe_rel(folder, DATA_ROOT)}</span>',
            unsafe_allow_html=True,
        )

        # Row 1: Comparison image + Error map
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**Comparison (Input · Predicted · Ground Truth)**")
            if comparison_img:
                st.image(str(comparison_img), use_container_width=True)
            else:
                st.info("No comparison image found.")

        with img_col2:
            st.markdown("**Error Map**")
            if error_img:
                st.image(str(error_img), use_container_width=True)
            else:
                st.info("No error map found.")

        # Row 2: Metrics + 3D point cloud
        st.markdown("---")
        metric_col, cloud_col = st.columns([1, 1.6])

        with metric_col:
            st.markdown("**Evaluation Metrics**")
            if eval_txt.exists():
                st.markdown(f'<div class="metric-box">{eval_txt.read_text()}</div>', unsafe_allow_html=True)
            else:
                st.info("No eval_results.txt in this folder.")

        with cloud_col:
            st.markdown("**3D Point Cloud**")
            if ply_file:
                with st.spinner("Loading point cloud…"):
                    pts = read_ply_numpy(ply_file)
                if pts is not None:
                    st.plotly_chart(make_3d_figure(pts), use_container_width=True)
                else:
                    st.error("Could not parse the .ply file.")
            else:
                st.info("No .ply file found for this scene.")

        st.markdown("</div>", unsafe_allow_html=True)
        rendered += 1

    if rendered == 0:
        st.error("No scene folders rendered. Make sure your outputs folder contains subfolders (e.g. `idx_0/`).")
else:
    # Flat folder mode: just list files and show quick previews
    st.info("No per-scene subfolders detected. Showing flat-folder viewer.")
    files = [p for p in DATA_ROOT.rglob("*") if p.is_file()]
    if not files:
        st.warning("No files found in this folder.")
        st.stop()

    imgs = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    plys = [p for p in files if p.suffix.lower() == ".ply"]
    txts = [p for p in files if p.suffix.lower() in {".txt", ".md"}]

    tabs = st.tabs([f"Images ({len(imgs)})", f"Point clouds ({len(plys)})", f"Text ({len(txts)})"])
    with tabs[0]:
        for p in sorted(imgs)[:60]:
            st.image(str(p), caption=_safe_rel(p, DATA_ROOT), use_container_width=True)
    with tabs[1]:
        if plys:
            choice = st.selectbox("Select .ply", options=[_safe_rel(p, DATA_ROOT) for p in sorted(plys)])
            chosen = next(p for p in plys if _safe_rel(p, DATA_ROOT) == choice)
            pts = read_ply_numpy(chosen)
            if pts is not None:
                st.plotly_chart(make_3d_figure(pts), use_container_width=True)
        else:
            st.info("No .ply files found.")
    with tabs[2]:
        if txts:
            choice = st.selectbox("Select text file", options=[_safe_rel(p, DATA_ROOT) for p in sorted(txts)])
            chosen = next(p for p in txts if _safe_rel(p, DATA_ROOT) == choice)
            st.code(chosen.read_text(errors="replace"))
        else:
            st.info("No text files found.")

st.sidebar.caption("Built with Streamlit + Plotly · No Open3D required")