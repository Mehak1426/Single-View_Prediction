import os
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import open3d as o3d
except Exception:
    o3d = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEXT_EXTS = {".txt", ".md", ".log", ".json", ".yaml", ".yml", ".csv"}
POINT_CLOUD_EXTS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}
IDX_RE = re.compile(r"(?:^|[_-])idx(?P<idx>\d+)(?:[_-]|\.|$)", re.IGNORECASE)


def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return p.name


@st.cache_data(show_spinner=False)
def discover_outputs(root_dir_str: str) -> dict:
    root_dir = Path(root_dir_str)
    files = [p for p in root_dir.rglob("*") if p.is_file()]
    files.sort(key=lambda p: (_safe_relpath(p, root_dir).lower()))

    by_idx: dict[str, list[Path]] = {}
    ungrouped: list[Path] = []
    for p in files:
        m = IDX_RE.search(p.name)
        if m:
            idx = m.group("idx")
            by_idx.setdefault(idx, []).append(p)
        else:
            ungrouped.append(p)

    return {"by_idx": by_idx, "ungrouped": ungrouped, "all": files}


def show_text_file(p: Path) -> None:
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        st.error(f"Failed to read {p.name}: {e}")
        return

    if p.suffix.lower() == ".md":
        st.markdown(content)
    else:
        st.text(content)


def show_point_cloud(p: Path, max_points: int = 200_000) -> None:
    if o3d is None:
        st.warning("open3d isn't available in this environment, so point clouds can't be previewed.")
        return

    try:
        pcd = o3d.io.read_point_cloud(str(p))
    except Exception as e:
        st.error(f"Failed to load point cloud {p.name}: {e}")
        return

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        st.info("Point cloud has 0 points.")
        return

    if len(pts) > max_points:
        pts = pts[:max_points]
        colors = np.asarray(pcd.colors)
        if colors.shape[0] == len(np.asarray(pcd.points)):
            colors = colors[:max_points]
        else:
            colors = np.empty((len(pts), 0))
    else:
        colors = np.asarray(pcd.colors)

    has_rgb = colors.ndim == 2 and colors.shape[1] == 3 and colors.shape[0] == len(pts)
    if has_rgb:
        color = colors
    else:
        # Fallback: color by depth (z)
        z = pts[:, 2]
        z = (z - z.min()) / (z.ptp() + 1e-9)
        color = z

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=1),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            bgcolor="black",
            aspectmode="data",
        ),
        paper_bgcolor="black",
        margin=dict(l=0, r=0, b=0, t=0),
        height=800,
    )
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Test Outputs Showcase", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  .sv-card {
    padding: 0.9rem 1rem;
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
  }
  .sv-muted { opacity: 0.75; }
  .sv-title { font-size: 2.0rem; font-weight: 750; margin: 0.1rem 0 0.2rem 0; }
  .sv-subtitle { font-size: 1rem; margin: 0 0 0.6rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="sv-card">
  <div class="sv-title">Test Outputs Showcase</div>
  <div class="sv-subtitle sv-muted">Browse depth-map images, metrics, and point clouds directly from your <code>test_outputs</code> folder.</div>
</div>
""",
    unsafe_allow_html=True,
)

default_root = Path("test_outputs")
with st.sidebar:
    st.header("Controls")
    root = st.text_input("Outputs folder", value=str(default_root))
    root_dir = Path(root)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with c2:
        auto_refresh = st.toggle("Auto-refresh", value=False, help="Refresh the page every few seconds.")

    if auto_refresh:
        st.caption("Auto-refresh is on.")
        st.autorefresh(interval=5000, key="sv_autorefresh")

if not root_dir.exists():
    st.error(f"Folder not found: {root_dir}")
    st.stop()

discovered = discover_outputs(str(root_dir))

all_files: list[Path] = discovered["all"]
images_all = sum(1 for p in all_files if p.suffix.lower() in IMAGE_EXTS)
texts_all = sum(1 for p in all_files if p.suffix.lower() in TEXT_EXTS)
clouds_all = sum(1 for p in all_files if p.suffix.lower() in POINT_CLOUD_EXTS)
others_all = len(all_files) - images_all - texts_all - clouds_all

stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("Files", len(all_files))
stat2.metric("Images", images_all)
stat3.metric("Text", texts_all)
stat4.metric("Point clouds", clouds_all)

with st.sidebar:
    st.caption(f"Files found: {len(all_files)}")
    idxs = sorted(discovered["by_idx"].keys(), key=lambda s: int(s) if s.isdigit() else s)

    group_options = ["(ungrouped)"] + [f"idx{n}" for n in idxs]
    group_counts = {"(ungrouped)": len(discovered["ungrouped"])}
    for n in idxs:
        group_counts[f"idx{n}"] = len(discovered["by_idx"].get(n, []))

    selected_idx = st.selectbox(
        "Select idx (group)",
        options=group_options,
        format_func=lambda g: f"{g}  ({group_counts.get(g, 0)})",
    )

    search = st.text_input("Search filename", value="", placeholder="e.g. comparison, error_map, eval")
    sort_mode = st.selectbox("Sort", options=["Name (A→Z)", "Name (Z→A)", "Modified (new→old)", "Modified (old→new)"])
    img_cols = st.slider("Image grid columns", min_value=2, max_value=6, value=3, step=1)

if selected_idx == "(ungrouped)":
    group_files = discovered["ungrouped"]
    group_label = "Ungrouped files"
else:
    idx = selected_idx.replace("idx", "")
    group_files = discovered["by_idx"].get(idx, [])
    group_label = f"Group: {selected_idx}"

st.subheader(group_label)

if search.strip():
    s = search.strip().lower()
    group_files = [p for p in group_files if s in p.name.lower() or s in _safe_relpath(p, root_dir).lower()]

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0

if sort_mode == "Name (A→Z)":
    group_files = sorted(group_files, key=lambda p: _safe_relpath(p, root_dir).lower())
elif sort_mode == "Name (Z→A)":
    group_files = sorted(group_files, key=lambda p: _safe_relpath(p, root_dir).lower(), reverse=True)
elif sort_mode == "Modified (new→old)":
    group_files = sorted(group_files, key=_mtime, reverse=True)
else:
    group_files = sorted(group_files, key=_mtime)

if not group_files:
    st.info("No files in this group.")
else:
    images = [p for p in group_files if p.suffix.lower() in IMAGE_EXTS]
    texts = [p for p in group_files if p.suffix.lower() in TEXT_EXTS]
    clouds = [p for p in group_files if p.suffix.lower() in POINT_CLOUD_EXTS]
    others = [p for p in group_files if p not in set(images + texts + clouds)]

    tab_names = []
    if images:
        tab_names.append(f"Depth maps / images ({len(images)})")
    if texts:
        tab_names.append(f"Metrics / text ({len(texts)})")
    if clouds:
        tab_names.append(f"Point clouds ({len(clouds)})")
    if others:
        tab_names.append(f"Other files ({len(others)})")

    tabs = st.tabs(tab_names if tab_names else ["Files"])

    tab_i = 0
    if images:
        with tabs[tab_i]:
            top = st.columns([1, 1, 2])
            with top[0]:
                st.caption("Tip: click an image to zoom in your browser.")
            with top[1]:
                per_page = st.selectbox("Images per page", options=[12, 24, 48, 96], index=1)
            with top[2]:
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max(1, (len(images) + per_page - 1) // per_page),
                    value=1,
                    step=1,
                )

            start = (int(page) - 1) * int(per_page)
            end = min(len(images), start + int(per_page))
            page_imgs = images[start:end]

            cols = st.columns(int(img_cols))
            for i, p in enumerate(page_imgs):
                with cols[i % int(img_cols)]:
                    st.image(str(p), caption=_safe_relpath(p, root_dir), use_container_width=True)
                    with st.expander("Details", expanded=False):
                        st.code(_safe_relpath(p, root_dir))
                        st.caption(f"Modified: {_mtime(p):.0f}")
                        st.download_button(
                            "Download image",
                            data=p.read_bytes(),
                            file_name=p.name,
                            mime="application/octet-stream",
                            key=f"dl_img_{p}",
                        )
        tab_i += 1

    if texts:
        with tabs[tab_i]:
            choice = st.selectbox("Open file", options=[_safe_relpath(p, root_dir) for p in texts])
            chosen = next(p for p in texts if _safe_relpath(p, root_dir) == choice)
            with st.container(border=True):
                st.caption(_safe_relpath(chosen, root_dir))
                show_text_file(chosen)
            st.download_button(
                "Download",
                data=chosen.read_bytes(),
                file_name=chosen.name,
                mime="text/plain",
                key=f"dl_txt_{chosen}",
            )
        tab_i += 1

    if clouds:
        with tabs[tab_i]:
            choice = st.selectbox("Open point cloud", options=[_safe_relpath(p, root_dir) for p in clouds])
            chosen = next(p for p in clouds if _safe_relpath(p, root_dir) == choice)
            max_pts = st.slider("Max points to render", 10_000, 500_000, 200_000, step=10_000)
            st.caption(_safe_relpath(chosen, root_dir))
            show_point_cloud(chosen, max_points=int(max_pts))
            st.download_button(
                "Download",
                data=chosen.read_bytes(),
                file_name=chosen.name,
                mime="application/octet-stream",
                key=f"dl_cloud_{chosen}",
            )
        tab_i += 1

    if others:
        with tabs[tab_i]:
            for p in others:
                st.write(_safe_relpath(p, root_dir))
                st.download_button(
                    "Download",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="application/octet-stream",
                    key=f"dl_other_{p}",
                )