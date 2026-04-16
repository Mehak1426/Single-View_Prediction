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

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Single-View Prediction Showcase", layout="wide")

# Custom CSS for a professional "Card" look
st.markdown("""
<style>
    .idx-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
        margin-bottom: 2rem;
    }
    .idx-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 1rem;
        border-bottom: 1px solid #444;
    }
    .metric-box {
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
POINT_CLOUD_EXTS = {".ply", ".pcd"}
IDX_RE = re.compile(r"idx(?P<idx>\d+)", re.IGNORECASE)

# --- HELPER FUNCTIONS ---
def get_files_for_idx(root_dir: Path, target_idx: str):
    """Finds all specific components for a given index."""
    found = {"images": [], "clouds": [], "text": []}
    # Search specifically in folders named like idx_314 or for files containing idx314
    for p in root_dir.rglob(f"*idx{target_idx}*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS: found["images"].append(p)
            elif ext in POINT_CLOUD_EXTS: found["clouds"].append(p)
            elif ext in {".txt", ".json"}: found["text"].append(p)
    return found

def show_mini_pc(p: Path):
    """Renders a lightweight 3D preview."""
    if o3d is None: return st.warning("Open3D not available.")
    try:
        pcd = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pcd.points)
        # Downsample for performance in grid view
        if len(pts) > 50000:
            pts = pts[np.random.choice(len(pts), 50000, replace=False)]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=1, color=pts[:,2], colorscale='Viridis', opacity=0.8)
        )])
        fig.update_layout(
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
            margin=dict(l=0,r=0,b=0,t=0), height=350, paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"3D Error: {e}")

# --- MAIN UI ---
st.title("🔬 Prediction Showcase Gallery")
st.markdown("Select specific indices from the sidebar to compare model outputs and metrics.")

# Sidebar Controls
default_root = Path("test_outputs")
with st.sidebar:
    st.header("Settings")
    root_input = st.text_input("Data Root", value=str(default_root))
    root_dir = Path(root_input)
    
    if root_dir.exists():
        # Auto-discover all available IDX values
        all_files = list(root_dir.rglob("*"))
        all_idxs = sorted(list(set(IDX_RE.findall(" ".join([p.name for p in all_files])))), key=int)
        
        selected_idxs = st.multiselect(
            "Select Indices to Showcase", 
            options=all_idxs, 
            default=all_idxs[:3] if len(all_idxs) >= 3 else all_idxs
        )
        
        st.divider()
        render_3d = st.checkbox("Enable 3D Previews", value=False)
        img_mode = st.radio("Primary Image", ["comparison", "error_map", "sparsity"])
    else:
        st.error("Path not found.")
        st.stop()

# --- CONTENT DISPLAY ---
if not selected_idxs:
    st.info("Please select at least one index in the sidebar to display results.")
else:
    for idx_val in selected_idxs:
        data = get_files_for_idx(root_dir, idx_val)
        
        # Start Card Container
        st.markdown(f'<div class="idx-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="idx-header">Result Set: idx_{idx_val}</div>', unsafe_allow_html=True)
        
        col_img, col_metrics, col_3d = st.columns([2, 1, 1.2])
        
        # 1. Image Column
        with col_img:
            target_img = next((p for p in data["images"] if img_mode in p.name.lower()), None)
            if target_img:
                st.image(str(target_img), caption=f"View: {img_mode}", use_container_width=True)
            else:
                st.warning(f"No '{img_mode}' image found.")

        # 2. Metrics Column
        with col_metrics:
            st.subheader("📊 Metrics")
            eval_file = next((p for p in data["text"] if "eval" in p.name.lower()), None)
            if eval_file:
                content = eval_file.read_text().strip()
                st.markdown(f'<div class="metric-box">{content}</div>', unsafe_allow_html=True)
            else:
                st.caption("No eval_results.txt found.")
            
            # Show other files in this group
            with st.expander("Other Files"):
                for p in data["images"] + data["text"]:
                    st.caption(f"📄 {p.name}")

        # 3. 3D Column
        with col_3d:
            st.subheader("☁️ Point Cloud")
            cloud_file = next((p for p in data["clouds"]), None)
            if cloud_file:
                if render_3d:
                    show_mini_pc(cloud_file)
                else:
                    st.info("3D rendering disabled in sidebar.")
                    st.download_button(f"Download {cloud_file.name}", cloud_file.read_bytes(), file_name=cloud_file.name)
            else:
                st.caption("No .ply file found for this index.")

        st.markdown('</div>', unsafe_allow_html=True) # End Card

st.sidebar.markdown("---")
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.rerun()