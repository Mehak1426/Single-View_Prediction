import streamlit as st
import os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import open3d as o3d

# --- PAGE CONFIG ---
st.set_page_config(page_title="Depth Prediction Showcase", layout="wide")

# Custom CSS for the "Showcase" Look
st.markdown("""
<style>
    .idx-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #11141c;
        border: 1px solid #30363d;
        margin-bottom: 40px;
    }
    .scene-title {
        color: #58a6ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #0d1117;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #238636;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
DATA_ROOT = Path("test_output")
# Selected 5 Indices to Showcase
SHOWCASE_IDXS = ["idx_0", "idx_1239", "idx_1602", "idx_17049", "idx_18627"]

def get_plotly_pc(ply_path):
    """Uses Open3D to read and Plotly to render for best compatibility."""
    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        
        # Performance: Downsample if cloud is too dense for browser
        if len(points) > 70000:
            indices = np.random.choice(len(points), 70000, replace=False)
            points = points[indices]

        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=1.2, color=points[:, 2], colorscale='Viridis', opacity=0.8)
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                aspectmode='data'
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except Exception as e:
        return None

# --- UI HEADER ---
st.title(" Depth Prediction Showcase")
st.write("Visualizing high-fidelity depth estimation and 3D reconstruction results.")
st.divider()

if not DATA_ROOT.exists():
    st.error(f"Directory '{DATA_ROOT}' not found. Please check your folder structure.")
else:
    for idx_name in SHOWCASE_IDXS:
        folder_path = DATA_ROOT / idx_name
        
        if not folder_path.exists():
            continue

        # Extract numeric ID for file matching (e.g., "5077" from "idx_5077")
        # Handles both "idx0" and "idx_0" formats
        num_match = "".join(filter(str.isdigit, idx_name))

        st.markdown(f'<div class="idx-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="scene-title">📍 Scene Index: {idx_name}</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader(" Result Comparison")
            # Pattern matching for images based on your screenshots
            img_path = next(folder_path.glob(f"comparison*idx{num_match}.png"), None)
            
            if img_path and img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Comparison image (comparison_idx{num_match}.png) missing.")

            # Display Metrics
            eval_path = folder_path / "eval_results.txt"
            if eval_path.exists():
                st.markdown("**Evaluation Metrics:**")
                st.markdown(f'<div class="metric-container">{eval_path.read_text()}</div>', unsafe_allow_html=True)

        with col_right:
            st.subheader(" 3D Point Cloud")
            ply_path = next(folder_path.glob(f"pointcloud*idx{num_match}.ply"), None)
            
            if ply_path and ply_path.exists():
                with st.spinner(f"Loading 3D Scene..."):
                    fig = get_plotly_pc(ply_path)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to render 3D cloud.")
            else:
                st.info("No .ply file found for this scene.")

        st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("### Dashboard Info")
st.sidebar.write("This gallery showcases a curated subset of model predictions.")
if st.sidebar.button("Refresh Cache"):
    st.rerun()