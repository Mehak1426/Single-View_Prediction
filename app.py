import streamlit as st
import os
import plotly.graph_objects as go
import numpy as np
import open3d as o3d

st.set_page_config(page_title="Metric 3D Complete Analysis", layout="wide")
st.title(" Full Metric Reconstruction Analysis")

folder = 'test_outputs'

# --- SECTION 1: VISUAL DEPTH MAPS ---
st.header("1. Depth Map Pipeline")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Comparison View")
    img_path = os.path.join(folder, 'comparison_idx0.png')
    if os.path.exists(img_path):
        st.image(img_path, caption="RGB | Relative | Fused | GT", use_column_width=True)

with col2:
    st.subheader("Error Heatmap")
    err_path = os.path.join(folder, 'error_map_idx0.png')
    if os.path.exists(err_path):
        st.image(err_path, caption="Red = High Error | Blue = Accurate", use_column_width=True)

st.divider()

# --- SECTION 2: METRICS & GRAPHS ---
st.header("2. Performance Metrics")
m_col1, m_col2 = st.columns([1, 2])

with m_col1:
    st.subheader("Numerical Results")
    eval_path = os.path.join(folder, 'eval_results.txt')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            st.text(f.read())
    
    st.subheader("Sparsity Data")
    sparse_txt = os.path.join(folder, 'sparsity_data.txt')
    if os.path.exists(sparse_txt):
        with open(sparse_txt, 'r') as f:
            st.text(f.read())

with m_col2:
    st.subheader("Sensitivity Analysis")
    sens_path = os.path.join(folder, 'sparsity_sensitivity.png')
    if os.path.exists(sens_path):
        st.image(sens_path, use_column_width=True)

st.divider()

# --- SECTION 3: THE DENSE CLOUD ---
st.header("3. High-Density 3D Fused Cloud")
ply_path = os.path.join(folder, 'pointcloud_idx0.ply')

if os.path.exists(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    clr = np.asarray(pcd.colors)

    # We take enough points to look dense (200k), but not so many it crashes
    limit = 200000
    pts = pts[:limit]
    clr = clr[:limit]

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=clr, opacity=1)
    )])
    
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, 
                   bgcolor="black", aspectmode='data'),
        paper_bgcolor="black", margin=dict(l=0, r=0, b=0, t=0), height=800
    )
    st.plotly_chart(fig, use_container_width=True)