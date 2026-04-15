import streamlit as st
import os
import plotly.graph_objects as go
import numpy as np
import open3d as o3d

st.set_page_config(page_title="Metric 3D Analysis", layout="wide")

st.title(" High-Resolution Metric Reconstruction")

folder = 'test_outputs'

# --- 3D RECONSTRUCTION (FULL DENSITY) ---
st.header("Interactive Fused Point Cloud")
ply_path = os.path.join(folder, 'pointcloud_idx0.ply')

if os.path.exists(ply_path):
    # Load the full model
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # NO DOWNSAMPLING - we keep every single point
    pts = np.asarray(pcd.points)
    clr = np.asarray(pcd.colors)

    # Use Scatter3d with 'gl' markers for better performance with high point counts
    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=1.8,          # Slightly larger size helps "fill" the gaps
            color=clr, 
            opacity=1.0        # Solid points
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_visible=False, 
            yaxis_visible=False, 
            zaxis_visible=False, 
            bgcolor="black",
            aspectmode='data'
        ),
        paper_bgcolor="black",
        margin=dict(l=0, r=0, b=0, t=0),
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Point cloud file not found.")