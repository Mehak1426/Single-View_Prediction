import streamlit as st
import os
import plotly.graph_objects as go
import numpy as np
import open3d as o3d

st.set_page_config(page_title="3D Reconstruction", layout="wide")
st.title(" Metric 3D Reconstruction Dashboard")


folder = 'test_outputs'


st.header("1. Depth Pipeline Comparison")
img_path = os.path.join(folder, 'comparison_idx0.png')

if os.path.exists(img_path):
    st.image(img_path, use_column_width=True, caption="Pipeline: Predicted -> Fused (Metric) -> Ground Truth")
else:
    st.error(f"Cannot find {img_path}. Check if the folder name is correct.")

st.divider()


st.header("2. Interactive 3D Model")
ply_path = os.path.join(folder, 'pointcloud_idx0.ply')

if os.path.exists(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    clr = np.asarray(pcd.colors)
    
    fig = go.Figure(data=[go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], 
                          mode='markers', marker=dict(size=1.2, color=clr))])
    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), 
                      paper_bgcolor="black", margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)