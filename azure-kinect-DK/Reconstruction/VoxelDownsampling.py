import open3d as o3d
import numpy as np
import plotly.graph_objects as go


# READ POINT CLOUD
print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
points = np.asarray(pcd.points)
print(pcd)
print(np.asarray(pcd.points))

# DOWNSAMPLING

print("Downsample the point cloud with a voxel of 0.02")
downpcd = pcd.voxel_down_sample(voxel_size=0.02)
ddown_points = np.asarray(downpcd.points)downpcd = pcd.voxel_down_sample(voxel_size=0.02)

# VISUALIZE

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=down_points[:,0], y=down_points[:,1], z=down_points[:,2], 
            mode='markers',
            marker=dict(size=1)
        )
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
)
fig.show()