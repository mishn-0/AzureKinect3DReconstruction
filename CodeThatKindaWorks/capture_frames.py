import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import numpy as np
import os
import sys
import cv2

print("Initializing...")

# Initialize the Azure Kinect DK
k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P,
                  depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                  synchronized_images_only=True))
k4a.start()

# Initialize TSDF Volume
voxel_length = 0.005
sdf_trunc = 0.04
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# Initialize odometry
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
odo_init = np.identity(4)

# Initialize pose graph
pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

# Create necessary directories
os.makedirs('data/pose_graphs', exist_ok=True)
os.makedirs('data/rgbd_frames', exist_ok=True)

def process_color_image(color_np):
    """Process color image to ensure it's in the correct format."""
    # Convert RGBA to RGB if necessary
    if color_np.shape[-1] == 4:
        print("Converting RGBA to RGB")
        color_np = color_np[:, :, :3]
    
    # Ensure the array is contiguous
    if not color_np.flags['C_CONTIGUOUS']:
        print("Making color array contiguous")
        color_np = np.ascontiguousarray(color_np)
    
    return color_np

def process_depth_image(depth_np):
    """Process depth image to ensure it's in the correct format."""
    # Ensure the array is contiguous
    if not depth_np.flags['C_CONTIGUOUS']:
        print("Making depth array contiguous")
        depth_np = np.ascontiguousarray(depth_np)
    
    return depth_np

# Function to save pose graph
def save_pose_graph(pose_graph, index):
    filename = os.path.join('data', 'pose_graphs', f'pose_graph_{index:04d}.json')
    o3d.io.write_pose_graph(filename, pose_graph)

# Function to save RGBD frame
def save_rgbd_frame(rgbd_image, index):
    filename = os.path.join('data', 'rgbd_frames', f'frame_{index:04d}.npz')
    np.savez(filename, 
             color=np.asarray(rgbd_image.color),
             depth=np.asarray(rgbd_image.depth))

# Function to optimize pose graph
def optimize_pose_graph(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.02,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.0,
        reference_node=0
    )
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        method,
        criteria,
        option
    )

# Function to reintegrate frames
def reintegrate_frames(volume, pose_graph, intrinsic):
    # Clear the volume
    volume.reset()
    
    # Reintegrate all frames using optimized poses
    for i, node in enumerate(pose_graph.nodes):
        frame_path = os.path.join('data', 'rgbd_frames', f'frame_{i:04d}.npz')
        if os.path.exists(frame_path):
            frame_data = np.load(frame_path)
            color = o3d.geometry.Image(frame_data['color'])
            depth = o3d.geometry.Image(frame_data['depth'])
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, convert_rgb_to_intensity=False)
            volume.integrate(rgbd_image, intrinsic, np.linalg.inv(node.pose))

print("Starting capture loop...")

# Main capture loop
frame_index = 0
try:
    while True:
        # Capture a frame
        capture = k4a.get_capture()
        if capture.color is not None and capture.depth is not None:
            try:
                # Process images
                color_np = process_color_image(capture.color)
                depth_np = process_depth_image(capture.transformed_depth)
                
                print("\nProcessing new frame...")
                print(f"Color image shape: {color_np.shape}, dtype: {color_np.dtype}")
                print(f"Depth image shape: {depth_np.shape}, dtype: {depth_np.dtype}")
                print(f"Color array flags: {color_np.flags}")
                print(f"Depth array flags: {depth_np.flags}")
                
                # Convert to Open3D images
                color_image = o3d.geometry.Image(color_np)
                depth_image = o3d.geometry.Image(depth_np)
                    
                # Convert to RGBD image
                print("Creating RGBD image...")
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image, depth_image, 
                    convert_rgb_to_intensity=False,
                    depth_scale=1000.0,  # Azure Kinect depth is in mm
                    depth_trunc=3.0  # 3m truncation
                )
                print("RGBD image created")

                # Save RGBD frame
                save_rgbd_frame(rgbd_image, frame_index)

                # Perform RGBD odometry
                if 'prev_rgbd' in locals():
                    try:
                        print("Computing odometry...")
                        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                            rgbd_image, prev_rgbd, intrinsic, odo_init,
                            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                            o3d.pipelines.odometry.OdometryOption())
                        
                        if success:
                            print("Odometry computed successfully")
                            odo_init = trans
                            try:
                                volume.integrate(rgbd_image, intrinsic, np.linalg.inv(odo_init))
                                print("Frame integrated into volume")
                            except Exception as e:
                                print(f"Warning: Failed to integrate frame into volume: {e}")
                                continue

                            # Add edge to pose graph
                            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odo_init)))
                            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                                frame_index - 1, frame_index, trans, info, uncertain=False))
                            
                            if frame_index % 10 == 0:  # Print progress more frequently
                                print(f"Successfully processed frame {frame_index}")
                        else:
                            print(f"Warning: RGBD odometry failed for frame {frame_index}. Skipping frame.")
                            continue
                    except Exception as e:
                        print(f"Warning: Error during odometry computation for frame {frame_index}: {e}")
                        continue

                prev_rgbd = rgbd_image
                frame_index += 1

                # Save pose graph periodically
                if frame_index % 100 == 0:
                    save_pose_graph(pose_graph, frame_index)
                    print(f"Saved pose graph at frame {frame_index}")
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

except KeyboardInterrupt:
    print("\nStopping capture...")
finally:
    k4a.stop()
    # Optimize pose graph
    print("Optimizing pose graph...")
    optimize_pose_graph(pose_graph)
    
    # Reintegrate frames using optimized poses
    print("Reintegrating frames...")
    reintegrate_frames(volume, pose_graph, intrinsic)
    
    # Extract and save final results
    print("Saving final reconstruction...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("bundlefusion_reconstruction_mesh.ply", mesh)
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud("bundlefusion_reconstruction_pcd.ply", pcd)
    print("Reconstruction complete!") 