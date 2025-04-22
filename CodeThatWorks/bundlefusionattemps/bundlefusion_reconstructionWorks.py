import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time
import signal
import sys
import os
from datetime import datetime

# Global variables for graceful shutdown
running = True
current_frame = 0
frames_dir = "frames"
pose_graph_file = "pose_graph.json"

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping reconstruction...")
    running = False

def setup_kinect():
    """Initialize and configure Azure Kinect"""
    try:
        k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        k4a.start()
        return k4a
    except Exception as e:
        print(f"Error initializing Kinect: {e}")
        return None

def create_output_directories():
    """Create necessary directories for storing frames and results"""
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    if not os.path.exists("results"):
        os.makedirs("results")

def process_frame(color_img, depth_img):
    """Convert Kinect frames to Open3D RGBD format"""
    # Flip images to correct orientation
    color_img = cv2.flip(color_img, -1)
    depth_img = cv2.flip(depth_img, -1)
    
    # Convert color format - Azure Kinect uses BGRA format
    if len(color_img.shape) == 3:
        if color_img.shape[2] == 4:  # BGRA format
            # First convert BGRA to BGR
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
            # Then convert BGR to RGB
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        elif color_img.shape[2] == 3:  # BGR format
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    
    # Create Open3D images
    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_img)

    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=5.0,
    )
    return rgbd, color_img

def setup_tsdf_volume():
    """Initialize TSDF volume with CUDA support if available"""
    try:
        # In ai-cold-env, we'll use the standard CPU implementation
        print("Using CPU TSDF volume")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.004,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        return volume
    except Exception as e:
        print(f"Error setting up TSDF volume: {e}")
        return None

def setup_odometry():
    """Initialize RGBD odometry parameters"""
    int_vector = o3d.utility.IntVector([20, 10, 5])
    return (
        o3d.pipelines.odometry.OdometryOption(
            iteration_number_per_pyramid_level=int_vector,
            max_depth_diff=0.03,
            min_depth=0.0,
            max_depth=4.0
        ),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    )

def main():
    global running, current_frame
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create output directories
    create_output_directories()
    
    # Initialize Kinect
    k4a = setup_kinect()
    if not k4a:
        return
    
    # Initialize TSDF volume
    volume = setup_tsdf_volume()
    if not volume:
        k4a.stop()
        return
    
    # Initialize odometry
    odometry_option, jacobian = setup_odometry()
    pose_graph = o3d.pipelines.registration.PoseGraph()
    
    # Get first frame for initialization
    capture = k4a.get_capture()
    if capture.color is None or capture.transformed_depth is None:
        print("Failed to get initial frame")
        k4a.stop()
        return
    
    # Process first frame
    rgbd, _ = process_frame(capture.color, capture.transformed_depth)
    
    # Initialize camera intrinsics
    width, height = capture.color.shape[1], capture.color.shape[0]
    fx, fy = width * 1.03, width * 1.03  # approximation for Kinect
    cx, cy = width / 2, height / 2
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Initialize first pose
    current_pose = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(current_pose))
    
    print("Starting reconstruction...")
    print("Press Ctrl+C to stop")
    
    try:
        while running:
            # Get new frame
            capture = k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue
            
            # Process frame
            rgbd, color_img = process_frame(capture.color, capture.transformed_depth)
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{current_frame:06d}.npz")
            np.savez(frame_path, color=color_img, depth=capture.transformed_depth)
            
            # Perform odometry
            if current_frame > 0:
                success, transformation, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd, rgbd, intrinsics, current_pose,
                    jacobian, odometry_option
                )
                
                if success:
                    current_pose = transformation
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(current_pose)
                    )
                    
                    # Add edge to pose graph
                    if current_frame > 1:
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                current_frame - 1, current_frame,
                                transformation,
                                np.identity(6)
                            )
                        )
            
            # Integrate into TSDF volume
            volume.integrate(rgbd, intrinsics, current_pose)
            
            current_frame += 1
            if current_frame % 10 == 0:
                print(f"Processed {current_frame} frames")
    
    except Exception as e:
        print(f"Error during reconstruction: {e}")
    
    finally:
        # Save pose graph
        o3d.io.write_pose_graph(pose_graph_file, pose_graph)
        
        # Optimize pose graph
        print("Optimizing pose graph...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=2.0,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )
        
        # Reintegrate frames with optimized poses
        print("Reintegrating frames with optimized poses...")
        volume.reset()
        for i in range(len(pose_graph.nodes)):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.npz")
            frame_data = np.load(frame_path)
            rgbd, _ = process_frame(frame_data['color'], frame_data['depth'])
            volume.integrate(rgbd, intrinsics, pose_graph.nodes[i].pose)
        
        # Extract and save results
        print("Extracting mesh and point cloud...")
        mesh = volume.extract_triangle_mesh()
        pcd = volume.extract_point_cloud()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mesh_path = f"results/bundlefusion_reconstruction_mesh_{timestamp}.ply"
        pcd_path = f"results/bundlefusion_reconstruction_pcd_{timestamp}.ply"
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        o3d.io.write_point_cloud(pcd_path, pcd)
        
        print(f"Saved mesh to {mesh_path}")
        print(f"Saved point cloud to {pcd_path}")
        
        # Clean up
        k4a.stop()
        print("Reconstruction completed")

if __name__ == "__main__":
    main() 