import numpy as np
import open3d as o3d
import cv2
import os
import sys
import argparse
from datetime import datetime
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import signal
import time

# Global variables for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping camera stream...")
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

def process_depth_to_pointcloud(depth_img, color_img=None, depth_scale=1000.0, depth_trunc=3.0):
    """
    Convert depth image to point cloud with optional color information
    
    Args:
        depth_img: Depth image (in millimeters)
        color_img: Optional color image (BGR format)
        depth_scale: Scale factor for depth values (default: 1000.0 for Kinect)
        depth_trunc: Maximum depth to consider (in meters)
        
    Returns:
        point_cloud: Open3D point cloud
    """
    # Create Open3D images
    depth_o3d = o3d.geometry.Image(depth_img)
    
    # Process color image if provided
    if color_img is not None:
        # Convert BGR to RGB
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_o3d = o3d.geometry.Image(color_rgb)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc
        )
        
        # Create point cloud from RGBD image
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
        )
    else:
        # Create point cloud from depth image only
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            ),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc
        )
    
    # Flip the point cloud to correct orientation (same as in the Jupyter notebook)
    flip_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    point_cloud.transform(flip_transform)
    
    return point_cloud

def process_frame(color_img, depth_img, save_ply=False, output_dir=None, frame_number=0):
    """
    Process a single frame from the Kinect camera
    
    Args:
        color_img: Color image from Kinect
        depth_img: Depth image from Kinect
        save_ply: Whether to save the point cloud as PLY file
        output_dir: Directory to save output files
        frame_number: Frame number for saving files
        
    Returns:
        point_cloud: Open3D point cloud
    """
    # Create point cloud
    point_cloud = process_depth_to_pointcloud(depth_img, color_img)
    
    # Save point cloud if requested
    if save_ply and output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename - use a naming convention that LastReconstructionVisualizer.py will recognize
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"kinect_reconstruction_pcd_{timestamp}.ply")
        
        # Save point cloud
        o3d.io.write_point_cloud(output_path, point_cloud)
        print(f"Saved point cloud to: {output_path}")
        
        # Also save a copy with a fixed name for easy access
        fixed_output_path = os.path.join(output_dir, "latest_kinect_reconstruction.ply")
        o3d.io.write_point_cloud(fixed_output_path, point_cloud)
        print(f"Saved point cloud to: {fixed_output_path}")
    
    return point_cloud

def stream_from_kinect(save_ply=False, output_dir=None, visualize=True, save_frames=False, frames_dir=None):
    """
    Stream from Kinect camera and convert to point clouds in real-time
    
    Args:
        save_ply: Whether to save point clouds as PLY files
        output_dir: Directory to save PLY files
        visualize: Whether to visualize point clouds
        save_frames: Whether to save raw frames
        frames_dir: Directory to save raw frames
    """
    global running
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize Kinect
    print("Initializing Kinect camera...")
    k4a = setup_kinect()
    if not k4a:
        print("Failed to initialize Kinect camera. Exiting...")
        return
    
    print("Kinect camera initialized successfully!")
    
    # Create output directories if needed
    if save_ply and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving point clouds to: {os.path.abspath(output_dir)}")
    
    if save_frames and frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {os.path.abspath(frames_dir)}")
    
    # Initialize visualization
    vis = None
    if visualize:
        print("Initializing visualization...")
        vis = o3d.visualization.Visualizer()
        vis.create_window("Kinect Point Cloud", width=1280, height=720)
        print("Visualization window created")
    
    frame_count = 0
    print("Starting Kinect stream...")
    print("Press Ctrl+C to stop")
    
    try:
        while running:
            # Get new frame
            capture = k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                print("No frame received, retrying...")
                time.sleep(0.1)
                continue
            
            # Save raw frames if requested
            if save_frames and frames_dir:
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.npz")
                np.savez(frame_path, color=capture.color, depth=capture.transformed_depth)
            
            # Process frame
            point_cloud = process_frame(
                capture.color, 
                capture.transformed_depth,
                save_ply=save_ply,
                output_dir=output_dir,
                frame_number=frame_count
            )
            
            # Update visualization
            if visualize and vis is not None:
                vis.clear_geometries()
                vis.add_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames")
    
    except Exception as e:
        print(f"Error during streaming: {e}")
    
    finally:
        # Clean up
        if vis is not None:
            vis.destroy_window()
        k4a.stop()
        print("Streaming stopped")

def process_frame_file(frame_path, output_dir=None, save_ply=True, visualize=True):
    """
    Process a single frame file and convert to point cloud
    
    Args:
        frame_path: Path to the .npz frame file
        output_dir: Directory to save output files (default: same directory as frame)
        save_ply: Whether to save the point cloud as PLY file
        visualize: Whether to visualize the point cloud
        
    Returns:
        point_cloud: Open3D point cloud
    """
    # Load frame data
    frame_data = np.load(frame_path)
    depth_img = frame_data['depth']
    
    # Check if color data exists
    color_img = None
    if 'color' in frame_data:
        color_img = frame_data['color']
    
    # Create point cloud
    point_cloud = process_depth_to_pointcloud(depth_img, color_img)
    
    # Save point cloud if requested
    if save_ply:
        if output_dir is None:
            output_dir = os.path.dirname(frame_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename - use a naming convention that LastReconstructionVisualizer.py will recognize
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"kinect_reconstruction_pcd_{timestamp}.ply")
        
        # Save point cloud
        o3d.io.write_point_cloud(output_path, point_cloud)
        print(f"Saved point cloud to: {output_path}")
        
        # Also save a copy with a fixed name for easy access
        fixed_output_path = os.path.join(output_dir, "latest_kinect_reconstruction.ply")
        o3d.io.write_point_cloud(fixed_output_path, point_cloud)
        print(f"Saved point cloud to: {fixed_output_path}")
    
    # Visualize point cloud if requested
    if visualize:
        o3d.visualization.draw_geometries([point_cloud])
    
    return point_cloud

def process_directory(input_dir, output_dir=None, save_ply=True, visualize=False):
    """
    Process all frame files in a directory
    
    Args:
        input_dir: Directory containing frame files
        output_dir: Directory to save output files
        save_ply: Whether to save the point clouds as PLY files
        visualize: Whether to visualize each point cloud
    """
    # Get all .npz files in the directory
    frame_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    if not frame_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(frame_files)} frame files")
    
    # Process each frame
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_dir, frame_file)
        print(f"Processing frame {i+1}/{len(frame_files)}: {frame_file}")
        
        try:
            process_frame_file(frame_path, output_dir, save_ply, visualize)
        except Exception as e:
            print(f"Error processing {frame_file}: {e}")

def main():
    # Default settings
    save_ply = True
    output_dir = "results"  # Changed to "results" to match LastReconstructionVisualizer.py
    visualize = True
    save_frames = True
    frames_dir = "frames"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    print("=" * 50)
    print("Kinect Point Cloud Generator")
    print("=" * 50)
    print(f"Saving point clouds to: {os.path.abspath(output_dir)}")
    print(f"Saving frames to: {os.path.abspath(frames_dir)}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Stream from Kinect
    stream_from_kinect(
        save_ply=save_ply,
        output_dir=output_dir,
        visualize=visualize,
        save_frames=save_frames,
        frames_dir=frames_dir
    )

if __name__ == "__main__":
    main()