import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time
import threading
import sys

# Configure Kinect
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# Flag to control the visualization loop
running = True

# Process images for visualization
def process_images(color_img, depth_img):
    # Flip images
    color_img = cv2.flip(color_img, -1)
    depth_img = cv2.flip(depth_img, -1)
    
    # Convert color format - Azure Kinect uses BGRA format
    if len(color_img.shape) == 3:
        if color_img.shape[2] == 4:  # BGRA format
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2RGB)
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

# Function to apply position-based coloring (uncomment use if RGB is not working)
def colorize_pcd(pcd, rgb_image):
    points = np.asarray(pcd.points)
    
    # Normalize each dimension to [0,1]
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # Avoid division by zero
    normalized = (points - min_vals) / ranges
    
    # Use different dimensions for different color channels
    colors = np.zeros_like(points)
    colors[:, 0] = normalized[:, 0]  # X dimension → Red
    colors[:, 1] = normalized[:, 1]  # Y dimension → Green
    colors[:, 2] = normalized[:, 2]  # Z dimension → Blue
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    global running
    
    # Create visualizer with explicit RGB rendering enabled
    vis = o3d.visualization.Visualizer()
    vis.create_window("RGB Point Cloud Real-time Visualization")
    
    # Set rendering options to ensure RGB colors are displayed
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # Wait for a valid first capture
    while True:
        capture = k4a.get_capture()
        if capture.color is not None and capture.transformed_depth is not None:
            break
    
    # Transformation to align properly (floor down)
    flip_transform = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    # Get dimensions from the color image
    width, height = capture.color.shape[1], capture.color.shape[0]
    
    # Try to get camera calibration from k4a or use approximation
    try:
        calib = k4a.calibration
        
        if hasattr(calib, "get_camera_matrix"):
            camera_matrix = calib.get_camera_matrix(3)  # 3 for color camera
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        else:
            fx, fy = width * 1.03, width * 1.03  # approximation
            cx, cy = width / 2, height / 2       # approximation
    except Exception:
        # Fallback to default intrinsics
        fx, fy = width * 1.03, width * 1.03  # approximation for Kinect
        cx, cy = width / 2, height / 2
    
    # Create camera intrinsics
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Initialize the point cloud
    color = capture.color
    depth = capture.transformed_depth
    rgbd, rgb_image = process_images(color, depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.transform(flip_transform)
    
    # Add initial geometry
    vis.add_geometry(pcd)
    
    # Set initial camera view
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_zoom(0.5)
    
    # Store the last window active state
    last_active = True
    
    # Main visualization loop
    try:
        while running:
            # Step 1: Poll events at the start of each loop
            is_active = vis.poll_events()
            
            # Step 2: Check if window was closed
            if not is_active and last_active:
                running = False
                break
            
            last_active = is_active
            
            # Step 3: Try to get capture
            try:
                capture = k4a.get_capture()
                if capture.color is None or capture.transformed_depth is None:
                    continue
            except Exception as e:
                print(f"Capture error: {e}")
                continue

            # Step 4: Process images and create point cloud
            try:
                color = capture.color
                depth = capture.transformed_depth
                rgbd, rgb_image = process_images(color, depth)

                # Create new point cloud from RGBD image
                new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
                new_pcd.transform(flip_transform)
                
                # Uncomment the next line if you want to use position-based coloring instead of RGB
                # new_pcd = colorize_pcd(new_pcd, rgb_image)
                
                # Update the point cloud
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
            except Exception as e:
                print(f"Point cloud error: {e}")
                continue

            # Step 5: Update visualization
            try:
                vis.update_geometry(pcd)
                vis.update_renderer()
            except Exception as e:
                print(f"Visualization error: {e}")
                continue
                
            # Sleep to reduce CPU usage and keep frame rate reasonable
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("Stopping visualization.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up
        running = False
        
        # First stop Kinect
        try:
            k4a.stop()
        except Exception as e:
            print(f"Error stopping Kinect: {e}")
            
        # Then destroy visualization
        try:
            vis.destroy_window()
        except Exception as e:
            print(f"Error destroying window: {e}")
            
        print("Visualization closed.")

if __name__ == "__main__":
    main()