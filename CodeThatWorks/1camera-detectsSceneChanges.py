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

class PointCloudToMesh:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size
        
    def preprocess_point_cloud(self, pcd):
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample point cloud
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        
        return pcd
    
    def create_mesh_poisson(self, pcd, depth=9, width=0, scale=1.1, linear_fit=False):
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh
    
    def create_mesh_ball_pivoting(self, pcd, radii):
        # Create mesh using Ball Pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh

class KinectMeshReconstructor:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size
        self.mesh_converter = PointCloudToMesh(voxel_size)
        self.volume = None
        self.integrated_mesh = None
        self.integrated_pcd = None
        self.visualization_mode = "pointcloud"  # Start with point cloud visualization
        self.mesh_reconstruction = False
        self.output_folder = "results"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.04,  # 4cm voxel size
            sdf_trunc=0.04,     # 4cm truncation
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Initialize Kinect camera
        self.k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        
        # Start Kinect and get calibration
        self.k4a.start()
        self.calibration = self.k4a.calibration
        
        # Create camera intrinsics from Kinect calibration
        try:
            # Try to get camera matrix directly from calibration or use properties
            color_params = self.calibration.color_camera_calibration.intrinsics.parameters
            fx, fy = color_params.param.fx, color_params.param.fy
            cx, cy = color_params.param.cx, color_params.param.cy
            print(f"Using calibrated intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        except Exception as e:
            # Fallback to default values if calibration access fails
            print(f"Error accessing calibration: {e}")
            print("Using default camera intrinsics")
            fx, fy = 605.286, 605.699  # Default values for Kinect
            cx, cy = 637.134, 366.758  # Default values for Kinect
        
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=1280,  # Color camera width
            height=720,  # Color camera height
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        
        # Get color to depth transformation
        try:
            # Try to use PyK4A's transformation functions instead
            # Example: Just use the device's built-in transformation
            print("Using built-in depth to color transformation from PyK4A")
            # We'll apply the transformation at capture time using transformed_depth
            self.color_to_depth_transform = np.eye(4)  # Identity as placeholder
        except Exception as e:
            print(f"Error with transformation setup: {e}")
            # Use identity matrix as fallback
            self.color_to_depth_transform = np.eye(4)
            print("Using identity matrix for color to depth transform")
        
        # Flip transform for Kinect coordinate system
        self.flip_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Initialize odometry with updated parameters for newer Open3D versions
        self.odometry_option = o3d.pipelines.odometry.OdometryOption()
        # Set parameters correctly for newer Open3D versions
        # Modern Open3D doesn't use depth_scale in OdometryOption
        # Instead, it's handled during RGBD image creation
        
        # Initialize previous frame for odometry
        self.prev_rgbd = None
        self.prev_transform = np.eye(4)
        self.current_transform = np.eye(4)
        
        # Initialize visualization
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Kinect Mesh Reconstruction", 1280, 720)
        self.register_callbacks()
        
        # Set up render options
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
        opt.show_coordinate_frame = False  # Hide coordinate frame
        
    def setup_camera_view(self):
        """Set up the camera view to match the Kinect's perspective"""
        ctr = self.vis.get_view_control()
        # Set camera parameters to match Kinect's view
        ctr.set_front([0.0, 0.0, -1.0])  # Looking along negative z-axis
        ctr.set_up([0.0, 1.0, 0.0])      # Up direction (fixed: y-axis is up)
        ctr.set_lookat([0.0, 0.0, 1.0])  # Look at point 1 meter in front
        ctr.set_zoom(0.7)                # Adjust zoom level
        
    def register_callbacks(self):
        """Register keyboard callbacks for the visualizer"""
        def toggle_mesh_reconstruction(vis):
            self.mesh_reconstruction = not self.mesh_reconstruction
            print(f"Mesh reconstruction: {'enabled' if self.mesh_reconstruction else 'disabled'}")
            return True
            
        def toggle_visualization_mode(vis):
            self.visualization_mode = "mesh" if self.visualization_mode == "pointcloud" else "pointcloud"
            return True
            
        def save_current_state(vis):
            # Extract point cloud from TSDF volume
            pcd = self.volume.extract_point_cloud()
            if pcd is not None:
                pcd_path = os.path.join(self.output_folder, "latest_kinect_reconstruction.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                print("Point cloud saved")
            
            # Extract mesh from TSDF volume if mesh reconstruction is enabled
            if self.mesh_reconstruction:
                mesh = self.volume.extract_triangle_mesh()
                if mesh is not None:
                    mesh.compute_vertex_normals()
                    mesh_path = os.path.join(self.output_folder, "latest_kinect_reconstruction.obj")
                    o3d.io.write_triangle_mesh(mesh_path, mesh)
                    print("Mesh saved")
            
            return True
        
        def reset_reconstruction(vis):
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.04,
                sdf_trunc=0.04,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            self.prev_rgbd = None
            self.prev_transform = np.eye(4)
            self.current_transform = np.eye(4)
            print("Reconstruction reset")
            return True
        
        # Register key callbacks
        self.vis.register_key_callback(ord("M"), toggle_mesh_reconstruction)  # M to toggle mesh reconstruction
        self.vis.register_key_callback(ord("V"), toggle_visualization_mode)   # V to toggle visualization mode
        self.vis.register_key_callback(ord("S"), save_current_state)         # S to save current state
        self.vis.register_key_callback(ord("R"), lambda vis: self.setup_camera_view() or True)  # R to reset view
        self.vis.register_key_callback(ord("C"), reset_reconstruction)       # C to clear reconstruction
    
    def process_frame(self, color_img, depth_img):
        """Process a single frame from the Kinect"""
        # Create RGBD image
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,  # Apply depth scale here
            depth_trunc=3.0
        )
        
        return rgbd
    
    def compute_odometry(self, current_rgbd):
        """Compute camera movement between frames using RGBD odometry"""
        if self.prev_rgbd is None:
            self.prev_rgbd = current_rgbd
            return np.eye(4)
        
        # Compute odometry between previous and current frame using updated parameters
        # Note: depth_scale is now handled during RGBD image creation
        [success, transform, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            self.prev_rgbd, current_rgbd,
            self.intrinsics, self.prev_transform,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.odometry_option
        )
        
        # Update previous frame
        self.prev_rgbd = current_rgbd
        
        if success:
            # Update transforms
            self.prev_transform = transform
            self.current_transform = self.current_transform @ transform
            return transform
        else:
            # If odometry fails, return identity transform
            return np.eye(4)
    
    def update_visualization(self, rgbd):
        """Update the visualization with the current frame"""
        self.vis.clear_geometries()
        
        # Compute camera movement
        transform = self.compute_odometry(rgbd)
        
        # Integrate the new frame into the TSDF volume
        self.volume.integrate(
            rgbd,
            self.intrinsics,
            transform  # Use the computed transform
        )
        
        # Extract current visualization
        if self.visualization_mode == "mesh" and self.mesh_reconstruction:
            try:
                # Extract mesh from TSDF volume
                mesh = self.volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                self.vis.add_geometry(mesh)
            except Exception as e:
                print(f"Mesh extraction failed: {e}")
                # Fall back to point cloud
                pcd = self.volume.extract_point_cloud()
                self.vis.add_geometry(pcd)
        else:
            # Extract point cloud from TSDF volume
            pcd = self.volume.extract_point_cloud()
            self.vis.add_geometry(pcd)
        
        # Set up initial camera view
        self.setup_camera_view()
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        """Main loop for Kinect streaming and reconstruction"""
        global running
        
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        print("\nControls:")
        print("  M - Toggle mesh reconstruction")
        print("  V - Toggle visualization mode (point cloud/mesh)")
        print("  S - Save current reconstruction")
        print("  R - Reset camera view")
        print("  C - Clear reconstruction")
        print("  Ctrl+C - Stop streaming")
        
        try:
            while running:
                # Get new frame
                capture = self.k4a.get_capture()
                if capture.color is None or capture.transformed_depth is None:
                    continue
                
                # Process frame
                rgbd = self.process_frame(capture.color, capture.transformed_depth)
                
                # Update visualization
                self.update_visualization(rgbd)
                
        except Exception as e:
            print(f"Error during streaming: {e}")
        
        finally:
            self.k4a.stop()
            self.vis.destroy_window()
            print("Streaming stopped")

def main():
    reconstructor = KinectMeshReconstructor(voxel_size=0.01)
    reconstructor.run()

if __name__ == "__main__":
    main()