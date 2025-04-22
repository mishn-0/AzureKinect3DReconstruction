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
        print("Preprocessing point cloud...")
        print(f"Initial point cloud has {len(pcd.points)} points")
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"After outlier removal: {len(pcd.points)} points")
        
        # Downsample point cloud
        pcd = pcd.voxel_down_sample(self.voxel_size)
        print(f"After downsampling: {len(pcd.points)} points")
        
        # Estimate normals
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        print("Normals estimated")
        
        return pcd
    
    def create_mesh_poisson(self, pcd, depth=9, width=0, scale=1.1, linear_fit=False):
        print("Creating mesh using Poisson reconstruction...")
        print(f"Input point cloud has {len(pcd.points)} points")
        
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )
        
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"After removing low density vertices: {len(mesh.vertices)} vertices")
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh
    
    def create_mesh_ball_pivoting(self, pcd, radii):
        print("Creating mesh using Ball Pivoting...")
        print(f"Input point cloud has {len(pcd.points)} points")
        
        # Create mesh using Ball Pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
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
        print(f"Output folder: {os.path.abspath(self.output_folder)}")
        
        # Kinect camera intrinsics (using PrimeSense default as starting point)
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
        
        # Flip transform for Kinect coordinate system
        self.flip_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Initialize visualization
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Kinect Mesh Reconstruction", 1280, 720)
        self.register_callbacks()
        
        # Set up render options
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
        opt.show_coordinate_frame = True  # Show coordinate frame for reference
        
    def setup_camera_view(self):
        """Set up the camera view to match the Kinect's perspective"""
        ctr = self.vis.get_view_control()
        # Set camera parameters to match Kinect's view
        ctr.set_front([0.0, 0.0, -1.0])  # Looking along negative z-axis
        ctr.set_up([0.0, -1.0, 0.0])     # Up direction
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
            print(f"Visualization mode: {self.visualization_mode}")
            return True
            
        def save_current_state(vis):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save point cloud
            if self.integrated_pcd is not None:
                pcd_path = os.path.join(self.output_folder, f"reconstruction_{timestamp}_pcd.ply")
                print(f"Saving point cloud to: {os.path.abspath(pcd_path)}")
                print(f"Point cloud has {len(self.integrated_pcd.points)} points")
                o3d.io.write_point_cloud(pcd_path, self.integrated_pcd)
                print(f"Saved point cloud to: {pcd_path}")
                
                # Also save a copy with fixed name for LastReconstructionVisualizer
                fixed_pcd_path = os.path.join(self.output_folder, "latest_kinect_reconstruction.ply")
                o3d.io.write_point_cloud(fixed_pcd_path, self.integrated_pcd)
                print(f"Saved point cloud to: {fixed_pcd_path}")
            
            # Save mesh if available
            if self.integrated_mesh is not None:
                mesh_path = os.path.join(self.output_folder, f"reconstruction_{timestamp}_mesh.obj")
                print(f"Saving mesh to: {os.path.abspath(mesh_path)}")
                print(f"Mesh has {len(self.integrated_mesh.vertices)} vertices and {len(self.integrated_mesh.triangles)} triangles")
                o3d.io.write_triangle_mesh(mesh_path, self.integrated_mesh)
                print(f"Saved mesh to: {mesh_path}")
            
            return True
        
        # Register key callbacks
        self.vis.register_key_callback(ord("M"), toggle_mesh_reconstruction)  # M to toggle mesh reconstruction
        self.vis.register_key_callback(ord("V"), toggle_visualization_mode)   # V to toggle visualization mode
        self.vis.register_key_callback(ord("S"), save_current_state)         # S to save current state
        self.vis.register_key_callback(ord("R"), lambda vis: self.setup_camera_view() or True)  # R to reset view
    
    def process_frame(self, color_img, depth_img):
        """Process a single frame from the Kinect"""
        # Create RGBD image
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,
            depth_trunc=3.0
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics)
        pcd.transform(self.flip_transform)
        
        print(f"Created point cloud with {len(pcd.points)} points")
        return pcd, rgbd
    
    def update_visualization(self, pcd):
        """Update the visualization with the current point cloud or mesh"""
        self.vis.clear_geometries()
        
        if self.visualization_mode == "mesh" and self.mesh_reconstruction:
            try:
                # Preprocess point cloud
                processed_pcd = self.mesh_converter.preprocess_point_cloud(pcd)
                
                # Try Poisson reconstruction first
                try:
                    mesh = self.mesh_converter.create_mesh_poisson(processed_pcd)
                    self.integrated_mesh = mesh
                    self.vis.add_geometry(mesh)
                except Exception as e:
                    print(f"Poisson reconstruction failed: {e}")
                    print("Trying Ball Pivoting...")
                    radii = [0.005, 0.01, 0.02, 0.04]
                    mesh = self.mesh_converter.create_mesh_ball_pivoting(processed_pcd, radii)
                    self.integrated_mesh = mesh
                    self.vis.add_geometry(mesh)
            except Exception as e:
                print(f"Mesh reconstruction failed: {e}")
                self.vis.add_geometry(pcd)
        else:
            self.vis.add_geometry(pcd)
            self.integrated_pcd = pcd
        
        # Set up initial camera view
        self.setup_camera_view()
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        """Main loop for Kinect streaming and reconstruction"""
        global running
        
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize Kinect
        print("Initializing Kinect camera...")
        k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        k4a.start()
        
        print("Kinect initialized. Press Ctrl+C to stop.")
        print("Controls:")
        print("  M - Toggle mesh reconstruction")
        print("  V - Toggle visualization mode (point cloud/mesh)")
        print("  S - Save current reconstruction")
        print("  R - Reset camera view")
        
        try:
            while running:
                # Get new frame
                capture = k4a.get_capture()
                if capture.color is None or capture.transformed_depth is None:
                    continue
                
                # Process frame
                pcd, _ = self.process_frame(capture.color, capture.transformed_depth)
                
                # Update visualization
                self.update_visualization(pcd)
                
        except Exception as e:
            print(f"Error during streaming: {e}")
        
        finally:
            k4a.stop()
            self.vis.destroy_window()
            print("Streaming stopped")

def main():
    reconstructor = KinectMeshReconstructor(voxel_size=0.01)
    reconstructor.run()

if __name__ == "__main__":
    main() 