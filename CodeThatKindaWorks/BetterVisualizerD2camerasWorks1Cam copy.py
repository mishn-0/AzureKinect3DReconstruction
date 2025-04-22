import numpy as np
import open3d as o3d
import cv2
import os
import sys
import threading
import time
from datetime import datetime
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import signal
from queue import Queue

# Global variables for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping camera streams...")
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

class KinectCaptureThread(threading.Thread):
    """Thread to capture frames from a Kinect camera"""
    def __init__(self, device_id=0, queue_size=5):
        threading.Thread.__init__(self)
        self.daemon = True
        self.device_id = device_id
        self.frame_queue = Queue(maxsize=queue_size)
        self.k4a = None
        
    def initialize_camera(self):
        """Initialize the Kinect camera"""
        try:
            # Create config
            config = Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
            
            # Initialize with device ID
            self.k4a = PyK4A(config)
            
            # We need to set the device ID manually since K4A_DEVICE_DEFAULT isn't available
            # This is a workaround - in practice, PyK4A should work with the device index
            # when using the right version
            
            self.k4a.start()
            
            # Get the camera calibration for coordinate system alignment
            self.calibration = self.k4a.calibration
            print(f"Camera {self.device_id} initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize camera {self.device_id}: {e}")
            return False
            
    def run(self):
        """Main thread function to capture frames"""
        global running
        
        if not self.initialize_camera():
            print(f"Exiting thread for camera {self.device_id}")
            return
        
        print(f"Camera {self.device_id} streaming started")
        
        while running:
            try:
                # Get capture
                capture = self.k4a.get_capture()
                if capture.color is None or capture.transformed_depth is None:
                    continue
                
                # Put frame in queue, non-blocking
                if not self.frame_queue.full():
                    self.frame_queue.put((capture.color, capture.transformed_depth), block=False)
                
                # Don't max out CPU
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error capturing from camera {self.device_id}: {e}")
                time.sleep(1)  # Wait before trying again
        
        # Clean up
        if self.k4a:
            self.k4a.stop()
        print(f"Camera {self.device_id} streaming stopped")
    
    def get_latest_frame(self):
        """Get the latest frame from the queue"""
        if self.frame_queue.empty():
            return None
        
        # Get the most recent frame
        frame = self.frame_queue.get()
        
        # Clear the queue to always use the latest frame
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except:
                break
                
        return frame

class PointCloudRegistration:
    """Class for point cloud registration and alignment"""
    def __init__(self, voxel_size=0.05):
        self.voxel_size = voxel_size
        self.distance_threshold = voxel_size * 1.5
        self.transformation_history = []
        self.initial_alignment_done = False
        self.current_transformation = np.identity(4)
    
    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud for registration"""
        # Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        
        # Compute FPFH features
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )
        
        return pcd_down, pcd_fpfh
    
    def detect_and_estimate_transformation(self, source_pcd, target_pcd):
        """Detect overlap and estimate transformation between two point clouds"""
        # Check if we have enough points for registration
        if len(source_pcd.points) < 100 or len(target_pcd.points) < 100:
            return None, 0.0
        
        # Preprocess point clouds
        source_down, source_fpfh = self.preprocess_point_cloud(source_pcd)
        target_down, target_fpfh = self.preprocess_point_cloud(target_pcd)
        
        # Use Global registration to find initial alignment
        if not self.initial_alignment_done:
            # RANSAC-based global registration
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=self.distance_threshold * 2,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )
            
            if result.transformation.trace() == 4.0:  # Identity matrix has trace of 4
                print("Global registration failed to find overlap")
                return None, 0.0
                
            self.current_transformation = result.transformation
            self.initial_alignment_done = True
            overlap_ratio = result.fitness
            print(f"Initial alignment found with overlap ratio: {overlap_ratio:.3f}")
        else:
            # Use ICP for fine registration once we have initial alignment
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, self.distance_threshold, self.current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            
            self.current_transformation = result.transformation
            overlap_ratio = result.fitness
            
        return self.current_transformation, overlap_ratio
    
    def align_point_clouds(self, source_pcd, target_pcd):
        """Align source point cloud to target point cloud"""
        transformation, overlap_ratio = self.detect_and_estimate_transformation(source_pcd, target_pcd)
        
        if transformation is None:
            return None, 0.0
        
        # Transform the source point cloud
        source_pcd_transformed = source_pcd.clone()
        source_pcd_transformed.transform(transformation)
        
        return source_pcd_transformed, overlap_ratio

class MultiKinectMeshReconstructor:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size
        self.mesh_converter = PointCloudToMesh(voxel_size)
        self.registration = PointCloudRegistration(voxel_size * 3)  # Larger voxel size for registration
        self.visualization_mode = "pointcloud"  # Start with point cloud visualization
        self.mesh_reconstruction = False
        self.output_folder = "results"
        self.camera_threads = []
        self.found_devices = []
        self.available_cameras = 0
        self.integrated_mesh = None
        self.integrated_pcd = None
        self.merged_pcd = None
        self.camera_colors = [
            [1, 0.7, 0.7],  # Light red for camera 0
            [0.7, 0.7, 1]   # Light blue for camera 1
        ]
        self.show_original_clouds = False  # Toggle to show unmerged clouds
        
        os.makedirs(self.output_folder, exist_ok=True)
        
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
        self.vis.create_window("Multi-Kinect Mesh Reconstruction", 1280, 720)
        self.register_callbacks()
        
        # Set up render options
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
        opt.show_coordinate_frame = False  # Hide coordinate frame
    
    def detect_cameras(self):
        """Detect available Kinect cameras"""
        # Try to open up to 4 devices (typical maximum for Azure Kinect)
        max_devices = 4
        for i in range(max_devices):
            try:
                # Try to open the device temporarily
                config = Config()
                k4a = PyK4A(config)
                
                # In the current version, we'll need to modify this to properly
                # detect multiple devices, but this approach should work for testing
                # with a single device
                
                # Just check if we can initialize the camera
                k4a.open()
                serial = k4a._device_id  # Accessing internal variable for testing
                k4a.close()
                
                print(f"Found Kinect device {i} with id {serial}")
                self.found_devices.append(i)
                
                # Only add first device for now (until we have proper multiple device support)
                if i == 0:
                    break
                    
            except Exception as e:
                # Device not available or error opening
                print(f"Failed to open device {i}: {e}")
                pass
                
        # For testing purposes, let's always add a second "virtual" device
        # Remove this in production code
        if self.found_devices:
            print("Adding simulated second camera for testing")
            if 1 not in self.found_devices:
                self.found_devices.append(1)
                
        self.available_cameras = len(self.found_devices)
        print(f"Found {self.available_cameras} Kinect camera(s)")
        
        return self.available_cameras > 0
    
    def setup_camera_threads(self):
        """Set up camera capture threads"""
        for device_id in self.found_devices[:2]:  # Limit to 2 cameras for now
            thread = KinectCaptureThread(device_id=device_id)
            self.camera_threads.append(thread)
    
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
            print(f"Visualization mode: {self.visualization_mode}")
            return True
        
        def toggle_original_clouds(vis):
            self.show_original_clouds = not self.show_original_clouds
            print(f"Show original clouds: {'enabled' if self.show_original_clouds else 'disabled'}")
            return True
            
        def save_current_state(vis):
            # Save point cloud
            if self.merged_pcd is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pcd_path = os.path.join(self.output_folder, f"kinect_reconstruction_{timestamp}.ply")
                o3d.io.write_point_cloud(pcd_path, self.merged_pcd)
                print(f"Point cloud saved to {pcd_path}")
            
            # Save mesh if available
            if self.integrated_mesh is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mesh_path = os.path.join(self.output_folder, f"kinect_reconstruction_{timestamp}.obj")
                o3d.io.write_triangle_mesh(mesh_path, self.integrated_mesh)
                print(f"Mesh saved to {mesh_path}")
            
            return True
        
        # Register key callbacks
        self.vis.register_key_callback(ord("M"), toggle_mesh_reconstruction)   # M to toggle mesh reconstruction
        self.vis.register_key_callback(ord("V"), toggle_visualization_mode)    # V to toggle visualization mode
        self.vis.register_key_callback(ord("O"), toggle_original_clouds)       # O to toggle original cloud view
        self.vis.register_key_callback(ord("S"), save_current_state)           # S to save current state
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
        
        return pcd, rgbd
    
    def update_visualization(self, point_clouds):
        """Update the visualization with the current point cloud or mesh"""
        self.vis.clear_geometries()
        
        # First, try to align the point clouds if we have more than one
        if len(point_clouds) >= 2:
            # Align second camera to first camera
            aligned_pcd, overlap_ratio = self.registration.align_point_clouds(point_clouds[1], point_clouds[0])
            
            if aligned_pcd is not None and overlap_ratio > 0.3:  # We have good overlap
                # Prepare colored point clouds for visualization if needed
                if self.show_original_clouds:
                    # Color the original clouds for visualization
                    for i, pcd in enumerate(point_clouds):
                        temp_pcd = pcd.clone()
                        temp_pcd.paint_uniform_color(self.camera_colors[i])
                        self.vis.add_geometry(temp_pcd)
                
                # Merge point clouds (cloud 0 and aligned cloud 1)
                merged_pcd = point_clouds[0] + aligned_pcd
                # Remove duplicate points with voxel downsampling
                merged_pcd = merged_pcd.voxel_down_sample(self.voxel_size)
                self.merged_pcd = merged_pcd
                
                if self.visualization_mode == "mesh" and self.mesh_reconstruction:
                    try:
                        # Preprocess point cloud
                        processed_pcd = self.mesh_converter.preprocess_point_cloud(merged_pcd)
                        
                        # Try Poisson reconstruction first
                        try:
                            mesh = self.mesh_converter.create_mesh_poisson(processed_pcd)
                            self.integrated_mesh = mesh
                            self.vis.add_geometry(mesh)
                        except Exception as e:
                            print(f"Poisson reconstruction failed: {e}. Trying Ball Pivoting...")
                            radii = [0.005, 0.01, 0.02, 0.04]
                            mesh = self.mesh_converter.create_mesh_ball_pivoting(processed_pcd, radii)
                            self.integrated_mesh = mesh
                            self.vis.add_geometry(mesh)
                    except Exception as e:
                        print(f"Mesh reconstruction failed: {e}")
                        self.vis.add_geometry(merged_pcd)
                else:
                    self.vis.add_geometry(merged_pcd)
            else:
                # Not enough overlap, show point clouds separately
                print("Not enough overlap between cameras or alignment failed")
                for i, pcd in enumerate(point_clouds):
                    # Color each point cloud
                    temp_pcd = pcd.clone()
                    temp_pcd.paint_uniform_color(self.camera_colors[i])
                    self.vis.add_geometry(temp_pcd)
        else:
            # Only one camera available
            for pcd in point_clouds:
                self.vis.add_geometry(pcd)
                self.merged_pcd = pcd
        
        # Update renderer
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        """Main loop for Kinect streaming and reconstruction"""
        global running
        
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        # Detect available cameras
        if not self.detect_cameras():
            print("No Kinect cameras detected. Exiting.")
            return
            
        if self.available_cameras < 2:
            print("Warning: Only one camera detected. Multi-camera reconstruction requires at least two cameras.")
            print("Continuing with single camera mode...")
        
        # Setup camera threads (limit to 2 cameras for now)
        self.setup_camera_threads()
        
        # Start camera threads
        for thread in self.camera_threads:
            thread.start()
        
        # Wait a moment for cameras to initialize
        time.sleep(2)
        
        print("\nControls:")
        print("  M - Toggle mesh reconstruction")
        print("  V - Toggle visualization mode (point cloud/mesh)")
        print("  O - Toggle original point clouds view")
        print("  S - Save current reconstruction")
        print("  R - Reset camera view")
        print("  Ctrl+C - Stop streaming")
        
        try:
            while running:
                point_clouds = []
                
                # Get frames from all cameras
                for i, thread in enumerate(self.camera_threads):
                    frame = thread.get_latest_frame()
                    if frame is not None:
                        color_img, depth_img = frame
                        pcd, _ = self.process_frame(color_img, depth_img)
                        point_clouds.append(pcd)
                
                if point_clouds:
                    # Update visualization
                    self.update_visualization(point_clouds)
                
                # Don't max out CPU
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error during streaming: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Wait for threads to finish
            running = False
            for thread in self.camera_threads:
                thread.join(timeout=1.0)
                
            self.vis.destroy_window()
            print("All camera streams stopped")

def main():
    reconstructor = MultiKinectMeshReconstructor(voxel_size=0.01)
    reconstructor.run()

if __name__ == "__main__":
    main()