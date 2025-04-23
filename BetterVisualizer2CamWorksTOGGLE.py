import numpy as np
import copy
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


'''Need to add color to it still, it's for later'''
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
            
            # Try to initialize with device ID in constructor
            try:
                self.k4a = PyK4A(config, device_id=self.device_id)
                self.k4a.start()
            except Exception as e:
                print(f"Failed to initialize with device_id in constructor: {e}")
                # Try alternate method
                self.k4a = PyK4A(config)
                # Some versions use different methods to select device
                try:
                    self.k4a.open(device_id=self.device_id)
                    self.k4a.start()
                except AttributeError:
                    # For older versions that might not have open() with device_id
                    self.k4a.open()
                    self.k4a.start()
            
            # Get calibration information
            self.calibration = self.k4a.calibration
            
            # Create a default intrinsic with Azure Kinect DK color camera parameters (focal length ~2.3mm)
            # These are approximate values that work for most Azure Kinect cameras at 720p
            self.color_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=1280,   # 720p width
                height=720,   # 720p height
                fx=605.6,     # typical focal length for Azure Kinect
                fy=605.9,     # slightly different in y-direction
                cx=637.7,     # principal point x
                cy=364.3      # principal point y
            )
            
            # Approximate depth camera values (focal length ~1.8mm)
            self.depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640,    # NFOV_UNBINNED width
                height=576,   # NFOV_UNBINNED height
                fx=504.2,     # typical focal length for depth
                fy=504.1,
                cx=319.8,
                cy=287.9
            )
            
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
                
                # Convert BGR to RGB as Open3D uses RGB
                color_image = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue, non-blocking
                if not self.frame_queue.full():
                    self.frame_queue.put((color_image, capture.transformed_depth, self.color_intrinsic), block=False)
                
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
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
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
            print("Not enough points for registration")
            return None, 0.0
        
        try:
            # Preprocess point clouds
            source_down, source_fpfh = self.preprocess_point_cloud(source_pcd)
            target_down, target_fpfh = self.preprocess_point_cloud(target_pcd)
            
            # Use Global registration to find initial alignment
            if not self.initial_alignment_done:
                # RANSAC-based global registration
                result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_down, target_down, source_fpfh, target_fpfh,
                    max_correspondence_distance=self.distance_threshold * 2,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=4,
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
            
        except Exception as e:
            print(f"Error during registration: {e}")
            return None, 0.0
    
    def align_point_clouds(self, source_pcd, target_pcd):
        """Align source point cloud to target point cloud"""
        transformation, overlap_ratio = self.detect_and_estimate_transformation(source_pcd, target_pcd)
        
        if transformation is None:
            return None, 0.0
        
        # Transform the source point cloud
        source_pcd_transformed = copy.deepcopy(source_pcd)
        source_pcd_transformed.transform(transformation)
        
        return source_pcd_transformed, overlap_ratio

class TSDFIntegration:
    """Class for TSDF volume integration"""
    def __init__(self, voxel_length=0.01, sdf_trunc=0.04, volume_bounds=None):
        if volume_bounds is None:
            # Default volume bounds (meters): ±2m in each direction
            volume_bounds = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
            
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.volume_bounds = volume_bounds
        
        # Initialize the TSDF volume
        self.reset_volume()
    
    def reset_volume(self):
        """Reset the TSDF volume"""
        # Create TSDF volume with color support
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    
    def integrate_frame(self, color_img, depth_img, intrinsic, extrinsic=np.eye(4)):
        """Integrate a frame into the TSDF volume"""
        # Create RGBD image with proper color handling
        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,  # Azure Kinect depth is in mm
            depth_trunc=3.0,     # 3 meters max depth
            convert_rgb_to_intensity=False  # Preserve RGB color
        )
        
        # Integrate with color information
        self.volume.integrate(rgbd, intrinsic, extrinsic)
    
    def extract_mesh(self):
        """Extract mesh from the TSDF volume"""
        mesh = self.volume.extract_triangle_mesh()
        if mesh is not None:
            mesh.compute_vertex_normals()
        return mesh
    
    def extract_point_cloud(self):
        """Extract point cloud from the TSDF volume"""
        pcd = self.volume.extract_point_cloud()
        if pcd is not None:
            pcd.estimate_normals()
        return pcd

class MultiKinectMeshReconstructor:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size
        self.mesh_converter = PointCloudToMesh(voxel_size)
        self.registration = PointCloudRegistration(voxel_size * 3)  # Larger voxel size for registration
        self.tsdf_integrator = TSDFIntegration(voxel_length=voxel_size, sdf_trunc=voxel_size*4)
        self.visualization_mode = "pointcloud"  # Start with point cloud visualization
        self.mesh_reconstruction = False
        self.use_tsdf = True  # TSDF integration enabled by default
        self.output_folder = "results"
        self.camera_threads = []
        self.found_devices = []
        self.available_cameras = 0
        self.integrated_mesh = None
        self.integrated_pcd = None
        self.merged_pcd = None
        self.camera_colors = [
            [0.7, 0.0, 0.0],  # Light Red for camera 0
            [0.0, 0.0, 0.7]   # Light Blue for camera 1
        ]
        self.show_original_clouds = False  # Toggle to show unmerged clouds
        self.extrinsic_calibration_done = False
        self.camera_extrinsics = []  # Store extrinsic matrices for each camera
        
        # New color mode states
        self.use_colors = True  # Enable color by default
        self.color_mode = "original"  # Options: "original", "uniform", "camera"
        
        os.makedirs(self.output_folder, exist_ok=True)
        
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
        opt.show_coordinate_frame = True  # Show coordinate frame
    
    # In the detect_cameras method, make sure we have better error handling
    def detect_cameras(self):
        """Detect available Kinect cameras"""
        self.found_devices = []
        
        # First, try to get the number of connected devices
        try:
            # Try to detect up to 2 cameras
            for device_id in range(2):
                try:
                    # In some versions of PyK4A, you need to specify device_id during initialization
                    config = Config(
                        color_resolution=ColorResolution.RES_720P,
                        depth_mode=DepthMode.NFOV_UNBINNED,
                        synchronized_images_only=True,
                    )
                    
                    # Try with device_id in constructor
                    try:
                        k4a = PyK4A(config, device_id=device_id)
                        k4a.open()
                        k4a.close()
                        self.found_devices.append(device_id)
                        print(f"Found camera {device_id}")
                    except Exception as e:
                        print(f"Error opening camera {device_id} with device_id in constructor: {e}")
                        
                        # Try alternate method (some PyK4A versions)
                        try:
                            k4a = PyK4A(config)
                            # Some versions use k4a.open(device_id)
                            k4a.open(device_id=device_id)
                            k4a.close()
                            self.found_devices.append(device_id)
                            print(f"Found camera {device_id} with alternate method")
                        except Exception as e2:
                            print(f"Error opening camera {device_id} with alternate method: {e2}")
                            
                except Exception as e:
                    print(f"Failed to initialize camera {device_id}: {e}")
            
            # If no cameras were found, print a warning
            if not self.found_devices:
                print("No Kinect cameras detected!")
                
            # For debugging purposes only - uncomment to simulate a second camera
            # if len(self.found_devices) == 1 and 1 not in self.found_devices:
            #     self.found_devices.append(1)
            #     print("Adding simulated second camera for testing")
                
        except Exception as e:
            print(f"Error in camera detection: {e}")
                
        self.available_cameras = len(self.found_devices)
        print(f"Using {self.available_cameras} Kinect camera(s): {self.found_devices}")
        
        return self.available_cameras > 0
    
    def setup_camera_threads(self):
        """Set up camera capture threads"""
        for device_id in self.found_devices[:2]:  # Limit to 2 cameras for now
            thread = KinectCaptureThread(device_id=device_id)
            self.camera_threads.append(thread)
            # Initialize extrinsic matrix (identity for camera 0, unknown for others)
            if device_id == 0:
                self.camera_extrinsics.append(np.eye(4))  # First camera is our reference frame
            else:
                self.camera_extrinsics.append(None)  # Will be calibrated later
    
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
        def cycle_color_mode(vis):
            modes = ["original", "uniform", "camera"]
            current_idx = modes.index(self.color_mode)
            self.color_mode = modes[(current_idx + 1) % len(modes)]
            print(f"Color mode: {self.color_mode}")
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
        
        def recalibrate_cameras(vis):
            self.extrinsic_calibration_done = False
            self.registration.initial_alignment_done = False
            print("Camera calibration reset. Will recalibrate on next frame.")
            return True
        
        # Register key callbacks
        self.vis.register_key_callback(ord("C"), cycle_color_mode)             # C to cycle color modes
        self.vis.register_key_callback(ord("S"), save_current_state)           # S to save current state
        self.vis.register_key_callback(ord("R"), recalibrate_cameras)          # R to recalibrate cameras
    
    def process_frame(self, color_img, depth_img, intrinsic):
        """Process a single frame from the Kinect"""
        # Create RGBD image
        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,  # Azure Kinect depth is in mm
            depth_trunc=3.0      # 3 meters max depth
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Apply flip transform to match expected coordinate system
        pcd.transform(self.flip_transform)
        
        # Estimate normals for better visualization
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        
        return pcd, rgbd
    
    def calibrate_cameras(self, point_clouds):
        """Perform automatic extrinsic calibration between cameras"""
        if len(point_clouds) < 2 or self.extrinsic_calibration_done:
            return True
            
        # We use the first camera as our reference frame
        source_pcd = point_clouds[1]  # Second camera's point cloud
        target_pcd = point_clouds[0]  # First camera's point cloud
        
        # Preprocess point clouds to focus on floor and major structures
        def preprocess_for_calibration(pcd):
            # Remove statistical outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Downsample
            pcd = pcd.voxel_down_sample(self.voxel_size)
            
            # Estimate normals
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
            )
            
            return pcd
        
        # Preprocess both point clouds
        source_pcd = preprocess_for_calibration(source_pcd)
        target_pcd = preprocess_for_calibration(target_pcd)
        
        # Perform point cloud alignment with increased iterations for better accuracy
        aligned_pcd, overlap_ratio = self.registration.align_point_clouds(source_pcd, target_pcd)
        
        if aligned_pcd is not None and overlap_ratio > 0.2:  # Lower threshold to allow for partial overlap
            # Store the transformation as camera 1's extrinsic (relative to camera 0)
            self.camera_extrinsics[1] = self.registration.current_transformation
            
            # Print detailed transformation information
            print(f"Extrinsic calibration successful. Overlap ratio: {overlap_ratio:.3f}")
            print(f"Extrinsic matrix:\n{self.camera_extrinsics[1]}")
            
            # Extract and print translation components
            translation = self.camera_extrinsics[1][:3, 3]
            print(f"Translation (x, y, z): {translation}")
            
            # Extract and print rotation angles
            rotation_matrix = self.camera_extrinsics[1][:3, :3]
            angles = np.degrees(np.array([
                np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
                np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)),
                np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            ]))
            print(f"Rotation angles (roll, pitch, yaw): {angles}")
            
            self.extrinsic_calibration_done = True
            return True
        else:
            print("Calibration failed: Not enough overlap between cameras")
            return False
    
    def display_camera_info(self, camera_id):
        """Display camera information in the console"""
        if camera_id == 0:
            print(f"Camera {camera_id} (Reference)")
        else:
            if self.camera_extrinsics[camera_id] is not None:
                # Extract translation and rotation
                translation = self.camera_extrinsics[camera_id][:3, 3]
                rotation_matrix = self.camera_extrinsics[camera_id][:3, :3]
                angles = np.degrees(np.array([
                    np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
                    np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)),
                    np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                ]))
                print(f"Camera {camera_id} - Position: {translation}, Rotation: {angles}")
            else:
                print(f"Camera {camera_id} (Not calibrated)")

    def update_visualization(self, frames):
        """Update the visualization with the current point cloud or mesh"""
        self.vis.clear_geometries()
        
        # Process frames to get point clouds
        point_clouds = []
        intrinsics = []
        rgbds = []
        
        for i, frame in enumerate(frames):
            if frame is not None:
                color_img, depth_img, color_intrinsic = frame
                pcd, rgbd = self.process_frame(color_img, depth_img, color_intrinsic)
                point_clouds.append(pcd)
                intrinsics.append(color_intrinsic)
                rgbds.append((rgbd, color_img, depth_img))
        
        if not point_clouds:
            return
            
        # Calibrate cameras if not already done
        if not self.extrinsic_calibration_done and len(point_clouds) >= 2:
            self.calibrate_cameras(point_clouds)
        
        # Show individual point clouds with different colors if requested
        if self.show_original_clouds:
            for i, pcd in enumerate(point_clouds):
                temp_pcd = copy.deepcopy(pcd)
                if self.color_mode == "uniform":
                    temp_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
                elif self.color_mode == "camera":
                    temp_pcd.paint_uniform_color(self.camera_colors[i])
                # "original" mode keeps original colors
                
                if i > 0 and self.extrinsic_calibration_done and self.camera_extrinsics[i] is not None:
                    temp_pcd.transform(self.camera_extrinsics[i])
                self.vis.add_geometry(temp_pcd)
        
        # If we have calibration, show the merged point cloud
        if self.extrinsic_calibration_done and len(point_clouds) >= 2:
            # Create separate point clouds for each camera with their colors
            camera_clouds = []
            for i, pcd in enumerate(point_clouds):
                temp_pcd = copy.deepcopy(pcd)
                if i > 0 and self.camera_extrinsics[i] is not None:
                    temp_pcd.transform(self.camera_extrinsics[i])
                if self.color_mode == "camera":
                    temp_pcd.paint_uniform_color(self.camera_colors[i])
                camera_clouds.append(temp_pcd)
            
            # Merge point clouds
            merged_pcd = camera_clouds[0]
            for i in range(1, len(camera_clouds)):
                merged_pcd += camera_clouds[i]
            
            # Downsample the merged point cloud while preserving colors
            merged_pcd = merged_pcd.voxel_down_sample(self.voxel_size)
            self.merged_pcd = merged_pcd
            
            # Apply color mode to merged point cloud
            if self.color_mode == "uniform":
                merged_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
            # "original" and "camera" modes keep their colors from above
            
            self.vis.add_geometry(merged_pcd)
        
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
        
        # Setup camera threads
        self.setup_camera_threads()
        
        # Start camera threads
        for thread in self.camera_threads:
            thread.start()
        
        # Wait a moment for cameras to initialize
        time.sleep(2)
        
        print("\nControls:")
        print("  C - Cycle color modes (original/uniform/camera)")
        print("  S - Save current reconstruction")
        print("  R - Recalibrate cameras")
        print("  Ctrl+C - Stop streaming")
        
        try:
            while running:
                frames = []
                
                # Get frames from all cameras
                for i, thread in enumerate(self.camera_threads):
                    frame = thread.get_latest_frame()
                    frames.append(frame)
                
                if any(frame is not None for frame in frames):
                    # Update visualization
                    self.update_visualization(frames)
                
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