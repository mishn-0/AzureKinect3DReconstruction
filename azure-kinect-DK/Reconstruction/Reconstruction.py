import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time
import copy
import os

class KinectReconstructor:
    def __init__(self):
        # Configure Kinect
        self.k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()
        
        # Camera intrinsics
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            1280, 720,  # Width, height
            605.286, 605.699,  # fx, fy
            637.134, 366.758  # cx, cy
        )
        
        # Transformation to align with correct axis (floor down)
        self.flip_transform = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        
        # Reconstruction parameters
        self.voxel_size = 0.01  # 1cm voxel size
        self.distance_threshold = 0.05  # 5cm threshold for ICP
        
        # Storage for reconstruction
        self.global_model = o3d.geometry.PointCloud()
        self.previous_pcd = None
        self.current_transformation = np.identity(4)
        
        # For tracking frames
        self.frame_count = 0
        self.keyframe_interval = 5  # Process every 5th frame for registration
        
        # For storing keyframes
        self.keyframes = []
        
        # Visualization
        self.vis = None
        self.is_recording = False
        self.output_folder = "reconstruction_output"
        os.makedirs(self.output_folder, exist_ok=True)
    
    def process_images(self, color_img, depth_img):
        # Flip vertical and horizontal (if needed for your setup)
        color_img = cv2.flip(color_img, -1)
        depth_img = cv2.flip(depth_img, -1)
        
        # Convert to Open3D format
        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,  # Azure Kinect depth is in mm
            depth_trunc=3.0,  # 3m truncation
        )
        return rgbd
    
    def preprocess_point_cloud(self, pcd):
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        # Compute normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        return pcd
    
    def register_frames(self, source, target):
        """Register source point cloud to target using point-to-plane ICP"""
        
        # Initial alignment using feature matching
        source_down = source.voxel_down_sample(self.voxel_size * 2)
        target_down = target.voxel_down_sample(self.voxel_size * 2)
        
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )
        
        # Fast global registration for coarse alignment
        result_fast = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.distance_threshold * 2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold * 2)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        # Point-to-plane ICP for fine alignment
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, self.distance_threshold, result_fast.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        return result_icp.transformation
    
    def add_frame_to_model(self, new_pcd):
        # First frame becomes the initial model
        if len(self.global_model.points) == 0:
            self.global_model = copy.deepcopy(new_pcd)
            self.previous_pcd = copy.deepcopy(new_pcd)
            return True
        
        # Register new frame to previous frame
        try:
            transformation = self.register_frames(new_pcd, self.previous_pcd)
            
            # Apply transformation to align with global model
            self.current_transformation = np.matmul(self.current_transformation, transformation)
            aligned_pcd = copy.deepcopy(new_pcd)
            aligned_pcd.transform(self.current_transformation)
            
            # Add to global model
            self.global_model += aligned_pcd
            
            # Downsample global model to manage size
            if self.frame_count % 10 == 0:
                self.global_model = self.global_model.voxel_down_sample(self.voxel_size)
            
            # Update previous frame
            self.previous_pcd = copy.deepcopy(new_pcd)
            
            return True
        except Exception as e:
            print(f"Registration failed: {e}")
            return False
    
    def start_visualization(self):
        # Create visualizer window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("3D Reconstruction from Kinect", 1280, 720)
        
        # Get first valid capture
        while True:
            capture = self.k4a.get_capture()
            if capture.color is not None and capture.transformed_depth is not None:
                break
        
        # Create initial point cloud
        color = capture.color
        depth = capture.transformed_depth
        rgbd = self.process_images(color, depth)
        initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics)
        initial_pcd.transform(self.flip_transform)
        initial_pcd = self.preprocess_point_cloud(initial_pcd)
        
        # Add to visualizer
        self.vis.add_geometry(initial_pcd)
        
        # Set initial view
        self.setup_camera_view()
        
        # Register keyboard callbacks
        self.register_callbacks()
        
        # Main visualization loop
        try:
            self.run_visualization_loop(initial_pcd)
        finally:
            self.cleanup()
    
    def setup_camera_view(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        ctr = self.vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(0.5)
    
    def register_callbacks(self):
        def toggle_recording(vis):
            self.is_recording = not self.is_recording
            print(f"Recording {'started' if self.is_recording else 'stopped'}")
            return True
        
        def save_model(vis):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.output_folder}/reconstruction_{timestamp}"
            
            # Save as point cloud
            o3d.io.write_point_cloud(f"{filename}.ply", self.global_model)
            
            # Create and save mesh using Poisson reconstruction
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.global_model, depth=9, width=0, scale=1.1, linear_fit=False
            )[0]
            o3d.io.write_triangle_mesh(f"{filename}.obj", mesh)
            
            print(f"Model saved to {filename}.ply and {filename}.obj")
            return True
        
        def reset_model(vis):
            self.global_model = o3d.geometry.PointCloud()
            self.current_transformation = np.identity(4)
            self.frame_count = 0
            print("Model reset")
            return True
        
        # Register key callbacks
        self.vis.register_key_callback(ord("R"), toggle_recording)
        self.vis.register_key_callback(ord("S"), save_model)
        self.vis.register_key_callback(ord("C"), reset_model)
    
    def run_visualization_loop(self, pcd):
        print("\nControls:")
        print("  R: Toggle recording (add frames to the reconstruction)")
        print("  S: Save the current reconstruction model")
        print("  C: Clear/reset the reconstruction")
        print("  Esc: Exit the program")
        print("In Open3D, keypresses are often captured only when the visualization window is in focus, so you need to press 'Shift + R'")
        print("(or just 'R', depending on how the key events are set up) while the window is active for the callback to trigger.")


        last_time = time.time()
        fps_count = 0
        fps = 0
        
        while True:
            # FPS calculation
            fps_count += 1
            if time.time() - last_time > 1.0:
                fps = fps_count
                fps_count = 0
                last_time = time.time()
                print(f"FPS: {fps}, Recording: {'ON' if self.is_recording else 'OFF'}, Frames: {self.frame_count}")
            
            # Get new frame
            capture = self.k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue
            
            # Process frame
            color = capture.color
            depth = capture.transformed_depth
            rgbd = self.process_images(color, depth)
            
            # Create and preprocess point cloud
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics)
            new_pcd.transform(self.flip_transform)
            new_pcd = self.preprocess_point_cloud(new_pcd)
            
            # Update visualization pcd
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            pcd.normals = new_pcd.normals
            
            # Add to reconstruction model if recording
            if self.is_recording and self.frame_count % self.keyframe_interval == 2:  # Process every 5th frame
                success = self.add_frame_to_model(copy.deepcopy(new_pcd))
                if success:
                    print(f"Added frame {self.frame_count} to model, total points: {len(self.global_model.points)}")
            
            self.frame_count += 1
            
            # Update visualizer
            self.vis.update_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def cleanup(self):
        self.k4a.stop()
        print("Kinect stopped and program finished.")

if __name__ == "__main__":
    reconstructor = KinectReconstructor()
    reconstructor.start_visualization()
