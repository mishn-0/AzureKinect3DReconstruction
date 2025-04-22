import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time
import os
import copy

class KinectReconstructor:
    def __init__(self):
        # Check for CUDA availability
        self.cuda_available = False
        try:
            import cupy as cp
            self.cuda_available = cp.cuda.runtime.getDeviceCount() > 0
            self.cp = cp
            if self.cuda_available:
                print(f"[INFO] CUDA acceleration enabled - Device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
            else:
                print("[INFO] No CUDA device found, falling back to CPU")
        except ImportError:
            print("[INFO] CuPy not installed, running in CPU-only mode")
        except Exception as e:
            print(f"[WARNING] CUDA initialization failed: {e}")
        
        self.k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        self.k4a.start()

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=1280, height=720,
            fx=605.286, fy=605.699,
            cx=637.134, cy=366.758
        )

        # Store flip transform in NumPy
        # This matches the coordinate systems between Open3D (right-handed coordinates with +Z going away from the camera) and Kinect
        self.flip_transform = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

        # IMPORTANT: Parameters for reconstruction
        self.voxel_size = 0.01  # Downsampling size
        self.sdf_trunc = 0.04  # Truncation value for signed distance function, TSDF distance truncation for smoothing
        self.block_resolution = 16  # Resolution of the TSDF volume block
        self.block_count = 1000  # Initial number of blocks     #Used for GPU-based TSDF
        
        # Registration parameters
        self.distance_threshold = 0.05  # For initial registration and alignment
        self.icp_distance_threshold = 0.03  # For ICP refinement
        self.keyframe_interval = 10  # Process every nth frame

        # Setup visualization and model
        self.frame_count = 0        # Track how many frames we have processed so far, initialized at 0, first frame triggers events
        self.is_recording = False       #track capture progress
        self.output_folder = "reconstruction_output"
        os.makedirs(self.output_folder, exist_ok=True)
        self.vis = None     # Open3D visualization window
        
        # State for tracking
        self.volume = None
        self.trajectory = []
        self.current_transformation = np.identity(4)
        self.mesh_reconstruction = True  # Whether to use TSDF volume for mesh reconstruction
        
        # Create initial point cloud for visualization
        self.vis_pcd = o3d.geometry.PointCloud()
        
        # For frame-to-frame tracking
        self.prev_rgbd = None
        self.prev_color = None
        self.prev_depth = None
        
        # For visualizing the integrated model
        self.integrated_mesh = None
        self.integrated_pcd = None
        self.show_integrated_model = True  # Start by showing the integrated model by default
        self.vis_update_interval = 5  # Update visualization every n frames (reduced for better feedback)
        self.visualization_mode = "pointcloud"  # "pointcloud" or "mesh"

        # Check Open3D CUDA
        try:
            self.o3d_cuda_available = (hasattr(o3d, 'core') and 
                                      hasattr(o3d.core, 'cuda') and 
                                      o3d.core.cuda.is_available())
            if self.o3d_cuda_available:
                print("[INFO] Open3D CUDA acceleration enabled")
                self.device = o3d.core.Device("CUDA:0")
            else:
                print("[INFO] Open3D CUDA not available")
                self.device = o3d.core.Device("CPU:0")
        except Exception as e:
            print(f"[WARNING] Open3D CUDA check failed: {e}")
            self.o3d_cuda_available = False
            self.device = None

    def initialize_volume(self):
        """Initialize TSDF volume for reconstruction"""
        if not self.mesh_reconstruction:
            return
            
        if hasattr(o3d, 'pipelines') and hasattr(o3d.pipelines, 'integration'):
            try:
                # Legacy ScalableTSDFVolume
                self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=self.voxel_size,
                    sdf_trunc=self.sdf_trunc,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )
                print("[INFO] Using legacy ScalableTSDFVolume")
            except Exception as e:
                print(f"[WARNING] Failed to initialize ScalableTSDFVolume: {e}")
                self.mesh_reconstruction = False
        else:
            print("[WARNING] o3d.pipelines.integration not available")
            self.mesh_reconstruction = False

    def process_images(self, color_img, depth_img):
        """Process color and depth images"""
        # For stable color processing, use CPU
        color_img = cv2.cvtColor(cv2.flip(color_img, -1), cv2.COLOR_BGR2RGB)
        depth_img = cv2.flip(depth_img, -1)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_img),
            o3d.geometry.Image(depth_img),
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,
            depth_trunc=3.0,
        )
        
        return rgbd, color_img, depth_img

    def preprocess_point_cloud(self, pcd):
        """Process point cloud for registration"""
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        return pcd

    def compute_features(self, pcd):
        """Compute FPFH features for registration"""
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )

    def global_registration(self, source, target):
        """Perform global registration with RANSAC"""
        source_down = source.voxel_down_sample(self.voxel_size * 2)
        target_down = target.voxel_down_sample(self.voxel_size * 2)
        
        source_fpfh = self.compute_features(source_down)
        target_fpfh = self.compute_features(target_down)
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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
        
        if result.fitness < 0.3:
            print(f"[WARNING] Low global registration fitness: {result.fitness:.3f}")
            return None
            
        return result.transformation

    def icp_registration(self, source, target, init_transform=None):
        """Perform ICP registration"""
        if init_transform is None:
            init_transform = np.identity(4)
            
        result = o3d.pipelines.registration.registration_icp(
            source, target, self.icp_distance_threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        if result.fitness < 0.5:
            print(f"[WARNING] Low ICP fitness: {result.fitness:.3f}")
            return None
            
        return result.transformation

    def register_frames(self, source_pcd, target_pcd):
        """Register source to target point cloud"""
        # First try standard registration pipeline
        global_transform = self.global_registration(source_pcd, target_pcd)
        if global_transform is None:
            return None
            
        # Refine with ICP
        transform = self.icp_registration(source_pcd, target_pcd, global_transform)
        if transform is None:
            return None
            
        return transform

    def register_rgbd(self, source_rgbd, target_rgbd, source_pcd=None, target_pcd=None):
        """Register two RGBD frames using colored ICP"""
        if source_pcd is None:
            source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                source_rgbd, self.intrinsics)
            source_pcd.transform(self.flip_transform)
            source_pcd = self.preprocess_point_cloud(source_pcd)
            
        if target_pcd is None:
            target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                target_rgbd, self.intrinsics)
            target_pcd.transform(self.flip_transform)
            target_pcd = self.preprocess_point_cloud(target_pcd)
        
        # Try colored ICP first
        try:
            # Initial motion estimate
            init_transform = np.identity(4)
            if len(self.trajectory) >= 2:
                # Estimate motion from previous frames
                prev_motion = np.matmul(
                    np.linalg.inv(self.trajectory[-2]), 
                    self.trajectory[-1]
                )
                init_transform = prev_motion
                
            result = o3d.pipelines.registration.registration_colored_icp(
                source_pcd, target_pcd, 
                self.voxel_size * 1.5,
                init_transform,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=100)
            )
            
            if result.fitness > 0.6:
                return result.transformation
        except Exception as e:
            print(f"[WARNING] Colored ICP failed: {e}")
        
        # Fallback to standard registration
        return self.register_frames(source_pcd, target_pcd)

    def add_rgbd_to_volume(self, rgbd, transform):
        """Add an RGBD frame to the TSDF volume"""
        if not self.mesh_reconstruction or self.volume is None:
            return
            
        try:
            self.volume.integrate(rgbd, self.intrinsics, np.linalg.inv(transform))
        except Exception as e:
            print(f"[WARNING] Failed to integrate frame into volume: {e}")

    def update_visualization_model(self):
        """Update the visualization model from the TSDF volume"""
        if not self.mesh_reconstruction or self.volume is None:
            return False
        
        try:
            print("[INFO] Updating visualization model...")
            
            # Extract the current model from the TSDF volume
            if self.visualization_mode == "mesh":
                # Extract mesh for visualization
                new_mesh = self.volume.extract_triangle_mesh()
                new_mesh.compute_vertex_normals()
                
                # Check if we already have a mesh in the visualizer
                if self.integrated_mesh is not None and self.integrated_mesh in self.vis.get_geometry_list():
                    # Remove old mesh
                    self.vis.remove_geometry(self.integrated_mesh, False)
                
                # If point cloud exists, remove it
                if self.integrated_pcd is not None and self.integrated_pcd in self.vis.get_geometry_list():
                    self.vis.remove_geometry(self.integrated_pcd, False)
                
                # Update mesh reference
                self.integrated_mesh = new_mesh
                
                # Add new mesh to visualizer
                self.vis.add_geometry(self.integrated_mesh)
                
            else:  # pointcloud mode
                # Extract point cloud for visualization
                new_pcd = self.volume.extract_point_cloud()
                
                # Check if we already have a point cloud in the visualizer
                if self.integrated_pcd is not None and self.integrated_pcd in self.vis.get_geometry_list():
                    # Remove old point cloud
                    self.vis.remove_geometry(self.integrated_pcd, False)
                
                # If mesh exists, remove it
                if self.integrated_mesh is not None and self.integrated_mesh in self.vis.get_geometry_list():
                    self.vis.remove_geometry(self.integrated_mesh, False)
                
                # Update point cloud reference
                self.integrated_pcd = new_pcd
                
                # Add new point cloud to visualizer
                self.vis.add_geometry(self.integrated_pcd)
            
            # Reset view to see the whole scene
            self.setup_camera_view()
            
            print("[INFO] Visualization model updated.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to update visualization model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_visualization(self):
        """Start the visualization and reconstruction"""
        # Initialize visualization window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("3D Reconstruction from Kinect", 1280, 720)
        self.register_callbacks()
        
        # Initialize TSDF volume for mesh reconstruction
        self.initialize_volume()
        
        # Get first frame
        print("[INFO] Waiting for first frame...")
        while True:
            capture = self.k4a.get_capture()
            if capture.color is not None and capture.transformed_depth is not None:
                break
        
        # Process first frame
        first_rgbd, first_color, first_depth = self.process_images(
            capture.color, capture.transformed_depth)
        
        # Create initial point cloud
        initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            first_rgbd, self.intrinsics)
        initial_pcd.transform(self.flip_transform)
        
        # Save initial visualization point cloud
        self.vis_pcd = o3d.geometry.PointCloud()
        self.vis_pcd.points = initial_pcd.points
        self.vis_pcd.colors = initial_pcd.colors
        
        # Add to visualization based on mode
        if not self.show_integrated_model:
            self.vis.add_geometry(self.vis_pcd)
            print("[INFO] Added initial frame to visualization")
        
        # Store first frame
        self.prev_rgbd = first_rgbd
        self.prev_color = first_color
        self.prev_depth = first_depth
        self.trajectory.append(np.identity(4))
        
        # Add first frame to TSDF volume
        if self.mesh_reconstruction and self.volume is not None:
            self.add_rgbd_to_volume(first_rgbd, np.identity(4))
            if self.show_integrated_model:
                self.update_visualization_model()
                print("[INFO] Added initial frame to TSDF volume")
        
        # Set up camera view
        self.setup_camera_view()
        
        # Start main loop
        try:
            self.run_visualization_loop()
        finally:
            self.cleanup()

    def setup_camera_view(self):
        """Set up the initial camera view"""
        self.vis.poll_events()
        self.vis.update_renderer()
        ctr = self.vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(0.5)

    def register_callbacks(self):
        """Register keyboard callbacks"""
        self.vis.register_key_callback(ord("R"), self.toggle_recording)
        self.vis.register_key_callback(ord("S"), self.save_model)
        self.vis.register_key_callback(ord("C"), self.reset_model)
        self.vis.register_key_callback(ord("M"), self.toggle_mesh_reconstruction)
        self.vis.register_key_callback(ord("V"), self.toggle_visualization_mode)
        self.vis.register_key_callback(ord("I"), self.toggle_show_integrated)
        self.vis.register_key_callback(ord("U"), self.force_update_visualization)

    def toggle_recording(self, vis):
        """Toggle recording on/off"""
        self.is_recording = not self.is_recording
        print(f"[INFO] Recording {'started' if self.is_recording else 'stopped'}")
        
        # Force update visualization when starting recording
        if self.is_recording and self.show_integrated_model:
            self.update_visualization_model()
        
        return True

    def toggle_mesh_reconstruction(self, vis):
        """Toggle mesh reconstruction on/off"""
        if not self.mesh_reconstruction:
            self.mesh_reconstruction = True
            self.initialize_volume()
            print("[INFO] Mesh reconstruction enabled")
            
            # Re-add first frame to volume
            if self.prev_rgbd is not None:
                self.add_rgbd_to_volume(self.prev_rgbd, self.current_transformation)
                if self.show_integrated_model:
                    self.update_visualization_model()
        else:
            self.mesh_reconstruction = False
            print("[INFO] Mesh reconstruction disabled")
        return True
        
    def toggle_visualization_mode(self, vis):
        """Toggle between mesh and point cloud visualization"""
        if self.visualization_mode == "mesh":
            self.visualization_mode = "pointcloud"
            print("[INFO] Visualization mode: Point Cloud")
        else:
            self.visualization_mode = "mesh"
            print("[INFO] Visualization mode: Mesh")
        
        self.update_visualization_model()
        return True
    
    def force_update_visualization(self, vis):
        """Force update of visualization model"""
        print("[INFO] Forcing visualization update...")
        self.update_visualization_model()
        return True
        
    def toggle_show_integrated(self, vis):
        """Toggle showing integrated model vs current frame"""
        self.show_integrated_model = not self.show_integrated_model
        
        if self.show_integrated_model:
            # Remove current frame point cloud if it exists
            if self.vis_pcd is not None and self.vis_pcd in vis.get_geometry_list():
                vis.remove_geometry(self.vis_pcd, False)
            
            # Update integrated model
            self.update_visualization_model()
            print("[INFO] Showing integrated model")
        else:
            # Remove integrated model if it exists
            if self.integrated_pcd is not None and self.integrated_pcd in vis.get_geometry_list():
                vis.remove_geometry(self.integrated_pcd, False)
            if self.integrated_mesh is not None and self.integrated_mesh in vis.get_geometry_list():
                vis.remove_geometry(self.integrated_mesh, False)
            
            # Add current frame point cloud
            if self.vis_pcd is not None:
                vis.add_geometry(self.vis_pcd)
                vis.update_geometry(self.vis_pcd)
                print("[INFO] Showing current frame")
        
        return True

    def reset_model(self, vis):
        """Reset the reconstruction"""
        if self.mesh_reconstruction and self.volume is not None:
            self.volume = None
            self.initialize_volume()
        
        self.trajectory = [np.identity(4)]
        self.current_transformation = np.identity(4)
        self.frame_count = 0
        
        # Reset visualization models
        if self.integrated_pcd is not None:
            if self.integrated_pcd in vis.get_geometry_list():
                vis.remove_geometry(self.integrated_pcd, False)
            self.integrated_pcd = None
            
        if self.integrated_mesh is not None:
            if self.integrated_mesh in vis.get_geometry_list():
                vis.remove_geometry(self.integrated_mesh, False)
            self.integrated_mesh = None
        
        # Get current frame for reference
        capture = self.k4a.get_capture()
        if capture.color is not None and capture.transformed_depth is not None:
            self.prev_rgbd, self.prev_color, self.prev_depth = self.process_images(
                capture.color, capture.transformed_depth)
            
            # Add initial frame to visualizer
            if not self.show_integrated_model:
                initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    self.prev_rgbd, self.intrinsics)
                initial_pcd.transform(self.flip_transform)
                
                self.vis_pcd.points = initial_pcd.points
                self.vis_pcd.colors = initial_pcd.colors
                
                if self.vis_pcd not in vis.get_geometry_list():
                    vis.add_geometry(self.vis_pcd)
                else:
                    vis.update_geometry(self.vis_pcd)
        
        print("[INFO] Reconstruction reset")
        return True

    def save_model(self, vis):
        """Save the reconstructed model"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{self.output_folder}/reconstruction_{timestamp}"
        
        if self.mesh_reconstruction and self.volume is not None:
            try:
                print("[INFO] Extracting mesh from TSDF volume...")
                mesh = self.volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(f"{base}_mesh.obj", mesh)
                print(f"[INFO] Saved mesh to {base}_mesh.obj")
                
                # Also extract point cloud
                pcd = self.volume.extract_point_cloud()
                o3d.io.write_point_cloud(f"{base}_volume.ply", pcd)
                print(f"[INFO] Saved volume point cloud to {base}_volume.ply")
            except Exception as e:
                print(f"[ERROR] Failed to extract and save mesh: {e}")
        
        # Save trajectory
        if len(self.trajectory) > 1:
            try:
                with open(f"{base}_trajectory.txt", 'w') as f:
                    for i, pose in enumerate(self.trajectory):
                        f.write(f"{i}\n")
                        for j in range(4):
                            row = ' '.join([str(pose[j, k]) for k in range(4)])
                            f.write(f"{row}\n")
                print(f"[INFO] Saved trajectory to {base}_trajectory.txt")
            except Exception as e:
                print(f"[ERROR] Failed to save trajectory: {e}")
                
        # Always save the current visualization point cloud
        if self.vis_pcd is not None and len(self.vis_pcd.points) > 0:
            o3d.io.write_point_cloud(f"{base}_vis.ply", self.vis_pcd)
            print(f"[INFO] Saved visualization point cloud to {base}_vis.ply")
        
        return True

    def run_visualization_loop(self):
        """Main visualization and reconstruction loop"""
        print("\nControls:")
        print("  R: Toggle recording")
        print("  S: Save model")
        print("  C: Reset model")
        print("  M: Toggle mesh reconstruction")
        print("  V: Toggle visualization mode (mesh/point cloud)")
        print("  I: Toggle between integrated model and current frame")
        print("  U: Force update visualization")
        print("  Esc: Exit\n")

        last_time = time.time()
        fps_count = 0
        success_count = 0
        update_vis_count = 0
        
        # For measuring registration quality
        registration_times = []
        
        while True:
            # Get new frame
            capture = self.k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue

            # Process images
            curr_rgbd, curr_color, curr_depth = self.process_images(
                capture.color, capture.transformed_depth)
            
            # Create point cloud for current frame visualization
            curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                curr_rgbd, self.intrinsics)
            curr_pcd.transform(self.flip_transform)
            
            # Update visualization point cloud for current frame view
            if not self.show_integrated_model:
                self.vis_pcd.points = curr_pcd.points
                self.vis_pcd.colors = curr_pcd.colors
                if self.vis_pcd in self.vis.get_geometry_list():
                    self.vis.update_geometry(self.vis_pcd)
                else:
                    self.vis.add_geometry(self.vis_pcd)
            
            # If recording is on, process for reconstruction
            if self.is_recording:
                start_time = time.time()
                
                try:
                    # If we have a previous frame, register against it
                    if self.prev_rgbd is not None:
                        # Process only certain frames for registration (but integrate all)
                        if self.frame_count % self.keyframe_interval == 0:
                            # Preprocess point clouds
                            source_pcd = self.preprocess_point_cloud(copy.deepcopy(curr_pcd))
                            
                            # Compute transformation between frames
                            transform = self.register_rgbd(
                                curr_rgbd, self.prev_rgbd, source_pcd)
                            
                            if transform is not None:
                                # Update global transformation
                                self.current_transformation = np.matmul(
                                    self.current_transformation, transform)
                                
                                # Save to trajectory
                                self.trajectory.append(self.current_transformation.copy())
                                
                                # Success
                                success_count += 1
                                reg_time = time.time() - start_time
                                registration_times.append(reg_time)
                                
                                print(f"[+] Frame {self.frame_count} registered in {reg_time:.2f}s")
                            else:
                                print(f"[!] Failed to register frame {self.frame_count}")
                        
                        # Always add to TSDF volume with current transformation
                        if self.mesh_reconstruction and self.volume is not None:
                            self.add_rgbd_to_volume(curr_rgbd, self.current_transformation)
                            
                            # Update visualization periodically
                            update_vis_count += 1
                            if self.show_integrated_model and update_vis_count >= self.vis_update_interval:
                                self.update_visualization_model()
                                update_vis_count = 0
                                print(f"[INFO] Updated visualization at frame {self.frame_count}")
                    
                    # Store current frame as previous
                    self.prev_rgbd = curr_rgbd
                    self.prev_color = curr_color
                    self.prev_depth = curr_depth
                    
                except Exception as e:
                    print(f"[!] Error processing frame {self.frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Update counters
            self.frame_count += 1
            fps_count += 1
            
            # FPS calculation
            if time.time() - last_time > 1.0:
                mode = "TSDF" if self.mesh_reconstruction else "PCL"
                vis_mode = "Integrated" if self.show_integrated_model else "Current"
                view_mode = self.visualization_mode.capitalize()
                avg_reg_time = (np.mean(registration_times) if registration_times else 0)
                print(f"[FPS] {fps_count} | Mode: {mode} | View: {vis_mode} ({view_mode}) | " + 
                      f"Recording: {'ON' if self.is_recording else 'OFF'} | " + 
                      f"Success: {success_count}/{self.frame_count} | Avg Reg: {avg_reg_time:.2f}s")
                fps_count = 0
                last_time = time.time()
                registration_times = []
            
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()

    def cleanup(self):
        """Clean up resources"""
        try:
            self.k4a.stop()
        except:
            pass
        print("[INFO] Kinect stopped. Program terminated.")

if __name__ == "__main__":
    KinectReconstructor().start_visualization()