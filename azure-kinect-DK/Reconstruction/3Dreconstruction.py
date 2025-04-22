'''Step 1. Make fragments: build local geometric surfaces (referred to as fragments) from short subsequences of the input RGBD sequence. This part uses RGBD Odometry, Multiway registration, and RGBD integration.

Step 2. Register fragments: the fragments are aligned in a global space to detect loop closure. This part uses Global registration, ICP registration, and Multiway registration.

Step 3. Refine registration: the rough alignments are aligned more tightly. This part uses ICP registration, and Multiway registration.

Step 4. Integrate scene: integrate RGB-D images to generate a mesh model for the scene. This part uses RGBD integration.'''

import open3d as o3d
import numpy as np
from typing import List, Tuple
import os
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time
from datetime import datetime

class RGBDReconstruction:
    def __init__(self, voxel_size: float = 0.01):
        # Initialize Kinect
        self.k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()
        
        # Camera intrinsics for Azure Kinect DK
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
        
        self.voxel_size = voxel_size
        self.fragments = []
        self.poses = []
        self.rgbd_images = []  # Store original RGBD images for integration
        
        # Create output directory
        self.output_folder = "reconstruction_output"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Visualization window
        self.vis = None
        self.is_capturing = False
        
    def process_images(self, color_img, depth_img):
        """Process Kinect images to create RGBD image"""
        # Flip vertical and horizontal (if needed for your setup)
        color_img = cv2.flip(color_img, -1)
        depth_img = cv2.flip(depth_img, -1)
        
        # Convert BGR to RGB (Kinect provides BGR format)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Convert to Open3D format
        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            convert_rgb_to_intensity=False,  # Keep color information
            depth_scale=1000.0,  # Azure Kinect depth is in mm
            depth_trunc=3.0,  # 3m truncation
        )
        return rgbd
        
    def make_fragments(self, rgbd_images: List[o3d.geometry.RGBDImage]) -> List[o3d.geometry.TriangleMesh]:
        """Step 1: Create fragments from RGBD sequences"""
        fragments = []
        self.rgbd_images = rgbd_images  # Store RGBD images for later use
        
        for i, rgbd in enumerate(rgbd_images):
            # Create point cloud from RGBD image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, self.intrinsics)
            
            # Apply Kinect coordinate transformation
            pcd.transform(self.flip_transform)
            
            # Downsample point cloud
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2, max_nn=30))
            
            # Create mesh from point cloud
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9)[0]
            
            fragments.append(mesh)
            print(f"Created fragment {i+1}/{len(rgbd_images)}")
        
        self.fragments = fragments
        return fragments

    def register_fragments(self) -> List[np.ndarray]:
        """Step 2: Register fragments in global space"""
        poses = []
        poses.append(np.eye(4))  # First fragment is reference
        
        # Convert first fragment to point cloud for registration
        target_pcd = self.fragments[0].sample_points_uniformly(number_of_points=100000)
        
        for i in range(1, len(self.fragments)):
            # Convert current fragment to point cloud
            source_pcd = self.fragments[i].sample_points_uniformly(number_of_points=100000)
            
            # Perform global registration
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 2,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
            
            poses.append(result.transformation)
            print(f"Registered fragment {i+1}/{len(self.fragments)}")
        
        self.poses = poses
        return poses

    def refine_registration(self) -> List[np.ndarray]:
        """Step 3: Refine the registration"""
        refined_poses = []
        refined_poses.append(self.poses[0])  # First pose remains unchanged
        
        # Convert first fragment to point cloud for registration
        target_pcd = self.fragments[0].sample_points_uniformly(number_of_points=100000)
        
        for i in range(1, len(self.poses)):
            # Convert current fragment to point cloud
            source_pcd = self.fragments[i].sample_points_uniformly(number_of_points=100000)
            
            # Perform multi-way registration
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
            
            refined_poses.append(result.transformation)
            print(f"Refined registration for fragment {i+1}/{len(self.fragments)}")
        
        self.poses = refined_poses
        return refined_poses

    def integrate_scene(self) -> o3d.geometry.TriangleMesh:
        """Step 4: Integrate all fragments into final mesh"""
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=3 * self.voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        for i, (rgbd, pose) in enumerate(zip(self.rgbd_images, self.poses)):
            # Integrate RGBD image into volume
            volume.integrate(rgbd, self.intrinsics, pose)
            print(f"Integrated frame {i+1}/{len(self.rgbd_images)}")
        
        # Extract final mesh
        final_mesh = volume.extract_triangle_mesh()
        return final_mesh

    def visualize_capture(self, color_img, depth_img):
        """Show the current camera view"""
        # Convert depth to color for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Convert color image to BGR if it's RGBA
        if color_img.shape[2] == 4:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
        elif color_img.shape[2] == 3:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        
        # Resize images to same height if needed
        if color_img.shape[0] != depth_colormap.shape[0]:
            height = min(color_img.shape[0], depth_colormap.shape[0])
            color_img = cv2.resize(color_img, (int(color_img.shape[1] * height/color_img.shape[0]), height))
            depth_colormap = cv2.resize(depth_colormap, (int(depth_colormap.shape[1] * height/depth_colormap.shape[0]), height))
        
        # Combine color and depth images
        combined = np.hstack((color_img, depth_colormap))
        
        # Add text overlay
        cv2.putText(combined, "Press 'c' to capture frame", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show color + depth view
        cv2.imshow("Kinect View (Color + Depth)", combined)
        
        # Show integrated depth view
        integrated_depth = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), 
            cv2.COLORMAP_TURBO  # Using TURBO colormap for better depth visualization
        )
        cv2.imshow("Integrated Depth View", integrated_depth)

    def save_reconstruction(self, mesh, prefix=""):
        """Save reconstruction results with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if prefix:
            prefix = f"{prefix}_"
        
        # Save mesh
        mesh_path = os.path.join(self.output_folder, f"reconstruction_{prefix}{timestamp}_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Saved mesh to {mesh_path}")
        
        # Save point cloud
        pcd = mesh.sample_points_uniformly(number_of_points=100000)
        pcd_path = os.path.join(self.output_folder, f"reconstruction_{prefix}{timestamp}_vis.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Saved point cloud to {pcd_path}")

def main():
    # Initialize reconstructor
    reconstructor = RGBDReconstruction()
    
    # Collect frames from Kinect
    rgbd_images = []
    print("Press 'c' to capture a frame, 'q' to quit")
    
    try:
        while True:
            capture = reconstructor.k4a.get_capture()
            if capture.color is not None and capture.transformed_depth is not None:
                # Show current view
                reconstructor.visualize_capture(capture.color, capture.transformed_depth)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Capture frame
                    rgbd = reconstructor.process_images(capture.color, capture.transformed_depth)
                    rgbd_images.append(rgbd)
                    print(f"Captured frame {len(rgbd_images)}")
                elif key == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()
        reconstructor.k4a.stop()
    
    if len(rgbd_images) > 0:
        print(f"\nProcessing {len(rgbd_images)} captured frames...")
        
        # Run reconstruction pipeline
        fragments = reconstructor.make_fragments(rgbd_images)
        poses = reconstructor.register_fragments()
        refined_poses = reconstructor.refine_registration()
        final_mesh = reconstructor.integrate_scene()
        
        # Save the result
        reconstructor.save_reconstruction(final_mesh)
        print("Reconstruction complete!")
    else:
        print("No frames were captured.")

if __name__ == "__main__":
    main()