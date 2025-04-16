import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time
import os
import copy

class KinectReconstructor:
    def __init__(self):
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

        self.flip_transform = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

        self.voxel_size = 0.01
        self.distance_threshold = 0.05
        self.keyframe_interval = 5

        self.global_model = o3d.geometry.PointCloud()
        self.previous_pcd = None
        self.current_transformation = np.identity(4)
        self.frame_count = 0
        self.is_recording = False

        self.output_folder = "reconstruction_output"
        os.makedirs(self.output_folder, exist_ok=True)

        self.vis = None

    def process_images(self, color_img, depth_img):
        color_img = cv2.cvtColor(cv2.flip(color_img, -1), cv2.COLOR_BGR2RGB)
        depth_img = cv2.flip(depth_img, -1)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_img),
            o3d.geometry.Image(depth_img),
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,
            depth_trunc=3.0,
        )
        return rgbd

    def preprocess_point_cloud(self, pcd):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return pcd

    def compute_fpfh(self, pcd):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )

    def register_frames(self, source, target):
        source_down = source.voxel_down_sample(self.voxel_size * 2)
        target_down = target.voxel_down_sample(self.voxel_size * 2)
        source_fpfh = self.compute_fpfh(source_down)
        target_fpfh = self.compute_fpfh(target_down)

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

        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, self.distance_threshold, result_fast.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        return result_icp.transformation

    def add_frame_to_model(self, new_pcd):
        if not self.global_model.points:
            self.global_model = new_pcd
            self.previous_pcd = new_pcd
            return True

        try:
            transformation = self.register_frames(new_pcd, self.previous_pcd)
            self.current_transformation = self.current_transformation @ transformation
            new_pcd.transform(self.current_transformation)
            self.global_model += new_pcd

            if self.frame_count % 10 == 0:
                self.global_model = self.global_model.voxel_down_sample(self.voxel_size)

            self.previous_pcd = new_pcd
            return True
        except Exception as e:
            print(f"[!] Registration failed: {e}")
            return False

    def start_visualization(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("3D Reconstruction from Kinect", 1280, 720)
        self.register_callbacks()

        while True:
            capture = self.k4a.get_capture()
            if capture.color is not None and capture.transformed_depth is not None:
                break

        rgbd = self.process_images(capture.color, capture.transformed_depth)
        initial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics)
        initial_pcd.transform(self.flip_transform)
        initial_pcd = self.preprocess_point_cloud(initial_pcd)
        self.vis.add_geometry(initial_pcd)
        self.setup_camera_view()

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
        self.vis.register_key_callback(ord("R"), self.toggle_recording)
        self.vis.register_key_callback(ord("S"), self.save_model)
        self.vis.register_key_callback(ord("C"), self.reset_model)

    def toggle_recording(self, vis):
        self.is_recording = not self.is_recording
        print(f"[INFO] Recording {'started' if self.is_recording else 'stopped'}")
        return True

    def save_model(self, vis):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{self.output_folder}/reconstruction_{timestamp}"

        if not self.global_model.has_normals():
            print("[INFO] Estimating normals for the global model before Poisson reconstruction...")
            self.global_model.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self.global_model.orient_normals_consistent_tangent_plane(30)

        o3d.io.write_point_cloud(f"{base}.ply", self.global_model)

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.global_model, depth=9
        )
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(f"{base}.obj", mesh)

        print(f"[INFO] Saved model to {base}.ply and {base}.obj")
        return True

    def reset_model(self, vis):
        self.global_model.clear()
        self.current_transformation = np.identity(4)
        self.frame_count = 0
        print("[INFO] Reconstruction reset")
        return True

    def run_visualization_loop(self, pcd):
        print("\nControls:\n  R: Toggle recording\n  S: Save model\n  C: Reset model\n  Esc: Exit\n")

        last_time = time.time()
        fps_count = 0

        while True:
            capture = self.k4a.get_capture()
            if capture.color is None or capture.transformed_depth is None:
                continue

            rgbd = self.process_images(capture.color, capture.transformed_depth)
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics)
            new_pcd.transform(self.flip_transform)
            new_pcd = self.preprocess_point_cloud(new_pcd)

            if not new_pcd.colors:
                print("[!] Warning: Point cloud has no colors!")

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            pcd.normals = new_pcd.normals

            if self.is_recording and self.frame_count % self.keyframe_interval == 2:
                if self.add_frame_to_model(copy.deepcopy(new_pcd)):
                    print(f"[+] Frame {self.frame_count} added - Total points: {len(self.global_model.points)}")

            self.frame_count += 1
            fps_count += 1

            if time.time() - last_time > 1.0:
                print(f"[FPS] {fps_count} | Recording: {'ON' if self.is_recording else 'OFF'} | Frames: {self.frame_count}")
                fps_count = 0
                last_time = time.time()

            self.vis.update_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def cleanup(self):
        self.k4a.stop()
        print("[INFO] Kinect stopped. Program terminated.")

if __name__ == "__main__":
    KinectReconstructor().start_visualization()
