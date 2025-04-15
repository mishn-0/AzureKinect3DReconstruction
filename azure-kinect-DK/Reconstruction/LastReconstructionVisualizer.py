import open3d as o3d
import os

class PointCloudVisualizer:
    def __init__(self, reconstruction_folder="reconstruction_output"):
        self.reconstruction_folder = reconstruction_folder

    def load_model(self):
        # Load the latest model (point cloud or mesh)
        try:
            # List all .ply files in the reconstruction folder
            ply_files = [f for f in os.listdir(self.reconstruction_folder) if f.endswith(".ply")]
            if not ply_files:
                print("No point cloud files found in the reconstruction folder.")
                return None

            # Sort files by modification date and get the latest one
            ply_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.reconstruction_folder, f)))
            latest_ply = ply_files[-1]
            print(f"Loading the latest model: {latest_ply}")

            # Load the point cloud from the latest .ply file
            pcd = o3d.io.read_point_cloud(os.path.join(self.reconstruction_folder, latest_ply))
            return pcd
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def visualize(self, pcd):
        if pcd is None:
            print("No point cloud to visualize.")
            return

        # Visualize the loaded point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window("3D Reconstruction Visualization")
        vis.add_geometry(pcd)

        # Customize the view
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(0.5)

        print("Press 'Q' to quit.")
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    visualizer = PointCloudVisualizer()
    pcd = visualizer.load_model()
    visualizer.visualize(pcd)
