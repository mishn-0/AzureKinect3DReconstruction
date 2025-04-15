import open3d as o3d
import os

class PointCloudVisualizer:
    def __init__(self, reconstruction_folder="reconstruction_output"):
        self.reconstruction_folder = reconstruction_folder

    def list_files(self):
        """List all .ply files in the reconstruction folder."""
        try:
            ply_files = [f for f in os.listdir(self.reconstruction_folder) if f.endswith(".ply")]
            if not ply_files:
                print("No point cloud files found in the reconstruction folder.")
                return None
            return ply_files
        except Exception as e:
            print(f"Error listing files: {e}")
            return None

    def load_model(self, filename):
        """Load a point cloud model from a specific .ply file."""
        try:
            pcd = o3d.io.read_point_cloud(os.path.join(self.reconstruction_folder, filename))
            return pcd
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def visualize(self, pcd):
        """Visualize the point cloud in 3D."""
        if pcd is None:
            print("No point cloud to visualize.")
            return

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

    # List all .ply files in the reconstruction folder
    ply_files = visualizer.list_files()
    if ply_files is None:
        exit()

    # Print the list of files
    print("Available .ply files:")
    for idx, file in enumerate(ply_files, 1):
        print(f"{idx}. {file}")

    # Ask the user to choose a file
    try:
        choice = int(input(f"Enter the number of the file to open (1-{len(ply_files)}): "))
        if 1 <= choice <= len(ply_files):
            filename = ply_files[choice - 1]
            print(f"Loading: {filename}")
            pcd = visualizer.load_model(filename)
            visualizer.visualize(pcd)
        else:
            print("Invalid choice. Exiting.")
    except ValueError:
        print("Invalid input. Exiting.")
