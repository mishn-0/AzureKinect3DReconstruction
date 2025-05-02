import open3d as o3d
import os
import numpy as np
from datetime import datetime

class ChooseReconstructionVisualizer:
    def __init__(self, results_folder="results"):
        self.results_folder = results_folder
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    def list_reconstructions(self):
        """List all available reconstruction files in the results folder"""
        try:
            # List all .obj files in the results folder
            reconstruction_files = [f for f in os.listdir(self.results_folder) if f.endswith(".obj")]
            if not reconstruction_files:
                print("No reconstruction files found")
                return None
            # Sort files by modification time (newest first)
            reconstruction_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.results_folder, f)), reverse=True)
            return reconstruction_files
        except Exception as e:
            print(f"Error listing reconstructions: {str(e)}")
            return None

    def load_reconstruction(self, filename):
        """Load the specified reconstruction file"""
        try:
            file_path = os.path.join(self.results_folder, filename)
            print(f"Loading: {filename}")
            if filename.endswith(".ply") or filename.endswith(".obj"):
                geometry = o3d.io.read_triangle_mesh(file_path)
                if len(geometry.vertices) == 0 or len(geometry.triangles) == 0:
                    print("Warning: Empty or invalid mesh")
                    return None
                geometry.compute_vertex_normals()
            else:
                geometry = o3d.io.read_point_cloud(file_path)
                if len(geometry.points) == 0:
                    print("Warning: Empty point cloud")
                    return None
            return geometry
        except Exception as e:
            print(f"Error loading reconstruction: {str(e)}")
            return None

    def setup_camera_view(self, vis):
        """Set up the camera view to match the Kinect's perspective"""
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])  # Looking along negative z-axis
        ctr.set_up([0.0, 1.0, 0.0])      # Up direction (fixed: y-axis is up)
        ctr.set_lookat([0.0, 0.0, 1.0])  # Look at point 1 meter in front
        ctr.set_zoom(0.7)                # Adjust zoom level

    def visualize(self, geometry):
        if geometry is None:
            print("No geometry to visualize.")
            return
        try:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window("3D Reconstruction Viewer", width=1280, height=720)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
            opt.show_coordinate_frame = False  # Hide coordinate frame
            if not vis.add_geometry(geometry):
                print("Failed to add geometry to visualizer!")
                return
            self.setup_camera_view(vis)
            def reset_view(vis):
                self.setup_camera_view(vis)
                return True
            vis.register_key_callback(ord("R"), reset_view)
            print("\nControls:")
            print("- Left click + drag: Rotate")
            print("- Right click + drag: Pan")
            print("- Mouse wheel: Zoom")
            print("- R: Reset camera view")
            print("- Q or Esc: Quit")
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"Error during visualization: {str(e)}")

def main():
    visualizer = ChooseReconstructionVisualizer()
    reconstructions = visualizer.list_reconstructions()
    if reconstructions:
        print("\nAvailable reconstructions:")
        for i, filename in enumerate(reconstructions):
            print(f"{i + 1}. {filename}")
        choice = input("\nEnter the number of the reconstruction to visualize: ")
        try:
            choice = int(choice) - 1
            if 0 <= choice < len(reconstructions):
                geometry = visualizer.load_reconstruction(reconstructions[choice])
                if geometry is not None:
                    visualizer.visualize(geometry)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    else:
        print("No reconstructions available.")

if __name__ == "__main__":
    main()
