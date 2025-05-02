import open3d as o3d
import os
import numpy as np
from datetime import datetime

class MeshReconstructionVisualizer:
    def __init__(self, results_folder="results"):
        self.results_folder = results_folder
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    def load_latest_mesh(self):
        """Load the latest mesh (.ply or .obj) from the results folder"""
        try:
            # List all mesh files in the results folder
            mesh_files = [f for f in os.listdir(self.results_folder) if f.endswith(".ply") or f.endswith(".obj")]
            if not mesh_files:
                print("No mesh files found")
                return None

            # Sort files by modification time to get the latest one
            latest_mesh = max(mesh_files, key=lambda f: os.path.getmtime(
                os.path.join(self.results_folder, f)))
            file_path = os.path.join(self.results_folder, latest_mesh)
            print(f"Loading mesh: {latest_mesh}")

            # Load mesh
            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) == 0:
                print("Warning: Empty mesh")
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            print(f"Error loading mesh: {str(e)}")
            return None

    def setup_camera_view(self, vis):
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_lookat([0.0, 0.0, 1.0])
        ctr.set_zoom(0.7)

    def visualize(self, mesh):
        if mesh is None:
            print("No mesh to visualize.")
            return
        try:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window("Mesh Viewer", width=1280, height=720)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.show_coordinate_frame = False
            if not vis.add_geometry(mesh):
                print("Failed to add mesh to visualizer!")
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
    visualizer = MeshReconstructionVisualizer()
    mesh = visualizer.load_latest_mesh()
    if mesh is not None:
        visualizer.visualize(mesh)

if __name__ == "__main__":
    main()
