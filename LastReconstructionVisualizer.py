import open3d as o3d
import os
import numpy as np
from datetime import datetime

class ReconstructionVisualizer:
    def __init__(self, results_folder="results"):
        self.results_folder = results_folder
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    def load_latest_reconstruction(self):
        """Load the latest reconstruction (either mesh or point cloud)"""
        try:
            # List all .ply files in the results folder
            ply_files = [f for f in os.listdir(self.results_folder) if f.endswith(".ply")]
            if not ply_files:
                print("No reconstruction files found")
                return None

            # Sort files by modification time to get the latest one
            latest_ply = max(ply_files, key=lambda f: os.path.getmtime(
                os.path.join(self.results_folder, f)))
            
            file_path = os.path.join(self.results_folder, latest_ply)
            print(f"Loading: {latest_ply}")
            
            # Try to load as mesh first
            if "mesh" in latest_ply:
                geometry = o3d.io.read_triangle_mesh(file_path)
                if len(geometry.vertices) == 0:
                    print("Warning: Empty mesh")
                geometry.compute_vertex_normals()
            else:
                geometry = o3d.io.read_point_cloud(file_path)
                if len(geometry.points) == 0:
                    print("Warning: Empty point cloud")
            
            return geometry
            
        except Exception as e:
            print(f"Error loading reconstruction: {str(e)}")
            return None

    def setup_camera_view(self, vis):
        """Set up the camera view to match the Kinect's perspective"""
        ctr = vis.get_view_control()
        # Set camera parameters to match Kinect's view
        ctr.set_front([0.0, 0.0, -1.0])  # Looking along negative z-axis
        ctr.set_up([0.0, 1.0, 0.0])      # Up direction (fixed: y-axis is up)
        ctr.set_lookat([0.0, 0.0, 1.0])  # Look at point 1 meter in front
        ctr.set_zoom(0.7)                # Adjust zoom level

    def visualize(self, geometry):
        if geometry is None:
            print("No geometry to visualize.")
            return

        try:
            # Create visualizer
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window("3D Reconstruction Viewer", width=1280, height=720)
            
            # Set up render options
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
            opt.show_coordinate_frame = False  # Hide coordinate frame
            
            # Add geometry
            if not vis.add_geometry(geometry):
                print("Failed to add geometry to visualizer!")
                return
            
            # Set up initial camera view
            self.setup_camera_view(vis)
            
            # Register key callback for resetting view
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
            
            # Run visualizer
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")

def main():
    visualizer = ReconstructionVisualizer()
    geometry = visualizer.load_latest_reconstruction()
    if geometry is not None:
        visualizer.visualize(geometry)

if __name__ == "__main__":
    main()
