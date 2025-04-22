import numpy as np
import open3d as o3d
import os
from datetime import datetime

class PointCloudToMesh:
    def __init__(self, voxel_size=0.01):
        """
        Initialize the PointCloudToMesh converter
        
        Args:
            voxel_size: Size of voxels for downsampling (default: 0.01)
        """
        self.voxel_size = voxel_size
        
    def preprocess_point_cloud(self, pcd):
        """
        Preprocess the point cloud by removing outliers and estimating normals
        
        Args:
            pcd: Open3D point cloud
            
        Returns:
            Processed point cloud
        """
        print("Preprocessing point cloud...")
        
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
        """
        Create a mesh using Poisson surface reconstruction
        
        Args:
            pcd: Open3D point cloud
            depth: Depth of the octree used for reconstruction
            width: Width parameter for the reconstruction
            scale: Scale parameter for the reconstruction
            linear_fit: Whether to use linear fitting
            
        Returns:
            Open3D triangle mesh
        """
        print("Creating mesh using Poisson reconstruction...")
        
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
        """
        Create a mesh using Ball Pivoting algorithm
        
        Args:
            pcd: Open3D point cloud
            radii: List of radii for the ball pivoting algorithm
            
        Returns:
            Open3D triangle mesh
        """
        print("Creating mesh using Ball Pivoting...")
        
        # Create mesh using Ball Pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh
    
    def save_mesh(self, mesh, output_dir="results", prefix=""):
        """
        Save the mesh to a file
        
        Args:
            mesh: Open3D triangle mesh
            output_dir: Directory to save the mesh
            prefix: Prefix for the output filename
            
        Returns:
            Path to the saved mesh file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            prefix = f"{prefix}_"
        output_path = os.path.join(output_dir, f"mesh_{prefix}{timestamp}.obj")
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Saved mesh to: {output_path}")
        
        return output_path

def main():
    # Example usage
    converter = PointCloudToMesh(voxel_size=0.01)
    
    # Load point cloud
    pcd_path = "results/latest_kinect_reconstruction.ply"  # Update this path as needed
    if not os.path.exists(pcd_path):
        print(f"Error: Point cloud file not found at {pcd_path}")
        return
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded point cloud with {len(pcd.points)} points")
    
    # Preprocess point cloud
    pcd = converter.preprocess_point_cloud(pcd)
    print(f"Preprocessed point cloud has {len(pcd.points)} points")
    
    # Try Poisson reconstruction first
    try:
        mesh = converter.create_mesh_poisson(pcd)
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        print("Trying Ball Pivoting instead...")
        # Try Ball Pivoting as fallback
        radii = [0.005, 0.01, 0.02, 0.04]  # Adjust these values based on your point cloud
        mesh = converter.create_mesh_ball_pivoting(pcd, radii)
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Save the mesh
    converter.save_mesh(mesh)

if __name__ == "__main__":
    main() 