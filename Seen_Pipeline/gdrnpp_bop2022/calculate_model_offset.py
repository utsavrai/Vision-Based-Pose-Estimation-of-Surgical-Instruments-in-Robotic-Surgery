import open3d as o3d
import numpy as np

def compute_bounding_box_offset(ply_file: str) -> np.ndarray:
    """Computes the bounding box center offset from the 3D model in a .ply file."""
    
    # Read the 3D model from the .ply file
    mesh = o3d.io.read_triangle_mesh(ply_file)
    
    # Get the vertices of the mesh as a numpy array
    vertices = np.asarray(mesh.vertices)
    
    # Compute the min and max coordinates along each axis
    min_coords = vertices.min(axis=0)  # Minimum x, y, z
    max_coords = vertices.max(axis=0)  # Maximum x, y, z
    
    # Compute the bounding box center (geometric center)
    bounding_box_center = (min_coords + max_coords) / 2.0
    
    # Compute the model offset (assuming the model origin is at (0, 0, 0))
    # The offset is the shift required to move the bounding box center to the origin
    model_offset = -bounding_box_center
    
    return model_offset

# Example usage:
ply_file_path = 'datasets/BOP_DATASETS/lnd/models/obj_000001.ply'
model_offset = compute_bounding_box_offset(ply_file_path)
print("Model offset:", model_offset)
