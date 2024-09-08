import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from plyfile import PlyData

def compute_diameter(ply_file_path):
    # Load the PLY file
    ply_data = PlyData.read(ply_file_path)
    vertex_data = ply_data['vertex']

    # Extract the x, y, z coordinates
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    # Compute the convex hull of the points
    hull = ConvexHull(points)

    # Get the vertices of the convex hull
    hull_points = points[hull.vertices]

    # Compute the pairwise distance matrix of the hull points
    dist_matrix = distance_matrix(hull_points, hull_points)

    # The diameter is the maximum distance in this matrix
    diameter = np.max(dist_matrix)

    return diameter

# Example usage
ply_file_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/reconstructed_mesh.ply'
diameter = compute_diameter(ply_file_path)
print(f'The diameter of the model is: {diameter}')
