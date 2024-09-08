# import numpy as np
# from stl import mesh

# # Load the STL file
# stl_path = 'tool.stl'
# your_mesh = mesh.Mesh.from_file(stl_path)

# # Extract the 3D points directly from the mesh vectors
# points = your_mesh.vectors.reshape(-1, 3)

# # Remove duplicate points
# points = np.unique(points, axis=0)

# print(f"Points shape: {points.shape}")

# # Save the points to a .npy file
# np.save("tool.npy", points)

# # Optionally, print some sample data
# print("Sample points:\n", points[:5])

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the STL file
stl_path = 'tool.stl'
mesh = trimesh.load_mesh(stl_path)

# Sample points on the surface of the mesh
num_points = 100000  # Adjust this number for desired density
points, _ = trimesh.sample.sample_surface_even(mesh, num_points)

# Mirror the points by negating the y-axis
# points[:, 1] = -points[:, 1]

# Visualize the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
plt.show()

# Save the points to a .npy file
np.save("tool.npy", points)

# Verify saved file
loaded_points = np.load("tool.npy")
print("Loaded points shape:", loaded_points.shape)
print("Sample loaded points:\n", loaded_points[:5])
