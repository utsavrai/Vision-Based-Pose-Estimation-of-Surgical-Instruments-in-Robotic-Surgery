import sys
import numpy as np
import open3d as o3d
import cv2
import os

# Arguments: left_image_path, disparity_path, output_pointcloud_path, depth_output_path
left_image_path = sys.argv[1]
disparity_path = sys.argv[2]
output_pointcloud_path = sys.argv[3]
depth_output_path = sys.argv[4]

# Load the left image and disparity map
left_image = cv2.imread(left_image_path)
disparity = np.load(disparity_path)

# Intrinsic parameters of the left camera
RECT_M1 = np.array([
    [801.57991404, 0, 583.56089783],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1.0]
])

# Intrinsic parameters of the second camera
RECT_M2 = np.array([
    [801.57991404, 0, 491.24100494],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1.0]
])


# Focal length and principal points
focal_length = RECT_M1[0, 0]
cx0 = RECT_M1[0, 2]
cx1 = RECT_M2[0, 2]


# Baseline
baseline = 5.516  # The baseline value from T[0] in meters

# Convert disparity to positive values
disparity = np.abs(disparity)

# Ensure valid disparity values
valid_disparity_mask = disparity > 0

# Convert disparity to depth using the provided formula
depth = np.zeros(disparity.shape, dtype=np.float32)
depth[valid_disparity_mask] = (focal_length * baseline) / abs(disparity[valid_disparity_mask] +cx1-cx0)
max_depth_mm = np.max(depth)
min_depth_mm = np.min(depth)

print(f"Maximum depth value: {max_depth_mm} mm")
print(f"Minimum depth value: {min_depth_mm} mm")

depth[depth > 200] = 0


# Apply scaling to depth
scaling_factor = 1.0
depth *= scaling_factor

# Convert depth to millimeter scale and to uint16 format
# depth_mm = (depth * 100).astype(np.uint16)
depth_mm = (depth).astype(np.uint16)
# Get the maximum and minimum values

# Save the depth map as PNG
cv2.imwrite(depth_output_path, depth_mm)

# Get image dimensions
# h, w, channels = left_image.shape

# # Create point cloud
# points = []
# colors = []

# for v in range(h):
#     for u in range(w):
#         if not valid_disparity_mask[v, u]:  # Skip invalid depth values
#             continue
#         Z = depth[v, u]
#         if Z <= 0 or not np.isfinite(Z):  # Skip invalid depth values
#             continue
#         X = (u - cx0) * Z / focal_length
#         Y = (v - RECT_M1[1, 2]) * Z / focal_length  # Using cy from RECT_M1
#         points.append([X, Y, Z])
#         colors.append(left_image[v, u] / 255.0)

# points = np.array(points)
# colors = np.array(colors)

# # Check if points and colors arrays are not empty
# if points.size == 0 or colors.size == 0:
#     raise ValueError("No valid points found in the point cloud")

# # Ensure points and colors arrays have the correct shape
# if points.ndim != 2 or points.shape[1] != 3:
#     raise ValueError(f"Expected points shape (N, 3), but got {points.shape}")
# if colors.ndim != 2 or colors.shape[1] != 3:
#     raise ValueError(f"Expected colors shape (N, 3), but got {colors.shape}")

# # Create Open3D point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # Apply scaling
# pcd.scale(scaling_factor, center=pcd.get_center())

# # Save the point cloud to a file
# o3d.io.write_point_cloud(output_pointcloud_path, pcd)

# print(f"Point cloud saved to {output_pointcloud_path}")
print(f"Depth map saved to {depth_output_path}")

# Optionally, visualize the point cloud
# o3d.visualization.draw_geometries([pcd])
