import numpy as np
import open3d as o3d
import cv2

# Load the left image and disparity map
left_image = cv2.imread('/home/utsav/IProject/data/lnd1/rect_left/1252.png')
disparity = np.load('/home/utsav/IProject/RAFT-Stereo/demo_output/rect_left.npy')

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
depth[valid_disparity_mask] = (focal_length * baseline) / (disparity[valid_disparity_mask] +cx1-cx0)

# Debug print statements
print(f"focal_length: {focal_length}")
print(f"cx1: {cx1}")
print(f"cx0: {cx0}")
print(f"baseline: {baseline}")
print(f"Valid disparity values: {np.sum(valid_disparity_mask)}")

# Get image dimensions
h, w, channels = left_image.shape

# Create point cloud
points = []
colors = []

for v in range(h):
    for u in range(w):
        if not valid_disparity_mask[v, u]:  # Skip invalid depth values
            continue
        Z = depth[v, u]
        # if Z <= 0 or not np.isfinite(Z):  # Skip invalid depth values
        #     continue
        X = (u - cx0) * Z / focal_length
        Y = -(v - RECT_M1[1, 2]) * Z / focal_length  # Using cy from RECT_M1
        points.append([X, Y, Z])
        colors.append(left_image[v, u] / 255.0)

points = np.array(points)
colors = np.array(colors)

# Check if points and colors arrays are not empty
if points.size == 0 or colors.size == 0:
    raise ValueError("No valid points found in the point cloud")

# Ensure points and colors arrays have the correct shape
if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError(f"Expected points shape (N, 3), but got {points.shape}")
if colors.ndim != 2 or colors.shape[1] != 3:
    raise ValueError(f"Expected colors shape (N, 3), but got {colors.shape}")

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)


# Apply scaling
scaling_factor = 1.0
pcd.scale(scaling_factor, center=pcd.get_center())

# Save the point cloud to a file
o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
