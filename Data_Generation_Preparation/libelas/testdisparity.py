import numpy as np
import open3d as o3d
import cv2

# Load the left image and disparity map
left_image = cv2.imread('/home/utsav/IProject/data/lnd1/filtered_left_rect_images/1245.png')
disparity = cv2.imread('/home/utsav/IProject/libelas/build/img/1245_left_disp.pgm', cv2.IMREAD_UNCHANGED)

# Check if disparity needs scaling
disparity = disparity.astype(np.float32)

# Ensure valid disparity values
valid_disparity_mask = disparity > 0

# Convert disparity to depth
focal_length = 801.57991404  # Calibrated focal length
baseline = 5.5160  # Calibrated baseline
depth = np.zeros(disparity.shape, dtype=np.float32)
depth[valid_disparity_mask] = (focal_length * baseline) / (disparity[valid_disparity_mask] + 1e-6)

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
        X = (u - w / 2.0) * Z / focal_length
        Y = -(v - h / 2.0) * Z / focal_length  # Adjust y-coordinate to correct mirroring
        points.append([X, Y, Z])
        colors.append(left_image[v, u] / 255.0)

points = np.array(points)
colors = np.array(colors)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save the point cloud to a file
o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
