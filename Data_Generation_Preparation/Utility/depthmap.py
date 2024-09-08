import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Load rectified stereo images
left_image_path = '/home/utsav/IProject/data/captured/rectified_left_image.png'  # Update with actual path
right_image_path = '/home/utsav/IProject/data/captured/rectified_right_image.png'  # Update with actual path

left_rectified = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
right_rectified = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

if left_rectified is None or right_rectified is None:
    raise FileNotFoundError("One or both of the rectified stereo images couldn't be loaded. Check the file paths.")

# Display the input images to verify they are correct
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(left_rectified, cmap='gray')
plt.title('Left Image')
plt.subplot(1, 2, 2)
plt.imshow(right_rectified, cmap='gray')
plt.title('Right Image')
plt.show()

# Camera intrinsic matrices for rectified images
# RECT_M1 = np.array([
#     [847.28300117, 0, 537.40095139],
#     [0, 847.28300117, 309.78999329],
#     [0, 0, 1.0]
# ])
# RECT_M2 = np.array([
#     [847.28300117, 0, 537.40095139],
#     [0, 847.28300117, 309.78999329],
#     [0, 0, 1.0]
# ])

# RECT_M1 = np.array([
#     [826.559161,     0.,         509.46055984],
#     [  0.,         826.559161,   308.02173233],
#     [  0.,           0.,           1.        ]
# ])



# Transformation between the left and right camera
# R = np.array([
#     [0.9999, -0.0054, -0.0086],
#     [0.0054, 1.0000, -0.0005],
#     [0.0086, 0.0004, 1.0000]
# ])
# T = np.array([5.5160, 0.0481, -0.0733])  # Baseline in millimeters if that is the actual unit

# Print baseline and focal length for verification
# baseline = np.linalg.norm(T)
baseline =  5.4782
# focal_length = RECT_M1[0, 0]
focal_length = 813.07050464
cx1 = 472.18536377
cx0 = 546.73575592
cy0 = 308.02173233
cy1 = 308.02173233
print("Baseline (T):", baseline)
print("Focal Length (f):", focal_length)

# StereoSGBM parameters for high accuracy
min_disparity = 0
num_disparities = 16 * 10  # Increase the range of disparities
block_size = 5  # Adjust block size for better matching 

stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                               numDisparities=num_disparities,
                               blockSize=block_size,
                               P1=8 * 3 * block_size ** 2,
                               P2=32 * 3 * block_size ** 2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=200,
                               speckleRange=64,
                               preFilterCap=63,
                               mode=cv2.STEREO_SGBM_MODE_HH)

# Compute disparity map
disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

# Print sample disparity values for debugging
print("Sample disparity values:")
print(disparity_map[100:105, 100:105])

# Check for NaN values
nan_count = np.isnan(disparity_map).sum()
print(f"Number of NaN values in disparity map: {nan_count}")

# Load the left image in color for point cloud coloring
left_image_color = cv2.imread(left_image_path)

# Convert disparity to positive values
# disparity = np.abs(disparity_map)

# Ensure valid disparity values
valid_disparity_mask = (disparity_map > 0) & (disparity_map < num_disparities) & (~np.isnan(disparity_map))


# Convert disparity to depth using the provided formula
depth = np.zeros(disparity_map.shape, dtype=np.float32)
depth[valid_disparity_mask] = (focal_length * baseline) / (disparity_map[valid_disparity_mask] + cx1-cx0 )

# Debug print statements
print(f"focal_length: {focal_length}")
# print(f"cx1: {cx1}")
# print(f"cx0: {cx0}")
print(f"baseline: {baseline}")
print(f"Valid disparity values: {np.sum(valid_disparity_mask)}")

# Visualization of the disparity map (optional)
plt.figure(figsize=(10, 5))
plt.imshow(disparity_map, cmap='jet')
plt.colorbar()
plt.title('Disparity Map')
plt.show()

# Get image dimensions
h, w = left_rectified.shape

# Thresholds
black_threshold = 50
depth_threshold = 10  # Minimum depth value to consider

# Create point cloud
points = []
colors = []

for v in range(h):
    for u in range(w):
        if not valid_disparity_mask[v, u]:  # Skip invalid depth values
            continue
        Z = depth[v, u]
        if Z <= depth_threshold or not np.isfinite(Z):  # Skip invalid depth values and too close points
            continue
        X = (u - cx0) * Z / focal_length
        Y = -(v - cy0) * Z / focal_length  # Using cy from RECT_M1
        color = left_image_color[v, u]
        if np.all(color < black_threshold):  # Skip points with black color below the threshold
            continue
        points.append([X, Y, Z])
        colors.append(color / 255.0)

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
# o3d.visualization.draw_geometries([pcd])
