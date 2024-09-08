import open3d as o3d
import numpy as np
import cv2

# Intrinsic matrix
intrinsic_matrix = np.array([[801.57991404, 0, 583.56089783],
                             [0, 801.57991404, 309.78999329],
                             [0, 0, 1.0]])

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/utsav/IProject/data/lnd1/pointcloud/1064.png.ply")

# Load the RGB image
rgb_image = cv2.imread("/home/utsav/IProject/data/lnd1/filtered_left_rect_images/1064.png")

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Get the dimensions of the RGB image
height, width, _ = rgb_image.shape

# Initialize the depth map
depth_map = np.full((height, width), np.nan, dtype=np.float32)

# Convert point cloud to depth map
for point in points:
    # Project the 3D point to 2D pixel coordinates using the provided intrinsic matrix
    pixel = np.dot(intrinsic_matrix, point[:3])
    pixel /= pixel[2]
    x, y = int(pixel[0]), int(pixel[1])
    
    # Ensure the pixel coordinates are within the image bounds
    if 0 <= x < width and 0 <= y < height:
        depth_map[y, x] = point[2]

# Replace NaN values with zeros (or another appropriate value)
depth_map = np.nan_to_num(depth_map, nan=0)

# Scale the depth values to retain precision and convert to uint16
scaling_factor = 1000  # For example, to retain millimeter precision
depth_map_scaled = (depth_map * scaling_factor).astype(np.uint16)

# Save the scaled depth map as a PNG image in uint16 format
depth_map_filename = "depth_map_scaled.png"
cv2.imwrite(depth_map_filename, depth_map_scaled)

# Optionally, display the depth map for visualization
depth_map_visualization = cv2.normalize(depth_map_scaled, None, 0, 65535, cv2.NORM_MINMAX)
depth_map_visualization = np.uint8(depth_map_visualization // 256)
cv2.imwrite("depth_map_visualization.png", depth_map_visualization)

# Display the RGB image and depth map visualization
cv2.imshow("RGB Image", rgb_image)
cv2.imshow("Depth Map Visualization", depth_map_visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()
