import numpy as np
import cv2
import open3d as o3d

def create_point_cloud(rgb_image_path, depth_image_path, intrinsic_matrix, scale_factor=1.0):
    # Load the RGB and depth images
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * scale_factor

    # Get the dimensions of the images
    height, width = depth_image.shape

    # Generate a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to camera coordinates
    X = (x - intrinsic_matrix[0, 2]) * depth_image / intrinsic_matrix[0, 0]
    Y = (y - intrinsic_matrix[1, 2]) * depth_image / intrinsic_matrix[1, 1]
    Z = depth_image

    # Stack X, Y, and Z to get 3D points
    points_3d = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # Filter out points with zero depth
    mask = Z.flatten() > 0
    points_3d = points_3d[mask]
    colors = rgb_image.reshape(-1, 3)[mask]

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize RGB values to [0, 1]

    return pcd

def save_point_cloud(pcd, output_path):
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")

if __name__ == "__main__":
    # Paths to the RGB and depth images
    rgb_image_path = '/home/utsav/IProject/data/lnd1/filtered_left_rect_images/639.png'
    depth_image_path = '/home/utsav/IProject/data/3d_adjust/depth_map_corrected_int16.png'

    # Camera intrinsic parameters (example values, replace with your own)
    intrinsic_matrix = np.array([[801.57991404, 0, 583.56089783],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1.0]])

    # Depth scale factor (example, replace with your own if necessary)
    depth_scale = 0.001  # If depth image is in millimeters, convert to meters

    # Create the point cloud
    pcd = create_point_cloud(rgb_image_path, depth_image_path, intrinsic_matrix, scale_factor=depth_scale)

    # Save the point cloud
    output_path = 'colored_point_cloud.ply'
    save_point_cloud(pcd, output_path)
