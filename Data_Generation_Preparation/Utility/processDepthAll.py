import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
# Define the directories
left_folder = '/home/utsav/IProject/data/lnd1/filtered_left_rect_images'
right_folder = '/home/utsav/IProject/data/lnd1/rect_right'
depth_folder_int16 = '/home/utsav/IProject/data/lnd1/depth'
depth_folder_int32 = '/home/utsav/IProject/data/lnd1/depth_in32'
pointcloud_folder = '/home/utsav/IProject/data/lnd1/pointcloud'

# Ensure the depth folder exists
os.makedirs(depth_folder_int16, exist_ok=True)
os.makedirs(depth_folder_int32, exist_ok=True)
# Camera intrinsic matrices for rectified images
RECT_M1 = np.array([
    [801.57991404, 0, 583.56089783],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1.0]
])
RECT_M2 = np.array([
    [801.57991404, 0, 491.24100494],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1.0]
])

# Transformation between the left and right camera
R = np.array([
    [0.9999, -0.0054, -0.0086],
    [0.0054, 1.0000, -0.0005],
    [0.0086, 0.0004, 1.0000]
])
T = np.array([5.5160, 0.0481, -0.0733])

# Print baseline and focal length for verification
baseline = np.linalg.norm(T)
focal_length = RECT_M1[0, 0]
print("Baseline (T):", baseline)
print("Focal Length (f):", focal_length)

# StereoSGBM parameters for high accuracy
min_disparity = 0
num_disparities = 16 * 10  # Increase the range of disparities
block_size = 1

stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                               numDisparities=num_disparities,
                               blockSize=block_size,
                               P1=8 * 3 * block_size ** 2,
                               P2=32 * 3 * block_size ** 2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               preFilterCap=63,
                               mode=cv2.STEREO_SGBM_MODE_HH)

# Process each pair of images
for filename in os.listdir(left_folder):
    if filename.endswith('.png'):
        left_image_path = os.path.join(left_folder, filename)
        right_image_path = os.path.join(right_folder, filename)

        left_rectified = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        right_rectified = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

        if left_rectified is None or right_rectified is None:
            print(f"One or both of the rectified stereo images couldn't be loaded for {filename}. Skipping.")
            continue

        # Compute disparity map
        disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

        # Print sample disparity values for debugging
        # print(f"Sample disparity values for {filename}:")
        # print(disparity_map[100:105, 100:105])

        # Post-processing: apply a median filter and bilateral filter to the disparity map to reduce noise
        disparity_map = cv2.medianBlur(disparity_map, 5)
        disparity_map_filtered = cv2.bilateralFilter(disparity_map, 9, 75, 75)

        # Normalize the disparity map for visualization
        disparity_map_normalized = cv2.normalize(disparity_map_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_map_normalized = np.uint8(disparity_map_normalized)

        # Save the disparity map
        # cv2.imwrite(f'disparity_map_{filename}', disparity_map_normalized)

        # Optionally save the raw disparity map
        # cv2.imwrite(f'disparity_map_raw_{filename}', disparity_map)

        # Compute depth map from disparity map
        depth_map = (focal_length * baseline) / (disparity_map + 1e-6)

        # Convert depth map to int32 format
        depth_map_int32 = np.clip(depth_map, 0, np.iinfo(np.int32).max).astype(np.int32)

        # Save the depth map as int32 PNG file
        # depth_map_filename = os.path.join(depth_folder, f'{filename}')
        # cv2.imwrite(depth_map_filename, depth_map_int32)

        # Generate point cloud from depth map
        h, w = left_rectified.shape[:2]

        # Verify Q matrix for correctness for left-to-right transformation
        Q_left_to_right = np.float32([[1, 0, 0, -RECT_M1[0, 2]],
                                      [0, 1, 0, -RECT_M1[1, 2]],
                                      [0, 0, 0, focal_length],
                                      [0, 0, -1/baseline, 0]])

        # Verify Q matrix for correctness for right-to-left transformation
        Q_right_to_left = np.float32([[1, 0, 0, -RECT_M1[0, 2]],
                                      [0, 1, 0, -RECT_M1[1, 2]],
                                      [0, 0, 0, focal_length],
                                      [0, 0, 1/baseline, 0]])  # Note the change in sign for the baseline

        # Generate and compare point clouds
        def generate_point_cloud(Q, disparity_map, left_image_path, mask, black_threshold=50):
            points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
            colors = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)

            # Define the black pixel threshold
            black_pixel_mask = np.all(colors < black_threshold, axis=2)

            # Mask for points with valid depth and non-black colors within the threshold
            valid_mask = mask & ~black_pixel_mask

            output_points = points_3D[valid_mask]
            output_colors = colors[valid_mask]
            output_points = output_points.reshape(-1, 3).astype(np.float64)
            output_colors = (output_colors.reshape(-1, 3) / 255.0).astype(np.float64)
            valid_mask = valid_mask.reshape(disparity_map.shape)  # Ensure the mask is reshaped correctly
            return output_points, output_colors, points_3D, valid_mask

        # Mask for valid points
        mask = (disparity_map > disparity_map.min()) & np.isfinite(disparity_map) & (depth_map < 1000)  # Filter out points with very high depth values

        # # Generate point cloud for left-to-right transformation
        # output_points_lr, output_colors_lr,_,_ = generate_point_cloud(Q_left_to_right, depth_map, left_image_path, mask)

        # # Create Open3D point cloud for left-to-right
        # point_cloud_lr = o3d.geometry.PointCloud()
        # point_cloud_lr.points = o3d.utility.Vector3dVector(output_points_lr)
        # point_cloud_lr.colors = o3d.utility.Vector3dVector(output_colors_lr)
        # output_ply_path_lr = os.path.join(depth_folder, f'point_cloud_lr_{filename}.ply')
        # o3d.io.write_point_cloud(output_ply_path_lr, point_cloud_lr, write_ascii=True)
        # print(f"Point cloud (left-to-right) saved to {output_ply_path_lr}")

        # Generate point cloud for right-to-left transformation
        output_points_rl, output_colors_rl, points_3D_rl, valid_mask_rl = generate_point_cloud(Q_right_to_left, depth_map, left_image_path, mask)

        # Create Open3D point cloud for right-to-left
        point_cloud_rl = o3d.geometry.PointCloud()
        point_cloud_rl.points = o3d.utility.Vector3dVector(output_points_rl)
        point_cloud_rl.colors = o3d.utility.Vector3dVector(output_colors_rl)
        # output_ply_path_rl = os.path.join(pointcloud_folder, f'{filename}.ply')
        # o3d.io.write_point_cloud(output_ply_path_rl, point_cloud_rl, write_ascii=True)
        # print(f"Point cloud (right-to-left) saved to {output_ply_path_rl}")

        # Extract depth values from the valid points of the 3D points array
        depth_map_corrected = np.zeros(disparity_map.shape)
        depth_map_corrected[valid_mask_rl] = points_3D_rl[valid_mask_rl][:, 2]

        # # Convert the corrected depth map to int32 format
        # depth_map_corrected_int32 = np.clip(depth_map_corrected, 0, np.iinfo(np.int32).max).astype(np.int32)

        # # # Save the corrected depth map as int32 PNG file
        # depth_map_filename_int32 = os.path.join(depth_folder_int32, f'{filename}')
        # cv2.imwrite(depth_map_filename_int32, depth_map_corrected_int32)

        # print(f"Processed and saved depth map for {filename}")


        # Ensure depth values are in millimeters and within the uint16 range
        scaling_factor = 1000  # For example, to retain millimeter precision
        depth_map_corrected_uint16 = (depth_map_corrected * scaling_factor).astype(np.uint16)

        # Convert the numpy array to an Image object with mode 'I;16'
        depth_image = Image.fromarray(depth_map_corrected_uint16, mode='I;16')

        # Save the corrected depth map as uint16 PNG file
        corrected_depth_map_filename = os.path.join(depth_folder_int16, f'{filename}')
        depth_image.save(corrected_depth_map_filename)

        print(f"Processed and saved depth map for {filename}")

print("Processing complete.")
