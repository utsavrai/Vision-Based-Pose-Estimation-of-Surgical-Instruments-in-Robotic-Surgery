import os
import numpy as np
import cv2
from scipy.ndimage import binary_dilation

def load_camera_intrinsics():
    # Replace with your actual camera intrinsic matrix
    return np.array([[801.57991404, 0, 583.56089783],
                     [0, 801.57991404, 309.78999329],
                     [0,  0,  1]])

def load_pose(file_path):
    return np.load(file_path)

def apply_pose_transformation(points_3d, pose):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    transformed_points = (pose @ points_3d_homogeneous.T).T
    return transformed_points[:, :3]

def project_to_image_plane(points_3d, camera_intrinsics):
    points_2d_homogeneous = (camera_intrinsics @ points_3d.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]
    return points_2d

def generate_binary_mask(points_2d, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for point in points_2d:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            mask[y, x] = 1
    return mask

def generate_depth_map(points_2d, transformed_points, image_shape):
    depth_map = np.full(image_shape, np.inf)
    for i, point in enumerate(points_2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            current_depth = depth_map[y, x]
            new_depth = transformed_points[i][2]
            if current_depth == np.inf or new_depth < current_depth:
                depth_map[y, x] = new_depth
    return depth_map

def create_visibility_mask(depth_map, depth_image, full_mask, threshold=1):
    visibility_mask = np.zeros_like(depth_image, dtype=np.uint8)
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            if depth_map[y, x] < np.inf and abs(depth_map[y, x] - depth_image[y, x]) < threshold:
                visibility_mask[y, x] = 1
    return visibility_mask & full_mask

def expand_visibility_mask(visibility_mask, depth_image, full_mask, iterations=2, threshold=1):
    kernel = np.ones((3, 3), np.uint8)
    expanded_mask = binary_dilation(visibility_mask, structure=kernel, iterations=iterations)
    
    refined_mask = np.zeros_like(visibility_mask)
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            if expanded_mask[y, x] and abs(depth_image[y, x] - depth_image[visibility_mask > 0].mean()) < threshold:
                refined_mask[y, x] = 1
                
    return refined_mask & full_mask

def get_visible_mask(points_3d, pose, camera_intrinsics, depth_image, full_mask):
    transformed_points = apply_pose_transformation(points_3d, pose)
    points_2d = project_to_image_plane(transformed_points, camera_intrinsics)
    generated_depth_map = generate_depth_map(points_2d, transformed_points, depth_image.shape)

    visibility_mask = create_visibility_mask(generated_depth_map, depth_image, full_mask)
    expanded_visibility_mask = expand_visibility_mask(visibility_mask, depth_image, full_mask)
    
    return expanded_visibility_mask

def overlay_mask_on_rgb(rgb_image, mask, alpha=0.5):
    overlay = rgb_image.copy()
    mask_colored = cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET)
    cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def process_images(rgb_folder, depth_folder, mask_folder, pose_folder, output_folder, points_3d):
    camera_intrinsics = load_camera_intrinsics()
    
    for filename in os.listdir(rgb_folder):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            base_name = os.path.splitext(filename)[0]
            rgb_path = os.path.join(rgb_folder, filename)
            depth_path = os.path.join(depth_folder, f'{base_name}.png')
            mask_path = os.path.join(mask_folder, f'{base_name}_000000.png')
            pose_path = os.path.join(pose_folder, f'{base_name}.npy')
            
            # Load images and pose
            rgb_image = cv2.imread(rgb_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pose = load_pose(pose_path)
            
            print(f"Processing {filename}")
            print("RGB Image Shape:", rgb_image.shape)
            print("Depth Image Shape:", depth_image.shape)
            print("Mask Shape:", full_mask.shape)
            print("Pose:\n", pose)  # Debug: Print the pose matrix
            
            # Get the visible mask
            visible_mask = get_visible_mask(points_3d, pose, camera_intrinsics, depth_image, full_mask)
            
            # Overlay the mask on the RGB image
            overlay_image = overlay_mask_on_rgb(rgb_image, visible_mask)
            cv2.imshow('Overlay Image', overlay_image)  # Debug: Show overlay image
            
            # Save the visible mask and overlay image
            mask_output_path = os.path.join(output_folder, f'{base_name}_visible.png')
            overlay_output_path = os.path.join(output_folder, f'{base_name}_overlay.png')
            cv2.imwrite(mask_output_path, visible_mask * 255)
            cv2.imwrite(overlay_output_path, overlay_image)
            cv2.waitKey(0)  # Debug: Wait for a key press to view intermediate results

# Example usage
points_3d_file = '/home/utsav/IProject/data/captured/dvrk_model/LND/tool.npy'  # Replace with the actual path to your .npy file
points_3d = np.load(points_3d_file)

rgb_folder = '/home/utsav/IProject/data/dataset/lnd2/train/000001/rgb'
depth_folder = '/home/utsav/IProject/data/dataset/lnd2/train/000001/depth'
mask_folder = '/home/utsav/IProject/data/dataset/lnd2/train/000001/mask'
pose_folder = '/home/utsav/IProject/data/dataset/lnd2/train/000001/pose'
output_folder = '/home/utsav/IProject/data/dataset/lnd2/train/000001/output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images(rgb_folder, depth_folder, mask_folder, pose_folder, output_folder, points_3d)
