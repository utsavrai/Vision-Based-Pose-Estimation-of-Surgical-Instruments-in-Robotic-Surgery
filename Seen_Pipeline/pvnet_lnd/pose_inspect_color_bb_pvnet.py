import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple  # <-- Add this line

class CameraData:
    def __init__(self):
        self.K = np.array([
            [534.38660936, 0.0, 389.0405985533333],
            [0.0, 712.5154791466666, 275.3688829244444],
            [0.0, 0.0, 1.0]
        ])


def read_pose_from_txt(txt_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a 4x4 pose matrix from a text file and extracts the rotation matrix and translation vector."""
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        pose_matrix = np.array([list(map(float, line.split())) for line in lines])

    # Extract rotation (3x3) and translation (3x1) from the 4x4 pose matrix
    rotation_matrix = pose_matrix[:3, :3]
    translation_vector = pose_matrix[:3, 3]

    print(f"Rotation Matrix:\n{rotation_matrix}")
    print(f"Translation Vector:\n{translation_vector}")

    return rotation_matrix, translation_vector


def project_3d_bounding_box(camera_data: CameraData, rotation: np.ndarray, translation: np.ndarray, box_size: Tuple[float, float, float], model_offset: np.ndarray) -> np.ndarray:
    """Projects a 3D bounding box onto the image plane using the rotation matrix and translation vector."""
    
    half_size_x, half_size_y, half_size_z = box_size[0] / 2.0, box_size[1] / 2.0, box_size[2] / 2.0
    
    # Define the 3D bounding box corners in object space (centered at origin)
    box_corners = np.array([
        [-half_size_x, -half_size_y, -half_size_z],
        [ half_size_x, -half_size_y, -half_size_z],
        [ half_size_x,  half_size_y, -half_size_z],
        [-half_size_x,  half_size_y, -half_size_z],
        [-half_size_x, -half_size_y,  half_size_z],
        [ half_size_x, -half_size_y,  half_size_z],
        [ half_size_x,  half_size_y,  half_size_z],
        [-half_size_x,  half_size_y,  half_size_z]
    ])

    # Apply the model offset to align with the model's origin (shift the bounding box)
    box_corners += model_offset

    # Transform the corners to camera space
    box_corners_cam = (rotation @ box_corners.T).T + translation

    # Project the corners onto the image plane
    box_corners_proj = camera_data.K @ box_corners_cam.T
    box_corners_proj /= box_corners_proj[2, :]  # Normalize by depth

    return box_corners_proj[:2, :].T, box_corners_cam  # Return x, y pixel coordinates and the 3D camera space coordinates


def get_color_for_depth(depth: float, min_depth: float, max_depth: float) -> Tuple[int, int, int]:
    """Returns a BGR color based on the depth using a color gradient (closer = red, farther = blue)."""
    # Normalize the depth between 0 and 1
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    
    # Get the color in RGB format from a colormap (e.g., 'jet')
    cmap = plt.get_cmap('jet')
    color_rgb = cmap(normalized_depth)  # This returns a tuple (r, g, b, alpha) with values between 0 and 1
    
    # Convert the RGB values to the 0-255 range and cast them to Python int types
    color_rgb = tuple(map(int, (np.array(color_rgb[:3]) * 255)))

    # Convert RGB to BGR for OpenCV (OpenCV uses BGR by default)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

    # Ensure the final color is a tuple of native Python int, not numpy.int32
    return tuple(map(int, color_bgr))


def draw_3d_bounding_box_on_image(image: np.ndarray, corners_2d: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    """Draws the projected 3D bounding box on the image with a color gradient based on depth."""
    image_with_box = image.copy()
    corners_2d = corners_2d.astype(int)  # Ensure pixel coordinates are integers
    
    # Define the connections between corners to draw the edges of the box
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    # Get the depth of each corner (Z-coordinate in camera space)
    depths = corners_3d[:, 2]
    min_depth, max_depth = np.min(depths), np.max(depths)

    for start, end in connections:
        # Get average depth of the edge
        avg_depth = (depths[start] + depths[end]) / 2.0
        
        # Get the color for the edge based on its depth
        color = get_color_for_depth(avg_depth, min_depth, max_depth)

        # Convert corners from numpy array to tuple of integers
        start_pt = tuple(map(int, corners_2d[start]))
        end_pt = tuple(map(int, corners_2d[end]))

        # Debugging: print the values to ensure they are valid
        print(f"Drawing line from {start_pt} to {end_pt} with color {color}")
        
        # Draw the edge with the corresponding color
        cv2.line(image_with_box, start_pt, end_pt, color, 2)
    
    return image_with_box

def resize_to_960x540(image: np.ndarray) -> np.ndarray:
    """Resizes the image to 960x540 resolution while maintaining the aspect ratio."""
    return cv2.resize(image, (960, 540))
# Example usage:

# Initialize the camera data with intrinsic matrix
camera_data = CameraData()

# Define the box size (dimensions of the 3D bounding box)
box_size = (13.563604, 6.248400, 6.400742)

# Define the image ID for which you want to extract the rotation and translation
image_id = 25  # Example image ID #25 54 121 201

# Path to the text file containing the 4x4 pose matrix
txt_file_path = f'../data/dataset/lnd2/train/000001/pvnet_output/predictions/{image_id}.txt'  # Update with the path to your text file

# Load the rotation and translation from the text file
rotation_matrix, translation_vector = read_pose_from_txt(txt_file_path)

# Load the image (assuming you have the corresponding RGB image)
rgb_image = cv2.imread(f'data/custom_occluded/rgb/{image_id}.png')
model_offset = np.array([-3.9877885e+00, -0.0000000e+00,  -2.5000000e-06])

# Project the bounding box and get the corners in 2D and 3D camera space
corners_2d_pred, corners_3d_cam = project_3d_bounding_box(camera_data, rotation_matrix, translation_vector, box_size, model_offset)

# Draw the 3D bounding box with a color gradient based on depth
rgb_image_with_box = draw_3d_bounding_box_on_image(rgb_image, corners_2d_pred, corners_3d_cam)


rgb_image_with_box = draw_3d_bounding_box_on_image(rgb_image, corners_2d_pred, corners_3d_cam)
rgb_image_with_box_resized = resize_to_960x540(rgb_image_with_box)

# Display the resized result
cv2.imshow("3D Bounding Box with Gradient (960x540)", rgb_image_with_box_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
