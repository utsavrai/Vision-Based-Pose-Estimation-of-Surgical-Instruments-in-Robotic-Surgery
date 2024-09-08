import os
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R

def apply_transformation(matrix, translation, rotation):
    # Ensure the input matrix is 3x4
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, np.array([0, 0, 0, 1])))
    else:
        raise ValueError(f"Unexpected pose matrix shape: {matrix.shape}")

    # Create translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    # Create rotation matrix using roll, pitch, and yaw
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.from_euler('xyz', rotation, degrees=True).as_matrix()

    # Combine translation and rotation into a single transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)

    # Debug: Print transformation matrix
    print("Transformation matrix:\n", transformation_matrix)

    # Apply transformation to the input matrix
    transformed_matrix = np.dot(transformation_matrix, matrix)
    
    return transformed_matrix[:3, :]  # Return back to 3x4 matrix

def process_npy_files(npy_folder_path, output_npy_path, translation, rotation):
    # Get list of all .npy files in the folder
    npy_files = glob.glob(os.path.join(npy_folder_path, '*.npy'))

    for npy_file in npy_files:
        # Process the .npy file
        pose_matrix = np.load(npy_file)
        transformed_matrix = apply_transformation(pose_matrix, translation, rotation)
        new_file_path = os.path.join(output_npy_path, os.path.basename(npy_file))
        np.save(new_file_path, transformed_matrix)
        print(f'Saved transformed matrix to {new_file_path}')

# Define translation and rotation for the transformation
translation = [-17.16, -5.86, 0.33]  # Values in millimeters
rotation = [0.0, -2.4, -93.60]  # Rotation in degrees

# Folder containing the .npy files
npy_folder_path = '/home/utsav/IProject/data/captured/lnd1/output_left'  # Replace with actual path

# Folder to save transformed npy files
output_npy_path = '/home/utsav/IProject/data/captured/lnd1/output_left_transformed'  # Replace with actual path

# Ensure the output directory exists
os.makedirs(output_npy_path, exist_ok=True)

# Process the .npy files
process_npy_files(npy_folder_path, output_npy_path, translation, rotation)
