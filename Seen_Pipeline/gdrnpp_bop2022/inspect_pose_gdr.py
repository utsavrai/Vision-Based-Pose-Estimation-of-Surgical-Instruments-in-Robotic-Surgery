import os
import numpy as np
import open3d as o3d
import copy
import pandas as pd

def read_pose_from_npy(file_path):
    pose = np.load(file_path)
    print(f"Ground Truth Pose: {pose}")
    return pose

def extract_rotation_translation_from_csv(csv_file: str, im_id: int):
    """Extracts the rotation matrix and translation vector from the CSV based on the image ID."""
    df = pd.read_csv(csv_file, delimiter=',')
    print("Columns in CSV file:", df.columns)

    # Find the row matching the im_id
    row = df[df['im_id'] == im_id]
    if row.empty:
        raise ValueError(f"No entry found for im_id: {im_id}")

    # Extract and reshape the rotation matrix (R)
    R_values = list(map(float, row.iloc[0]['R'].split()))
    rotation_matrix = np.array(R_values).reshape(3, 3)

    # Extract the translation vector (t)
    t_values = list(map(float, row.iloc[0]['t'].split()))
    translation_vector = np.array(t_values)

    return rotation_matrix, translation_vector

def create_pose_matrix(rotation_matrix, translation_vector):
    """Creates a 4x4 pose matrix from a 3x3 rotation matrix and a translation vector."""
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = translation_vector
    print(f"Predicted Pose Matrix: {pose_matrix}")
    return pose_matrix

def transform_model(mesh, pose):
    vertices = np.asarray(mesh.vertices)
    homogenous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    transformed_vertices = (pose @ homogenous_vertices.T).T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices[:, :3])
    return mesh

def visualize_transformed_model(obj_file, pose_gt, pose_est, scale_factor=1.0):
    model = o3d.io.read_triangle_mesh(obj_file)
    model.compute_vertex_normals()

    # Scale the model
    model.scale(scale_factor, center=model.get_center())

    # Transform model using ground truth pose
    model_gt = copy.deepcopy(model)
    model_gt = transform_model(model_gt, pose_gt)
    model_gt.paint_uniform_color([0, 1, 0])  # Green for ground truth

    # Transform model using estimated pose
    model_est = copy.deepcopy(model)
    model_est = transform_model(model_est, pose_est)
    model_est.paint_uniform_color([1, 0, 0])  # Red for estimated

    # Visualize original and transformed models
    o3d.visualization.draw_geometries([model_gt, model_est], window_name="Pose Comparison")

def compare_poses_and_visualize(image_id, csv_file, npy_dir, obj_file, scale_factor=1.0):
    npy_file = f'{image_id:06d}.npy'
    npy_path = os.path.join(npy_dir, npy_file)

    if os.path.exists(npy_path):
        pose_gt = read_pose_from_npy(npy_path)

        # Extract rotation and translation from the CSV
        rotation_matrix, translation_vector = extract_rotation_translation_from_csv(csv_file, image_id)
        pose_est = create_pose_matrix(rotation_matrix, translation_vector)

        print(f"Visualizing differences for image ID {image_id}...")
        visualize_transformed_model(obj_file, pose_gt, pose_est, scale_factor)
    else:
        print(f"No corresponding .npy file for image ID {image_id}")

if __name__ == '__main__':
    image_id = 263  # Example image ID
    csv_file = 'occ.csv'
    npy_dir = '../data/dataset/lnd2/eval_pose'
    obj_file = '../FoundationPose/demo_data/Shaft_LND_Occ/tool.stl'
    scale_factor = 10000  # Adjust this scale factor as needed

    compare_poses_and_visualize(image_id, csv_file, npy_dir, obj_file, scale_factor)
