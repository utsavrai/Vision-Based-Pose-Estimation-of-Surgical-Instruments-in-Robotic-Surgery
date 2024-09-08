import os
import numpy as np
import open3d as o3d
import copy

def read_pose_from_npy(file_path):
    pose = np.load(file_path)
    print(f"Ground Truth Pose: {pose}")
    return pose

def read_pose_from_txt(txt_file: str):
    """Reads a 4x4 pose matrix from a text file."""
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        pose_matrix = np.array([list(map(float, line.split())) for line in lines])
    print(f"Predicted Pose Matrix from TXT: {pose_matrix}")
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

def compare_poses_and_visualize(image_id, txt_file, npy_dir, obj_file, scale_factor=1.0):
    npy_file = f'{image_id}.npy'
    npy_path = os.path.join(npy_dir, npy_file)

    if os.path.exists(npy_path):
        # Load ground truth pose from the npy file
        pose_gt = read_pose_from_npy(npy_path)

        # Load the estimated pose from the txt file
        pose_est = read_pose_from_txt(txt_file)

        # Visualize the poses
        print(f"Visualizing differences for image ID {image_id}...")
        visualize_transformed_model(obj_file, pose_gt, pose_est, scale_factor)
    else:
        print(f"No corresponding .npy file for image ID {image_id}")

if __name__ == '__main__':
    image_id = 201  # Example image ID
    txt_file = f'../data/dataset/lnd2/train/000001/pvnet_output/predictions/{image_id}.txt'  # Specify the path to the text file containing the 4x4 pose matrix
    npy_dir = '../data/dataset/lnd2/eval_pose_pvnet'
    obj_file = '../FoundationPose/demo_data/Shaft_LND_Occ/tool.stl'
    scale_factor = 10000  # Adjust this scale factor as needed

    compare_poses_and_visualize(image_id, txt_file, npy_dir, obj_file, scale_factor)
