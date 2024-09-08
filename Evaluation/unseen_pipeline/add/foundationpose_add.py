import os
import numpy as np
import trimesh

# Function to load the 3D model points from the PLY file
def load_ply_model(ply_file_path):
    model = trimesh.load(ply_file_path)
    points = model.vertices  # Extract the vertex points
    return points

# Function to load a 3x4 pose from a .npy file
def load_pose_npy(npy_file_path):
    pose = np.load(npy_file_path)
    return pose

# Function to load a 4x4 pose from a .txt file
def load_pose_txt(txt_file_path):
    pose = np.loadtxt(txt_file_path)
    return pose

# Function to compute the ADD metric
def compute_add_metric(points, gt_pose, pred_pose):
    # Extract the rotation and translation components from the ground truth pose
    R_gt = gt_pose[:, :3]
    t_gt = gt_pose[:, 3]

    # Extract the rotation and translation components from the predicted pose
    R_pred = pred_pose[:3, :3]
    t_pred = pred_pose[:3, 3]

    # Apply the ground truth pose to the model points
    points_gt = (R_gt @ points.T).T + t_gt

    # Apply the predicted pose to the model points
    points_pred = (R_pred @ points.T).T + t_pred

    # Compute the Euclidean distances between corresponding points
    distances = np.linalg.norm(points_gt - points_pred, axis=1)

    # Compute the average distance (ADD metric)
    ADD = np.mean(distances)
    return ADD

# Main function to iterate over the poses and compute the ADD metric for each pair
def main(ground_truth_dir, predicted_dir, ply_file_path):
    points = load_ply_model(ply_file_path)

    add_results = []

    for gt_file in sorted(os.listdir(ground_truth_dir)):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        pred_file = gt_file.replace('.npy', '.txt')
        pred_path = os.path.join(predicted_dir, pred_file)

        # Load the ground truth and predicted poses
        gt_pose = load_pose_npy(gt_path)
        pred_pose = load_pose_txt(pred_path)

        # Compute the ADD metric
        add = compute_add_metric(points, gt_pose, pred_pose)
        add_results.append((gt_file, add))

    return add_results

# Function to compute accuracy for different thresholds
def compute_accuracy_for_thresholds(add_values, step):
    threshold_accuracy = []
    thresholds = np.arange(0, .51, step)  # Create an array of thresholds from 0 to 0.5 with a given step
    total_poses = len(add_values)
    model_diameter = 13.789724874301163  # in mm
    for threshold in thresholds:
        
        correct_poses = np.sum(add_values < (threshold* model_diameter))
        accuracy = (correct_poses / total_poses) * 100
        threshold_accuracy.append((threshold, accuracy))

    return threshold_accuracy

# Main part of the script
ground_truth_dir = '../data/dataset/lnd1/eval_pose'
predicted_dir = '../data/dataset/debug01/debug/ob_in_cam'
ply_file_path = '../data/dataset/lnd2/train/000001/models/obj_000001.ply'

# Compute the ADD metrics for all pose pairs
add_metrics = main(ground_truth_dir, predicted_dir, ply_file_path)

# Extract the ADD values
add_values = np.array([add for _, add in add_metrics])

# Set the step size
step_size = 0.05  # You can change this value to set your desired step size

# Compute the accuracy for each threshold
threshold_accuracy = compute_accuracy_for_thresholds(add_values, step_size)

# Print the results in ascending order of threshold
for threshold, accuracy in threshold_accuracy:
    print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.2f}%")
