# import os
# import numpy as np
# import pandas as pd
# import trimesh

# # Function to load the 3D model points from the PLY file
# def load_ply_model(ply_file_path):
#     model = trimesh.load(ply_file_path)
#     points = model.vertices  # Extract the vertex points
#     return points

# # Function to load a 3x4 pose from a .npy file
# def load_pose_npy(npy_file_path):
#     pose = np.load(npy_file_path)
#     return pose

# # Function to parse the R and t components from the CSV
# def parse_pose_from_csv(row):
#     R = np.array(row['R'].split(), dtype=float).reshape(3, 3)
#     t = np.array(row['t'].split(), dtype=float)
#     pose = np.hstack((R, t.reshape(-1, 1)))
#     return pose

# # Function to compute the ADD metric
# def compute_add_metric(points, gt_pose, pred_pose):
#     # Extract the rotation and translation components from the ground truth pose
#     R_gt = gt_pose[:, :3]
#     t_gt = gt_pose[:, 3]

#     # Extract the rotation and translation components from the predicted pose
#     R_pred = pred_pose[:, :3]
#     t_pred = pred_pose[:, 3]

#     # Apply the ground truth pose to the model points
#     points_gt = (R_gt @ points.T).T + t_gt

#     # Apply the predicted pose to the model points
#     points_pred = (R_pred @ points.T).T + t_pred

#     # Compute the Euclidean distances between corresponding points
#     distances = np.linalg.norm(points_gt - points_pred, axis=1)

#     # Compute the average distance (ADD metric)
#     ADD = np.mean(distances)
#     return ADD

# # Main function to iterate over the poses and compute the ADD metric for each pair
# def main(ground_truth_dir, csv_file_path, ply_file_path):
#     points = load_ply_model(ply_file_path)

#     # Load the CSV file containing the predicted poses
#     csv_data = pd.read_csv(csv_file_path)

#     add_results = []

#     for gt_file in sorted(os.listdir(ground_truth_dir)):
#         # Extract the im_id from the ground truth file name (e.g., 000253.npy -> 253)
#         im_id = int(gt_file.split('.')[0])

#         # Find the corresponding row in the CSV
#         pred_row = csv_data[csv_data['im_id'] == im_id]

#         if pred_row.empty:
#             print(f"Prediction for im_id {im_id} not found in CSV. Skipping.")
#             continue

#         gt_path = os.path.join(ground_truth_dir, gt_file)

#         # Load the ground truth pose
#         gt_pose = load_pose_npy(gt_path)

#         # Parse the predicted pose from the CSV row
#         pred_pose = parse_pose_from_csv(pred_row.iloc[0])

#         # Compute the ADD metric
#         add = compute_add_metric(points, gt_pose, pred_pose)
#         add_results.append((gt_file, add))

#     return add_results

# ground_truth_dir = '../data/dataset/lnd1/eval_pose'
# csv_file_path = '../gdrnpp_bop2022/occ.csv'
# csv_file_path = '../gdrnpp_bop2022/nonocc.csv'
# ply_file_path = '../data/dataset/lnd2/train/000001/models/obj_000001.ply'

# # Compute the ADD metrics for all pose pairs
# add_metrics = main(ground_truth_dir, csv_file_path, ply_file_path)

# # Threshold for determining correct pose (10% of the model's diameter)
# # model_diameter = 13.789724874301163  # in mm
# # threshold = 0.1 * model_diameter  # 10% of the diameter
# # print(f"Threshold for correct pose: {threshold:.6f} mm")

# # # Calculate the number of correct poses
# # correct_poses = np.sum(np.array([add for _, add in add_metrics]) < threshold)
# # total_poses = len(add_metrics)

# # # Calculate the accuracy
# # accuracy = (correct_poses / total_poses) * 100

# # # Output the overall results
# # print(f"Number of correct poses: {correct_poses}/{total_poses}")
# # print(f"Accuracy: {accuracy:.2f}%")

# # Optionally, output the results with file names
# # for file_name, add_value in add_metrics:
# #     result = "Correct" if add_value < threshold else "Incorrect"
# #     print(f"{file_name}: ADD = {add_value:.6f} mm, Result: {result}")

# add_values = np.array([add for _, add in add_metrics])

# # Compute the standard deviation of the ADD values
# add_std = np.std(add_values)

# # Output the standard deviation of ADD
# print(f"Standard Deviation of ADD: {add_std:.6f} mm")

# # Threshold for determining correct pose (10% of the model's diameter)
# model_diameter = 13.789724874301163  # in mm
# threshold = 0.1 * model_diameter  # 10% of the diameter
# print(f"Threshold for correct pose: {threshold:.6f} mm")

# # Calculate the number of correct poses
# correct_poses = np.sum(add_values < threshold)
# total_poses = len(add_metrics)

# # Calculate the accuracy
# accuracy = (correct_poses / total_poses) * 100

# # Output the overall results
# print(f"Number of correct poses: {correct_poses}/{total_poses}")
# print(f"Accuracy: {accuracy:.2f}%")


import os
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation as R

# Function to load the 3D model points from the PLY file
def load_ply_model(ply_file_path):
    model = trimesh.load(ply_file_path)
    points = model.vertices  # Extract the vertex points
    return points

# Function to load a 3x4 pose from a .npy file
def load_pose_npy(npy_file_path):
    pose = np.load(npy_file_path)
    return pose

# Function to parse the R and t components from the CSV
def parse_pose_from_csv(row):
    R = np.array(row['R'].split(), dtype=float).reshape(3, 3)
    t = np.array(row['t'].split(), dtype=float)
    pose = np.hstack((R, t.reshape(-1, 1)))
    return pose

# Function to compute the ADD metric
def compute_add_metric(points, gt_pose, pred_pose):
    # Extract the rotation and translation components from the ground truth pose
    R_gt = gt_pose[:, :3]
    t_gt = gt_pose[:, 3]

    # Extract the rotation and translation components from the predicted pose
    R_pred = pred_pose[:, :3]
    t_pred = pred_pose[:, 3]

    # Apply the ground truth pose to the model points
    points_gt = (R_gt @ points.T).T + t_gt

    # Apply the predicted pose to the model points
    points_pred = (R_pred @ points.T).T + t_pred

    # Compute the Euclidean distances between corresponding points
    distances = np.linalg.norm(points_gt - points_pred, axis=1)

    # Compute the average distance (ADD metric)
    ADD = np.mean(distances)
    return ADD

# Function to compute translation and rotation error for 5mm 5 degree metric
def compute_5mm_5degree_metric(gt_pose, pred_pose):
    # Translation error
    translation_gt = gt_pose[:, 3]
    translation_pred = pred_pose[:, 3]
    translation_error = np.linalg.norm(translation_gt - translation_pred)  # in mm

    # Rotation error
    R_gt = gt_pose[:, :3]
    R_pred = pred_pose[:, :3]

    # Compute relative rotation matrix
    R_rel = R_gt @ R_pred.T
    # Convert to rotation vector and get the angle (in degrees)
    angle_error = np.abs(R.from_matrix(R_rel).as_euler('xyz', degrees=True)).sum()  # Sum of absolute angle differences

    # Check if errors are within 5mm and 5 degrees
    is_within_threshold = (translation_error <= 5.0) and (angle_error <= 5.0)
    return is_within_threshold

# Main function to iterate over the poses and compute all metrics
def main(ground_truth_dir, csv_file_path, ply_file_path):
    points = load_ply_model(ply_file_path)

    # Load the CSV file containing the predicted poses
    csv_data = pd.read_csv(csv_file_path)

    add_results = []
    metric_5mm_5degree_results = []

    for gt_file in sorted(os.listdir(ground_truth_dir)):
        # Extract the im_id from the ground truth file name (e.g., 000253.npy -> 253)
        im_id = int(gt_file.split('.')[0])

        # Find the corresponding row in the CSV
        pred_row = csv_data[csv_data['im_id'] == im_id]

        if pred_row.empty:
            print(f"Prediction for im_id {im_id} not found in CSV. Skipping.")
            continue

        gt_path = os.path.join(ground_truth_dir, gt_file)

        # Load the ground truth pose
        gt_pose = load_pose_npy(gt_path)

        # Parse the predicted pose from the CSV row
        pred_pose = parse_pose_from_csv(pred_row.iloc[0])

        # Compute the ADD metric
        add = compute_add_metric(points, gt_pose, pred_pose)
        add_results.append((gt_file, add))

        # Compute the 5 mm 5 degree metric
        is_within_5mm_5degree = compute_5mm_5degree_metric(gt_pose, pred_pose)
        metric_5mm_5degree_results.append((gt_file, is_within_5mm_5degree))

    return add_results, metric_5mm_5degree_results

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

# Example directories and file paths
ground_truth_dir = '../data/dataset/lnd2/eval_pose'
csv_file_path = '../gdrnpp_bop2022/occ.csv'
ply_file_path = '../data/dataset/lnd2/train/000001/models/obj_000001.ply'

# Compute the ADD metrics and 5mm 5 degree metrics for all pose pairs
add_metrics, metric_5mm_5degree = main(ground_truth_dir, csv_file_path, ply_file_path)

# # Compute summary statistics for ADD metric
# add_values = np.array([add for _, add in add_metrics])

# # Compute the standard deviation of the ADD values
# add_std = np.std(add_values)

# # Output the standard deviation of ADD
# print(f"Standard Deviation of ADD: {add_std:.6f} mm")

# # Threshold for determining correct pose (10% of the model's diameter)
# model_diameter = 13.789724874301163  # in mm
# threshold = 0.5 * model_diameter  # 10% of the diameter
# print(f"Threshold for correct pose: {threshold:.6f} mm")

# # Calculate the number of correct poses based on ADD threshold
# correct_poses = np.sum(add_values < threshold)
# total_poses = len(add_metrics)

# # Calculate the accuracy
# accuracy = (correct_poses / total_poses) * 100

# # Output the overall ADD results
# print(f"Number of correct poses (ADD): {correct_poses}/{total_poses}")
# print(f"Accuracy (ADD): {accuracy:.2f}%")

# # Output the results for 5 mm 5 degree metric
# correct_5mm_5degree = np.sum([res for _, res in metric_5mm_5degree])
# accuracy_5mm_5degree = (correct_5mm_5degree / total_poses) * 100
# print(f"5 mm 5 degree Accuracy: {accuracy_5mm_5degree:.2f}%")


# Extract the ADD values
add_values = np.array([add for _, add in add_metrics])

# Set the step size
step_size = 0.05  # You can change this value to set your desired step size

# Compute the accuracy for each threshold
threshold_accuracy = compute_accuracy_for_thresholds(add_values, step_size)

# Print the results in ascending order of threshold
for threshold, accuracy in threshold_accuracy:
    print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.2f}%")