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

# Function to project 3D points onto 2D image plane
def project_points(points, pose, K):
    # Apply the pose to the 3D points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    transformed_points = (pose @ points_homogeneous.T).T  # Apply the pose transformation

    # We only need the x, y, z components (ignore the homogeneous coordinate w)
    transformed_points = transformed_points[:, :3]

    # Project the 3D points onto the 2D image plane using the camera intrinsic matrix K
    projected_points = (K @ transformed_points.T).T  # Apply the camera intrinsics
    projected_points = projected_points[:, :2] / projected_points[:, 2, np.newaxis]  # Normalize by depth

    return projected_points


# Function to compute the projection distance metric
def compute_projection_metric(points, gt_pose, pred_pose, K):
    # Project the points using the ground truth and predicted poses
    projected_points_gt = project_points(points, gt_pose, K)
    projected_points_pred = project_points(points, pred_pose, K)

    # Compute the Euclidean distance between corresponding projected points
    distances = np.linalg.norm(projected_points_gt - projected_points_pred, axis=1)

    # Compute the mean distance
    mean_distance = np.mean(distances)
    return mean_distance

# Main function to iterate over the poses and compute the projection metric for each pair
def main(ground_truth_dir, predicted_dir, ply_file_path, K):
    points = load_ply_model(ply_file_path)

    projection_results = []

    for gt_file in sorted(os.listdir(ground_truth_dir)):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        pred_file = gt_file.replace('.npy', '.txt')
        pred_path = os.path.join(predicted_dir, pred_file)

        # Load the ground truth and predicted poses
        gt_pose = load_pose_npy(gt_path)
        pred_pose = load_pose_txt(pred_path)

        # Convert 3x4 gt_pose to 4x4 by adding [0 0 0 1] at the bottom
        gt_pose = np.vstack([gt_pose, [0, 0, 0, 1]])

        # Compute the projection metric
        projection_distance = compute_projection_metric(points, gt_pose, pred_pose, K)
        # projection_results.append((gt_file, projection_distance))
        projection_results.append(projection_distance)

    return projection_results

# Function to compute accuracy for different thresholds
def compute_accuracy_for_thresholds(projection_metrics, step):
    threshold_accuracy = []
    thresholds = np.arange(0, 51, step)  # Create an array of thresholds from 0 to 50 pixels with a given step
    total_poses = len(projection_metrics)

    for threshold in thresholds:
        correct_poses = np.sum(np.array(projection_metrics) < threshold)
        accuracy = (correct_poses / total_poses) * 100
        threshold_accuracy.append((threshold, accuracy))

    return threshold_accuracy

ground_truth_dir = '../data/dataset/lnd2/eval_pose_pvnet'
# ground_truth_dir = '../data/dataset/lnd2/train/000001/pose'
predicted_dir = '../data/dataset/lnd2/train/000001/pvnet_output/predictions'
# predicted_dir = '../data/dataset/debug03/debug/ob_in_cam'
ply_file_path = '../data/dataset/lnd2/train/000001/models/obj_000001.ply'

# Camera intrinsic matrix (example)
K = np.array([
    [801.57991404,0.0,583.56089783],
    [0.0,801.57991404,309.78999329],
    [0.0, 0.0, 1.0]
])

# Compute the projection distance metrics for all pose pairs
projection_metrics = main(ground_truth_dir, predicted_dir, ply_file_path, K)

# # Calculate the standard deviation of the projection metrics
# projection_std_dev = np.std(projection_metrics)

# # Threshold for determining correct pose (5 pixels)
# pixel_threshold = 50.0
# print(f"Threshold for correct pose: {pixel_threshold} pixels")

# # Calculate the number of correct poses
# correct_poses = np.sum(np.array(projection_metrics) < pixel_threshold)
# total_poses = len(projection_metrics)

# # Calculate the accuracy
# accuracy = (correct_poses / total_poses) * 100

# # Output the overall results
# print(f"Number of correct poses: {correct_poses}/{total_poses}")
# print(f"Accuracy: {accuracy:.2f}%")
# print(f"Standard Deviation of Projection Distances: {projection_std_dev:.6f} pixels")

# # Optionally, output the results with file names
# # for file_name, distance in projection_metrics:
# #     result = "Correct" if distance < pixel_threshold else "Incorrect"
# #     print(f"{file_name}: Projection Distance = {distance:.6f} pixels, Result: {result}")


# Set the step size
step_size = 5  # You can change this value to set your desired step size

# Compute the accuracy for each threshold
threshold_accuracy = compute_accuracy_for_thresholds(projection_metrics, step_size)

# Print the results in ascending order of threshold
for threshold, accuracy in threshold_accuracy:
    print(f"Threshold: {threshold} pixels, Accuracy: {accuracy:.2f}%")