import os
import json
import numpy as np

def create_scene_gt_json(rgb_folder, pose_folder, output_path):
    scene_gt = {}

    # Iterate through each file in the rgb folder
    for filename in os.listdir(rgb_folder):
        if filename.endswith(".png"):
            # Extract the image base name (without extension)
            image_base_name = os.path.splitext(filename)[0]

            # Construct the path to the corresponding pose file
            pose_path = os.path.join(pose_folder, f"{image_base_name}.npy")

            if os.path.exists(pose_path):
                # Load the pose data
                pose = np.load(pose_path)

                # Extract rotation matrix and translation vector
                cam_R_m2c = pose[:3, :3].flatten().tolist()
                cam_t_m2c = pose[:3, 3].flatten().tolist()

                # Add the data to the scene_gt dictionary
                scene_gt[image_base_name] = [{
                    "cam_R_m2c": cam_R_m2c,
                    "cam_t_m2c": cam_t_m2c,
                    "obj_id": 1
                }]
            else:
                print(f"Pose file {pose_path} does not exist. Skipping image {image_base_name}.")

    # Save the scene_gt dictionary to a JSON file
    with open(output_path, 'w') as f:
        json.dump(scene_gt, f, indent=4)

    print(f"scene_gt.json file created successfully at {output_path}")

# Example usage
rgb_folder = "/home/utsav/IProject/data/dataset/lnd2/rgb"
pose_folder = "/home/utsav/IProject/data/dataset/lnd2/pose"
output_path = "/home/utsav/IProject/data/dataset/lnd2/scene_gt.json"

create_scene_gt_json(rgb_folder, pose_folder, output_path)
