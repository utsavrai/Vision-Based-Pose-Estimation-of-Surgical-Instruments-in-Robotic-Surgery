import os
import json

# Define the common camera intrinsics
camera_intrinsics = [801.57991404, 0.0, 583.56089783, 0.0, 801.57991404, 309.78999329, 0.0, 0.0, 1.0]
depth_scale = 1.0

# Path to the rgb folder
rgb_folder = "/home/utsav/IProject/data/dataset/lnd2/rgb"

# Initialize the dictionary to store camera parameters for each image
scene_camera = {}

# Iterate through each file in the rgb folder
for filename in os.listdir(rgb_folder):
    if filename.endswith(".png"):
        # Extract the image base name (without extension)
        image_base_name = os.path.splitext(filename)[0]
        # Add the camera parameters for this image
        scene_camera[image_base_name] = {
            "cam_K": camera_intrinsics,
            "depth_scale": depth_scale
        }

# Path to save the scene_camera.json file
output_file = "/home/utsav/IProject/data/dataset/lnd2/scene_camera.json"

# Save the scene_camera dictionary to a JSON file
with open(output_file, 'w') as f:
    json.dump(scene_camera, f, indent=4)

print(f"scene_camera.json file created successfully at {output_file}")
