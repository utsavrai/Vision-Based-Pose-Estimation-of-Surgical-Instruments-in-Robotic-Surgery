import os
import glob
import subprocess
import shutil

# Define paths
cad_path = "/vol/bitbucket/ur23/lnd_new_capture/lnd2_sam6d/models/obj_000001.ply"
rgb_dir = "/vol/bitbucket/ur23/lnd_new_capture/lnd2_sam6d/rgb"
depth_dir = "/vol/bitbucket/ur23/lnd_new_capture/lnd2_sam6d/depth"
camera_path = "/vol/bitbucket/ur23/lnd_new_capture/lnd2_sam6d/camera.json"
output_dir = "/vol/bitbucket/ur23/lnd_new_capture/lnd2_sam6d/output"
final_output_dir = "/vol/bitbucket/ur23/lnd_new_capture/lnd2_sam6d/final_output_sam"
# segmentation_dir = "/vol/bitbucket/ur23/lnd_new_capture/lnd1_sam6d/rle_jsons"  # Directory containing segmentation JSON files

# Ensure final output directory exists
os.makedirs(final_output_dir, exist_ok=True)

# Get list of all RGB images
rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

for rgb_path in rgb_images:
    # Get the filename without extension
    filename = os.path.basename(rgb_path).split('.')[0]
    
    # Construct corresponding depth path
    depth_path = os.path.join(depth_dir, f"{filename}.png")
    
    # Run instance segmentation
    segmentor_model = "sam"
    instance_segmentation_command = [
        "python", "run_inference_custom.py",
        "--segmentor_model", segmentor_model,
        "--output_dir", output_dir,
        "--cad_path", cad_path,
        "--rgb_path", rgb_path,
        "--depth_path", depth_path,
        "--cam_path", camera_path
    ]
    
    subprocess.run(instance_segmentation_command, cwd="Instance_Segmentation_Model")
    
    # Set the segmentation JSON and visualization image paths
    seg_json_src = os.path.join(output_dir, "sam6d_results/detection_ism.json")
    seg_vis_src = os.path.join(output_dir, "sam6d_results/vis_ism.png")
    
    # Copy and rename the segmentation JSON file to the final output directory
    seg_json_dest = os.path.join(final_output_dir, f"{filename}_detection_ism.json")
    if os.path.exists(seg_json_src):
        shutil.copy(seg_json_src, seg_json_dest)
    else:
        print(f"Segmentation JSON file not found for {filename}.")
    
    # Copy and rename the segmentation visualization image to the final output directory
    seg_vis_dest = os.path.join(final_output_dir, f"{filename}_vis_ism.png")
    if os.path.exists(seg_vis_src):
        shutil.copy(seg_vis_src, seg_vis_dest)
    else:
        print(f"Segmentation visualization image not found for {filename}.")
    
    # Run pose estimation using the generated segmentation JSON file
    if not os.path.exists(seg_json_src):
        print(f"Segmentation JSON file for {filename} not found. Skipping pose estimation for this file.")
        continue
    
    pose_estimation_command = [
        "python", "run_inference_custom.py",
        "--output_dir", output_dir,
        "--cad_path", cad_path,
        "--rgb_path", rgb_path,
        "--depth_path", depth_path,
        "--cam_path", camera_path,
        "--seg_path", seg_json_src
    ]
    
    subprocess.run(pose_estimation_command, cwd="Pose_Estimation_Model")
    
    # Copy and rename the pose estimation output files to the final output directory
    pose_json_src = os.path.join(output_dir, "sam6d_results/detection_pem.json")
    pose_json_dest = os.path.join(final_output_dir, f"{filename}_detection_pem.json")
    if os.path.exists(pose_json_src):
        shutil.copy(pose_json_src, pose_json_dest)
    else:
        print(f"Pose estimation JSON file not found for {filename}.")

    pose_vis_src = os.path.join(output_dir, "sam6d_results/vis_pem.png")
    pose_vis_dest = os.path.join(final_output_dir, f"{filename}_vis_pem.png")
    if os.path.exists(pose_vis_src):
        shutil.copy(pose_vis_src, pose_vis_dest)
    else:
        print(f"Pose estimation visualization image not found for {filename}.")
    
    print(f"Processing completed for {filename}")

print(f"All images processed. Results saved in {final_output_dir}")
