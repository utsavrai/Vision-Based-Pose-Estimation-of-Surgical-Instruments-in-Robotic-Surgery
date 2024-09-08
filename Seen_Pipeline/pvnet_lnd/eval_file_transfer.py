import os
import shutil

# Define paths
eval_pose_dir = '../data/dataset/lnd1/eval_pose'
rgb_dir = '../data/dataset/lnd1/train/000001/rgb'
mask_visib_dir = '../data/dataset/lnd1/train/000001/mask_visib'
depth_dir = '../data/dataset/lnd1/train/000001/depth'
dest_dir = 'data/custom'

# Destination folders
pose_dest_dir = os.path.join(dest_dir, 'pose')
rgb_dest_dir = os.path.join(dest_dir, 'rgb')
mask_dest_dir = os.path.join(dest_dir, 'mask')
depth_dest_dir = os.path.join(dest_dir, 'depth')

# Create destination directories if they don't exist
os.makedirs(pose_dest_dir, exist_ok=True)
os.makedirs(rgb_dest_dir, exist_ok=True)
os.makedirs(mask_dest_dir, exist_ok=True)
os.makedirs(depth_dest_dir, exist_ok=True)

# List all .npy files in eval_pose directory
npy_files = sorted([f for f in os.listdir(eval_pose_dir) if f.endswith('.npy')])

# Process files
for idx, npy_file in enumerate(npy_files):
    base_name = os.path.splitext(npy_file)[0]  # Get the base name without extension
    rgb_file = f'{base_name}.png'
    depth_file = f'{base_name}.png'
    mask_file = f'{base_name}_000000.png'
    
    # Determine new file names
    new_file_name = f'{idx + 1}.npy'
    new_rgb_name = f'{idx + 1}.png'
    new_depth_name = f'{idx + 1}.png'
    new_mask_name = f'{idx + 1}.png'
    
    # Copy and rename files
    shutil.copy(os.path.join(eval_pose_dir, npy_file), os.path.join(pose_dest_dir, new_file_name))
    shutil.copy(os.path.join(rgb_dir, rgb_file), os.path.join(rgb_dest_dir, new_rgb_name))
    shutil.copy(os.path.join(depth_dir, depth_file), os.path.join(depth_dest_dir, new_depth_name))
    shutil.copy(os.path.join(mask_visib_dir, mask_file), os.path.join(mask_dest_dir, new_mask_name))

print("Files copied and renamed successfully.")
