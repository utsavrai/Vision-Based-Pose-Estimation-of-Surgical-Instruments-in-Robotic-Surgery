# from unidepth.models import UniDepthV2
# import open3d as o3d
# import numpy as np
# from PIL import Image
# import torch
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Set up the model and device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)

# # Load intrinsic parameters
# intrinsics_path = "UniDepth/assets/synapse/intrinsic.npy"
# intrinsics = np.load(intrinsics_path).astype(np.float32)
# if intrinsics.ndim < 2:
#     intrinsics = np.expand_dims(intrinsics, 0)  # Ensure intrinsics are two-dimensional
# intrinsics = torch.from_numpy(intrinsics).to(device)

# # Scale factor for depth and point cloud adjustment
# scale_factor = 0.3  # Adjust this value to calibrate depth and point cloud scale

# # Directories for input images and output depth maps and point clouds
# input_dir = Path("/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/image")  # Update with your input directory
# output_depth_dir = Path("UniDepth/assets/synapse/depth")  # Update with your output directory
# # output_pcd_dir = Path("UniDepth/assets/synapse/pcd")  # Update with your output point cloud directory
# output_depth_dir.mkdir(parents=True, exist_ok=True)  # Create output directories if they don't exist
# # output_pcd_dir.mkdir(parents=True, exist_ok=True)

# # Loop through each image in the input directory
# for image_path in input_dir.glob("*.png"):  # Assumes PNG files, change the pattern if needed
#     rgb_image = Image.open(image_path).convert('RGB')
#     rgb = np.array(rgb_image)
#     rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

#     # Run inference
#     predictions = model.infer(rgb_tensor, intrinsics)
#     depth = predictions['depth'].squeeze().cpu().numpy() * scale_factor
#     xyz = predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy() * scale_factor

#     # Save the depth map
#     output_depth_path = output_depth_dir / image_path.name
#     plt.imsave(output_depth_path, depth, cmap='gray')

#     # Create and save the point cloud
#     # point_cloud = o3d.geometry.PointCloud()
#     # point_cloud.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
#     # point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
#     # output_pcd_path = output_pcd_dir / image_path.with_suffix('.ply').name
#     # o3d.io.write_point_cloud(str(output_pcd_path), point_cloud, write_ascii=True)

#     print(f"Processed {image_path.name}: Depth map saved.")



from unidepth.models import UniDepthV2
import open3d as o3d
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# Set up the model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)

# Load intrinsic parameters
intrinsics_path = "/home/utsav/IProject/UniDepth/assets/synapse/intrinsic.npy"
intrinsics = np.load(intrinsics_path).astype(np.float32)
if intrinsics.ndim < 2:
    intrinsics = np.expand_dims(intrinsics, 0)
intrinsics = torch.from_numpy(intrinsics).to(device)

# Scale factor for depth and point cloud adjustment
# scale_factor = 0.3

# Directories for input images and output depth maps
input_dir = Path("/home/utsav/rgb")
output_depth_dir = Path("/home/utsav/unidepth")
output_depth_dir.mkdir(parents=True, exist_ok=True)

# Loop through each image in the input directory
for image_path in input_dir.glob("*.png"):
    rgb_image = Image.open(image_path).convert('RGB')
    rgb = np.array(rgb_image)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


    # Run inference
    predictions = model.infer(rgb_tensor, intrinsics)
    depth = predictions['depth'].squeeze().cpu().numpy()
    depth_in_mm = depth*1000
    depth_image = Image.fromarray(depth_in_mm.astype(np.int32), mode='I')
    output_depth_path = output_depth_dir / image_path.name
    output_depth_path = output_depth_path
    depth_image.save(output_depth_path)
    print(f"Processed {image_path.name}: Depth map saved.")