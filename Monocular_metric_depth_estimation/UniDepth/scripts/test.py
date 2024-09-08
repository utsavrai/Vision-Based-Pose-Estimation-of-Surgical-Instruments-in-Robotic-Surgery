
# from unidepth.models import UniDepthV2
# import open3d as o3d
# import numpy as np
# from PIL import Image
# import torch
# import matplotlib.pyplot as plt

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

# # Load and prepare the RGB image
# image_path = "/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/image/1.png"  # Update the image path
# rgb_image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
# rgb = np.array(rgb_image)
# rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor and adjust dimensions

# # Run inference
# predictions = model.infer(rgb_tensor, intrinsics)
# xyz = predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()  # Reshape from [1, 3, H, W] to [H, W, 3]

# # Create Open3D point cloud
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))  # Reshape to [N, 3]
# point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)  # Normalize RGB values

# # Save point cloud to a PLY file
# output_ply_path = "output_point_cloud.ply"
# o3d.io.write_point_cloud(output_ply_path, point_cloud, write_ascii=True)

# # Visualization of RGB and Depth images
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(rgb_image)
# plt.axis('off')
# plt.title('RGB Image')

# plt.subplot(1, 2, 2)
# depth = predictions['depth'].squeeze().cpu().numpy()
# plt.imshow(depth, cmap='gray')
# plt.title('Depth Map')
# plt.axis('off')
# plt.show()

# print(f"Point cloud and RGB-Depth images processed and displayed. Point cloud saved to {output_ply_path}")


from unidepth.models import UniDepthV2
import open3d as o3d
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Set up the model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)

# Load intrinsic parameters
intrinsics_path = "UniDepth/assets/synapse/intrinsic.npy"
intrinsics = np.load(intrinsics_path).astype(np.float32)
if intrinsics.ndim < 2:
    intrinsics = np.expand_dims(intrinsics, 0)  # Ensure intrinsics are two-dimensional
intrinsics = torch.from_numpy(intrinsics).to(device)

# Load and prepare the RGB image
image_path = "FoundationPose/demo_data/LND/test/1.png"
rgb_image = Image.open(image_path).convert('RGB')
rgb = np.array(rgb_image)
rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

# Run inference
predictions = model.infer(rgb_tensor, intrinsics)

# Scale factor for depth and point cloud adjustment
scale_factor = 0.3  # Adjust this value to calibrate depth scale

# Adjusting depth values
depth = predictions['depth'].squeeze().cpu().numpy() * scale_factor
print(depth.shape)

# Adjusting points based on new depth
xyz = predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy() * scale_factor

# Create Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)

# Save point cloud to a PLY file
output_ply_path = "output_point_cloud.ply"
o3d.io.write_point_cloud(output_ply_path, point_cloud, write_ascii=True)

# Visualization of RGB and Depth images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.axis('off')
plt.title('RGB Image')

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap='gray')
plt.colorbar()  # Added colorbar to visualize depth scale
plt.title('Depth Map')
plt.axis('off')
plt.show()

print(f"Point cloud and RGB-Depth images processed and displayed. Point cloud saved to {output_ply_path}")
