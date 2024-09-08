import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Calibration parameters
K1 = np.array([[816.774523, 0.000000, 559.211371],
               [0.000000, 822.016911, 309.985941],
               [0.000000, 0.000000, 1.000000]])
D1 = np.array([-0.347547, 0.463333, 0.000000, 0.000000])

K2 = np.array([[826.686752, 0.000000, 475.150191],
               [0.000000, 831.101411, 305.182810],
               [0.000000, 0.000000, 1.000000]])
D2 = np.array([-0.381301, 0.570584, 0.000000, 0.000000])

R = np.array([[0.999965, 0.005475, 0.006279],
              [-0.005473, 0.999985, -0.000411],
              [-0.006281, 0.000377, 0.999980]])
T = np.array([5.478062, 0.030863, 0.016790])

# Load images
left_img_path = '/home/utsav/IProject/data/captured/lnd1/left/210.png'
right_img_path = '/home/utsav/IProject/data/captured/lnd1/right/210.png'

left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
left_img_color = cv2.imread(left_img_path)
right_img_color = cv2.imread(right_img_path)

# Image size
image_size = (left_img.shape[1], left_img.shape[0])

# Rectify function
def rectify(m1, d1, m2, d2, width, height, r, t):
    R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(cameraMatrix1=m1,
                                                        distCoeffs1=d1,
                                                        cameraMatrix2=m2,
                                                        distCoeffs2=d2,
                                                        imageSize=(width, height),
                                                        R=r,
                                                        T=t,
                                                        flags=0,
                                                        alpha=0.0)
    map1_x, map1_y = cv2.initUndistortRectifyMap(cameraMatrix=m1,
                                                 distCoeffs=d1,
                                                 R=R1,
                                                 newCameraMatrix=P1,
                                                 size=(width, height),
                                                 m1type=cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(cameraMatrix=m2,
                                                 distCoeffs=d2,
                                                 R=R2,
                                                 newCameraMatrix=P2,
                                                 size=(width, height),
                                                 m1type=cv2.CV_32FC1)
    return map1_x, map1_y, map2_x, map2_y, Q, P1, P2

# Get rectification maps and parameters
map1_x, map1_y, map2_x, map2_y, Q, P1, P2 = rectify(K1, D1, K2, D2, image_size[0], image_size[1], R, T)

# Apply the rectification
rect_left_img = cv2.remap(left_img, map1_x, map1_y, cv2.INTER_LINEAR)
rect_right_img = cv2.remap(right_img, map2_x, map2_y, cv2.INTER_LINEAR)

# Apply the rectification to color images
rect_left_img_color = cv2.remap(left_img_color, map1_x, map1_y, cv2.INTER_LINEAR)
rect_right_img_color = cv2.remap(right_img_color, map2_x, map2_y, cv2.INTER_LINEAR)


# Save rectified intrinsics
rectified_intrinsic1 = P1[:3, :3]
rectified_intrinsic2 = P2[:3, :3]

# Save rectified images
cv2.imwrite('rectified_left_image.png', rect_left_img_color)
cv2.imwrite('rectified_right_image.png', rect_right_img_color)

# Display rectified intrinsics
print('Rectified Left Camera Intrinsics:\n', rectified_intrinsic1)
print('Rectified Right Camera Intrinsics:\n', rectified_intrinsic2)

# Visualize the original and rectified images

# Function to create an anaglyph
def create_anaglyph(img1, img2):
    anaglyph = np.zeros_like(cv2.merge([img1, img1, img1]))
    anaglyph[:, :, 0] = img1  # Red channel
    anaglyph[:, :, 1] = img2  # Green channel
    anaglyph[:, :, 2] = img2  # Blue channel
    return anaglyph

# Anaglyph images
anaglyph_left = create_anaglyph(left_img, rect_left_img)
anaglyph_right = create_anaglyph(right_img, rect_right_img)

# Display the images
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].imshow(left_img, cmap='gray')
axs[0, 0].set_title('Original Left Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(right_img, cmap='gray')
axs[0, 1].set_title('Original Right Image')
axs[0, 1].axis('off')

axs[1, 0].imshow(rect_left_img, cmap='gray')
axs[1, 0].set_title('Rectified Left Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(rect_right_img, cmap='gray')
axs[1, 1].set_title('Rectified Right Image')
axs[1, 1].axis('off')

plt.show()

# Display anaglyph images
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(anaglyph_left)
plt.title('Anaglyph: Original vs Rectified Left')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(anaglyph_right)
plt.title('Anaglyph: Original vs Rectified Right')
plt.axis('off')

plt.show()

# Compute disparity map
disparity_range = 16*16
stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=disparity_range,
                               blockSize=1,
                               uniquenessRatio=15,
                               speckleWindowSize=50,
                               speckleRange=2,
                               disp12MaxDiff=1,
                               P1=8*3*3**2,
                               P2=32*3*3**2)

disparity_map = stereo.compute(rect_left_img, rect_right_img).astype(np.float32) / 16.0

# Display disparity map
plt.figure(figsize=(10, 5))
plt.imshow(disparity_map, 'gray')
plt.colorbar()
plt.title('Disparity Map')
plt.show()

# Custom depth calculation
def compute_depth_map(disparity_map, baseline, focal_length, cx1, cx2):
    depth_map = np.zeros(disparity_map.shape, dtype=np.float32)
    valid_disparity_mask = disparity_map > 0
    depth_map[valid_disparity_mask] = (focal_length * baseline) / (disparity_map[valid_disparity_mask] + cx2 - cx1)
    return depth_map, valid_disparity_mask

# Extract focal length and principal points from intrinsic matrices
focal_length = rectified_intrinsic1[0, 0]
cx1 = rectified_intrinsic1[0, 2]
cx2 = rectified_intrinsic2[0, 2]
baseline = np.linalg.norm(T)

# Compute depth map
depth_map, valid_disparity_mask = compute_depth_map(disparity_map, baseline, focal_length, cx1, cx2)
print(depth_map[0][0])
# Debug print statements
print(f"focal_length: {focal_length}")
print(f"baseline: {baseline}")
print(f"Valid disparity values: {np.sum(valid_disparity_mask)}")

# Visualization of the disparity map (optional)
plt.figure(figsize=(10, 5))
plt.imshow(disparity_map, cmap='jet')
plt.colorbar()
plt.title('Disparity Map')
plt.show()

# Get image dimensions
h, w = rect_left_img.shape

# Thresholds
black_threshold = 0
depth_threshold = 10 # Minimum depth value to consider

# Create point cloud
points = []
colors = []

for v in range(h):
    for u in range(w):
        if not valid_disparity_mask[v, u]:  # Skip invalid depth values
            continue
        Z = depth_map[v, u]
        # if Z <= depth_threshold or not np.isfinite(Z):  # Skip invalid depth values and too close points
        #     continue
        X = (u - cx1) * Z / focal_length
        Y = -(v - cx2) * Z / focal_length  # Using cy from RECT_M1
        color = left_img_color[v, u]
        if np.all(color < black_threshold):  # Skip points with black color below the threshold
            continue
        points.append([X, Y, Z])
        colors.append(color / 255.0)

points = np.array(points)
colors = np.array(colors)

# Check if points and colors arrays are not empty
if points.size == 0 or colors.size == 0:
    raise ValueError("No valid points found in the point cloud")

# Ensure points and colors arrays have the correct shape
if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError(f"Expected points shape (N, 3), but got {points.shape}")
if colors.ndim != 2 or colors.shape[1] != 3:
    raise ValueError(f"Expected colors shape (N, 3), but got {colors.shape}")

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Apply scaling
scaling_factor = 1.0
pcd.scale(scaling_factor, center=pcd.get_center())

# Save the point cloud to a file
o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
