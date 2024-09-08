import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import open3d as o3d

# Camera parameters
RECT_M1 = np.array([
    [801.57991404, 0, 583.56089783],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1.0]
])

R = np.array([
    [0.9999, -0.0054, -0.0086],
    [0.0054, 1.0000, -0.0005],
    [0.0086, 0.0004, 1.0000]
])
T = np.array([5.5160, 0.0481, -0.0733])

def compute_Q_matrix(K, R, T):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    Tx = T[0]
    
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1/Tx, (cx - cx) / Tx]
    ])
    
    return Q

def compute_disparity(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    window_size = 3
    min_disp = 0
    num_disp = 16*16
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=1,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    
    return disparity

def disparity_to_depth(disparity, Q):
    points = cv2.reprojectImageTo3D(disparity, Q)
    depth = points[:,:,2]
    return depth, points

def save_point_cloud(points, colors, filename):
    # Remove invalid points
    mask = points[:,:,2] > points[:,:,2].min()
    valid_points = points[mask]
    valid_colors = colors[mask]
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(valid_colors.reshape(-1, 3) / 255.0)  # Normalize colors to [0, 1]
    
    # Save point cloud
    o3d.io.write_point_cloud(filename, pcd)

def visualize_disparity(disparity, filename):
    plt.figure(figsize=(10, 7))
    plt.imshow(disparity, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title('Disparity Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main process
if __name__ == "__main__":
    imgL = cv2.imread('/home/utsav/IProject/data/lnd1/filtered_left_rect_images/1065.png')
    imgR = cv2.imread('/home/utsav/IProject/data/lnd1/rect_right/1065.png')
    
    Q = compute_Q_matrix(RECT_M1, R, T)
    
    disparity = compute_disparity(imgL, imgR)
    
    # Visualize and save disparity map
    visualize_disparity(disparity, 'disparity_map.png')
    
    depth, points = disparity_to_depth(disparity, Q)
    
    # Save actual depth map as 32-bit float TIFF
    tifffile.imwrite('depth_map.tiff', depth.astype(np.float32))
    
    # Save point cloud using Open3D
    save_point_cloud(points, imgL, 'point_cloud.ply')
    
    print("Disparity visualization, actual depth map, and point cloud saved successfully!")






