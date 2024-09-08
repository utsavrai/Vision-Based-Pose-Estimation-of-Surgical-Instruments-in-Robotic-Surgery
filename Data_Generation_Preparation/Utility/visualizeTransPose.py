import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import trimesh

def show_3dmodel(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d, is_show=False):
    imgpts, jac = cv2.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    frame_centre = tuple(imgpts[0].ravel())
    radius = 1
    thickness = 1

    for pt in imgpts:
        pt_x = int(pt[0, 0])
        pt_y = int(pt[0, 1])
        try:
            cv2.circle(im, (pt_x, pt_y), radius, (66, 50, 255), thickness)
        except:
            continue
    if is_show:
        cv2.imshow("image", im)
        cv2.waitKey(0)
    return im

def pack_homo(R, T):
    T = T.reshape(3, -1)
    homo = np.concatenate((R, T), axis=1)
    ones = np.array([0, 0, 0, 1]).reshape(1, 4)
    homo = np.concatenate((homo, ones), axis=0)
    return homo

def unpack_homo(homo):
    R = homo[:3, :3]
    t = homo[:3, 3]
    return R, t

def angle_to_rot(angle_X, angle_Y, angle_Z):
    r = R.from_euler('xyz', [angle_X, angle_Y, angle_Z], degrees=True)
    return r.as_matrix()

def load_obj_vertices(obj_file_path):
    mesh = trimesh.load_mesh(obj_file_path)
    return mesh.vertices

def visualize_cad_model_on_image(image_path, cad_model_path, original_transformation_matrix_path, translation, rotation, camera_intrinsics):
    image = cv2.imread(image_path)
    pts_3d = load_obj_vertices(cad_model_path)

    height, width, _ = image.shape

    print(f"Image size: {width}x{height}")
    print(f"Camera intrinsics:\n{camera_intrinsics}")

    # Load the original transformation matrix
    original_transformation_matrix = np.load(original_transformation_matrix_path)

    # Ensure the transformation matrix is 4x4
    if original_transformation_matrix.shape == (3, 4):
        original_transformation_matrix = np.vstack([original_transformation_matrix, [0, 0, 0, 1]])

    # Apply the user-specified transformation
    user_translation = np.array(translation)
    user_rotation = angle_to_rot(*rotation)

    user_transformation_matrix = pack_homo(user_rotation, user_translation)

    # Combine the user transformation with the existing transformation
    combined_transformation_matrix = np.dot(user_transformation_matrix, original_transformation_matrix)

    # Extract rotation and translation vectors
    r, t = unpack_homo(combined_transformation_matrix)

    # Visualize the transformed model
    dist_coeff = np.zeros((4, 1))  # Assuming no lens distortion
    result_image = show_3dmodel(image, r, t, camera_intrinsics, dist_coeff, pts_3d, False)

    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example inputs
original_transformation_matrix_path = '/home/utsav/IProject/data/captured/lnd1/output_left/1300.npy'  # Path to the transformation matrix npy file
image_path = '/home/utsav/IProject/data/captured/lnd1/rect_left/1300.png'  # Path to the RGB image
cad_model_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/joint.stl'  # Path to the CAD model in .obj format

# Define translation and rotation for the user-specified transformation
translation = [-17.16, -5.86, 0.33]  # Translation in millimeters
rotation = [0.0, -2.4, -93.60]  # Rotation in degrees

# Define camera intrinsics for the left rectified camera
camera_intrinsics = np.array([
    [801.57991404, 0, 583.56089783],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1]
])

# Apply the transformation and visualize
visualize_cad_model_on_image(image_path, cad_model_path, original_transformation_matrix_path, translation, rotation, camera_intrinsics)
