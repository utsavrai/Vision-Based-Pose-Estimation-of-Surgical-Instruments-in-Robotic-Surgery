import pyvista as pv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def apply_transformation_to_mesh(mesh, transformation_matrix):
    transform = pv.vtk.vtkTransform()
    transform.SetMatrix(transformation_matrix.flatten())
    transform_filter = pv.vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return pv.wrap(transform_filter.GetOutput())

def render_mesh(mesh, camera_intrinsics, width, height):
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.add_mesh(mesh, color='red')
    plotter.camera.position = (0, 0, 1)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.viewup = (0, -1, 0)
    plotter.camera.view_angle = 2 * np.degrees(np.arctan2(height / 2, camera_intrinsics[1, 1]))
    plotter.set_background('black')
    image = plotter.screenshot(transparent_background=True)
    plotter.close()
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

def overlay_images(background_img, overlay_img, alpha=0.5):
    overlay_gray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)

    mask_3ch = cv2.merge([mask, mask, mask])

    foreground = overlay_img.astype(float)
    background = background_img.astype(float)

    alpha_mask = mask_3ch.astype(float) / 255
    alpha_mask = alpha * alpha_mask

    blended = cv2.addWeighted(background, 1.0, foreground, alpha, 0, dtype=cv2.CV_32F)
    result = np.clip(blended, 0, 255).astype(np.uint8)

    result = np.where(mask_3ch > 0, result, background_img)

    return result

def add_rotation_around_z(transformation_matrix, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_z = np.array([
        [cos_angle, -sin_angle, 0, 0],
        [sin_angle, cos_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(transformation_matrix, rotation_z)

def project_point(point, camera_intrinsics):
    projected_point = np.dot(camera_intrinsics, point[:3])
    projected_point /= projected_point[2]
    return projected_point[:2].astype(int)

def draw_axes(image, pose, camera_intrinsics, length=5):
    origin = pose[:3, 3]
    x_axis = origin + pose[:3, 0] * length
    y_axis = origin + pose[:3, 1] * length
    z_axis = origin + pose[:3, 2] * length

    origin_2d = project_point(np.append(origin, 1), camera_intrinsics)
    x_axis_2d = project_point(np.append(x_axis, 1), camera_intrinsics)
    y_axis_2d = project_point(np.append(y_axis, 1), camera_intrinsics)
    z_axis_2d = project_point(np.append(z_axis, 1), camera_intrinsics)

    cv2.line(image, tuple(origin_2d), tuple(x_axis_2d), (0, 0, 255), 2)  # X-axis in red
    cv2.line(image, tuple(origin_2d), tuple(y_axis_2d), (0, 255, 0), 2)  # Y-axis in green
    cv2.line(image, tuple(origin_2d), tuple(z_axis_2d), (255, 0, 0), 2)  # Z-axis in blue

def calculate_axis_alignment(transformation_matrix):
    rotation_matrix = transformation_matrix[:3, :3]
    x_axis_transformed = rotation_matrix[:, 0]
    y_axis_transformed = rotation_matrix[:, 1]
    z_axis_transformed = rotation_matrix[:, 2]

    x_axis_world = np.array([1, 0, 0])
    y_axis_world = np.array([0, 1, 0])
    z_axis_world = np.array([0, 0, 1])

    angle_x = np.arccos(np.clip(np.dot(x_axis_transformed, x_axis_world), -1.0, 1.0))
    angle_y = np.arccos(np.clip(np.dot(y_axis_transformed, y_axis_world), -1.0, 1.0))
    angle_z = np.arccos(np.clip(np.dot(z_axis_transformed, z_axis_world), -1.0, 1.0))

    angle_x_deg = np.degrees(angle_x)
    angle_y_deg = np.degrees(angle_y)
    angle_z_deg = np.degrees(angle_z)

    print(f"Angle between model's X-axis and world X-axis: {angle_x_deg} degrees")
    print(f"Angle between model's Y-axis and world Y-axis: {angle_y_deg} degrees")
    print(f"Angle between model's Z-axis and world Z-axis: {angle_z_deg} degrees")

def visualize_cad_model_on_image(image_path, cad_model_path, transformation_matrix, camera_intrinsics, z_rotation_angle):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    print(f"Image size: {width}x{height}")
    print(f"Camera intrinsics:\n{camera_intrinsics}")

    mesh = pv.read(cad_model_path)

    transformation_matrix = add_rotation_around_z(transformation_matrix, z_rotation_angle)

    transformed_mesh = apply_transformation_to_mesh(mesh, transformation_matrix)

    rendered_image = render_mesh(transformed_mesh, camera_intrinsics, width, height)

    result_image = overlay_images(image, rendered_image, alpha=0.5)

    draw_axes(result_image, transformation_matrix, camera_intrinsics)

    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Calculate and print the alignment angles
    calculate_axis_alignment(transformation_matrix)

def verify_transformation_with_pose(image_path, cad_model_path, transformation_matrix, camera_intrinsics):
    print("Visualizing with original pose")
    visualize_cad_model_on_image(image_path, cad_model_path, transformation_matrix.copy(), camera_intrinsics, 0)

def debug_transformation_matrix(transformation_matrix):
    print(f"Transformation matrix:\n{transformation_matrix}")
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]
    print(f"Rotation matrix:\n{rotation_matrix}")
    print(f"Translation vector:\n{translation_vector}")

    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    print(f"Euler angles (degrees): {euler_angles}")

    if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-6):
        print("Warning: The rotation matrix is not orthogonal.")

    det = np.linalg.det(rotation_matrix)
    print(f"Determinant of the rotation matrix: {det}")
    if not np.isclose(det, 1):
        print("Warning: The rotation matrix determinant is not 1, which suggests scaling or reflection.")

# Path to the RGB image
image_path = '/home/utsav/IProject/data/captured/lnd1/rect_left/1300.png'

# Path to the CAD model
cad_model_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/joint.stl'

# Path to the transformation matrix npy file
transformation_matrix_path = '/home/utsav/IProject/data/captured/lnd1/output_left/1300.npy'

# Define camera intrinsics for the left rectified camera
camera_intrinsics = np.array([
    [801.57991404, 0, 583.56089783],
    [0, 801.57991404, 309.78999329],
    [0, 0, 1]
])

# Load the transformation matrix from the npy file
transformation_matrix = np.load(transformation_matrix_path)

# Ensure the transformation matrix is 4x4
if transformation_matrix.shape == (3, 4):
    transformation_matrix = np.vstack([transformation_matrix, [0, 0, 0, 1]])

# Debug: Print transformation matrix
debug_transformation_matrix(transformation_matrix)

# Verify transformation with different rotation angles around Z-axis
verify_transformation_with_pose(image_path, cad_model_path, transformation_matrix, camera_intrinsics)
