# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

def load_observation(
    example_dir: Path,
    image_filename: str,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "rgb" / image_filename), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "depth" / image_filename), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    image_filename: str,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, image_filename, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
    image_filename: str,
) -> DetectionsType:
    label = "lnd1"  # Fixed label for all detections
    mask_filename = image_filename.replace(".png", "_000000.png")  # Add _000000 suffix to match the mask filename
    mask_path = example_dir / "mask_visib" / mask_filename

    binary_mask = Image.open(mask_path).convert('L')
    binary_mask = np.array(binary_mask)

    # Find the coordinates of the non-zero regions (i.e., the mask)
    non_zero_coords = np.argwhere(binary_mask)

    detections = []
    if non_zero_coords.size > 0:
        # Get the bounding box
        y_min, x_min = non_zero_coords.min(axis=0)
        y_max, x_max = non_zero_coords.max(axis=0)

        # Bounding box coordinates
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        
        # Create ObjectData from detection
        detection = ObjectData(label=label, bbox_modal=bbox)
        detections.append(detection)

    # Pass the list of ObjectData to make_detections_from_object_data
    detections = make_detections_from_object_data(detections).cuda()
    return detections



def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
    image_filename: str,
) -> None:
    rgb, _, _ = load_observation(example_dir, image_filename, load_depth=False)
    detections = load_detections(example_dir, image_filename)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / f"{image_filename}_detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    image_filename: str,
    pose_estimates: PoseEstimatesType,
) -> None:
    output_dir = Path("mega_output") / example_dir.name
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = output_dir / f"{image_filename}_object_data.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """Converts a quaternion to a rotation matrix."""
    return R.from_quat(quaternion).as_matrix()

def project_3d_bounding_box(camera_data: CameraData, rotation: np.ndarray, translation: np.ndarray, box_size: Tuple[float, float, float]) -> np.ndarray:
    """Projects a 3D bounding box onto the image plane using the rotation matrix and translation vector."""
    
    # Configuration 1:
    half_size_x, half_size_y, half_size_z = box_size[0] / 2.0, box_size[1] / 2.0, box_size[2] / 2.0
    
    # Configuration 2:
    # half_size_x, half_size_y, half_size_z = box_size[1] / 2.0, box_size[2] / 2.0, box_size[0] / 2.0

    # Configuration 3:
    # half_size_x, half_size_y, half_size_z = box_size[2] / 2.0, box_size[0] / 2.0, box_size[1] / 2.0
    
    # Define the 3D bounding box corners in object space
    box_corners = np.array([
        [-half_size_x, -half_size_y, -half_size_z],
        [ half_size_x, -half_size_y, -half_size_z],
        [ half_size_x,  half_size_y, -half_size_z],
        [-half_size_x,  half_size_y, -half_size_z],
        [-half_size_x, -half_size_y,  half_size_z],
        [ half_size_x, -half_size_y,  half_size_z],
        [ half_size_x,  half_size_y,  half_size_z],
        [-half_size_x,  half_size_y,  half_size_z]
    ])

    # Transform the corners to camera space
    box_corners_cam = (rotation @ box_corners.T).T + translation

    # Project the corners onto the image plane
    box_corners_proj = camera_data.K @ box_corners_cam.T
    box_corners_proj /= box_corners_proj[2, :]  # Normalize by depth

    return box_corners_proj[:2, :].T  # Return only the x, y pixel coordinates



def draw_3d_bounding_box_on_image(image: np.ndarray, corners_2d: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draws the projected 3D bounding box on the image."""
    image_with_box = image.copy()
    corners_2d = corners_2d.astype(int)
    
    # Define the connections between corners to draw the edges of the box
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    for start, end in connections:
        cv2.line(image_with_box, tuple(corners_2d[start]), tuple(corners_2d[end]), color, 2)
    
    return image_with_box

# def run_inference(
#     example_dir: Path,
#     model_name: str,
# ) -> None:
#     model_info = NAMED_MODELS[model_name]
#     rgb_dir = example_dir / "rgb"
#     depth_dir = example_dir / "depth"
#     mask_dir = example_dir / "mask_visib"
#     pose_dir = example_dir / "pose"  # Directory containing ground truth poses

#     # Assuming that the filenames in rgb and depth folders are identical
#     image_filenames = sorted([img.name for img in rgb_dir.glob("*.png")])

#     object_dataset = make_object_dataset(example_dir)
#     logger.info(f"Loading model {model_name}.")
#     pose_estimator = load_named_model(model_name, object_dataset).cuda()

#     box_size = (13.563604, 6.248400, 6.400742)  # The dimensions of the 3D bounding box

#     for image_filename in image_filenames:
#         logger.info(f"Processing {image_filename}")

#         # Load observation for the current RGB and depth image
#         rgb, depth, camera_data = load_observation(example_dir, image_filename, load_depth=model_info["requires_depth"])

#         # Convert to tensor
#         observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K).cuda()

#         # Load the corresponding detection (mask) for the current image
#         detections = load_detections(example_dir, image_filename).cuda()

#         logger.info(f"Running inference for {image_filename}.")
#         output, _ = pose_estimator.run_inference_pipeline(
#             observation, detections=detections, **model_info["inference_parameters"]
#         )

#         # Save predictions for the current image
#         save_predictions(example_dir, image_filename, output)

#         rgb_image = np.array(Image.open(rgb_dir / image_filename))

#         # Visualize the predicted pose (in green)
#         for i, pose in enumerate(output.poses):
#             pose_matrix = pose.cpu().numpy()  # Get the pose as a numpy array
#             rotation_matrix = pose_matrix[:3, :3]  # Extract the rotation part
#             translation = pose_matrix[:3, 3]       # Extract the translation part

#             # Use the `camera_data` obtained earlier when loading the observation
#             corners_2d_pred = project_3d_bounding_box(camera_data, rotation_matrix, translation, box_size)
#             rgb_image_with_box = draw_3d_bounding_box_on_image(rgb_image, corners_2d_pred, color=(0, 255, 0))  # Green

#         # Load the ground truth pose (3x4 matrix) from the .npy file
#         gt_pose_path = pose_dir / image_filename.replace('.png', '.npy')
#         if gt_pose_path.exists():
#             gt_pose = np.load(gt_pose_path)
#             gt_rotation_matrix = gt_pose[:3, :3]
#             gt_translation = gt_pose[:3, 3]

#             # Project and visualize the ground truth pose (in red)
#             corners_2d_gt = project_3d_bounding_box(camera_data, gt_rotation_matrix, gt_translation, box_size)
#             rgb_image_with_box = draw_3d_bounding_box_on_image(rgb_image_with_box, corners_2d_gt, color=(255, 0, 0))  # Red

#         # Save the image with both predicted and ground truth 3D bounding boxes
#         output_dir = Path("mega_output") / example_dir.name
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_image_path = output_dir / f"{image_filename}_with_box.png"
#         Image.fromarray(rgb_image_with_box).save(output_image_path)
#         logger.info(f"Wrote image with 3D bounding box: {output_image_path}")

#     return

def load_ply_mesh(mesh_path: Path) -> np.ndarray:
    """Loads a PLY file and returns the 3D points as a numpy array."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    return vertices

def project_mesh_to_image(camera_data: CameraData, vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Projects the 3D vertices of a mesh into the 2D image plane."""
    # Transform the mesh vertices using the pose (rotation and translation)
    transformed_vertices = (rotation @ vertices.T).T + translation

    # Project the vertices onto the image plane
    projected_vertices = camera_data.K @ transformed_vertices.T
    projected_vertices /= projected_vertices[2, :]  # Normalize by depth

    return projected_vertices[:2, :].T  # Return only the x, y pixel coordinates

def draw_projected_mesh_on_image(image: np.ndarray, projected_vertices: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draws the projected 3D mesh on the image."""
    image_with_mesh = image.copy()
    projected_vertices = projected_vertices.astype(int)

    for vertex in projected_vertices:
        cv2.circle(image_with_mesh, tuple(vertex), radius=1, color=color, thickness=-1)  # Draw as small dots

    return image_with_mesh

def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:
    model_info = NAMED_MODELS[model_name]
    rgb_dir = example_dir / "rgb"
    depth_dir = example_dir / "depth"
    mask_dir = example_dir / "mask_visib"
    pose_dir = example_dir / "pose"  # Directory containing ground truth poses
    mesh_path = example_dir / "meshes" / "lnd1" / "obj_000001.ply"  # Path to the object's .ply file

    # Create a directory to save the predicted 3x4 matrices
    prediction_output_dir = example_dir / "predicted_poses"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    # Load the mesh vertices
    mesh_vertices = load_ply_mesh(mesh_path)

    # Assuming that the filenames in rgb and depth folders are identical
    image_filenames = sorted([img.name for img in rgb_dir.glob("*.png")])

    object_dataset = make_object_dataset(example_dir)
    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    for image_filename in image_filenames:
        logger.info(f"Processing {image_filename}")

        # Load observation for the current RGB and depth image
        rgb, depth, camera_data = load_observation(example_dir, image_filename, load_depth=model_info["requires_depth"])

        # Convert to tensor
        observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K).cuda()

        # Load the corresponding detection (mask) for the current image
        detections = load_detections(example_dir, image_filename).cuda()
        # make_detections_visualization(example_dir,image_filename)

        logger.info(f"Running inference for {image_filename}.")
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )

        # Save predictions for the current image
        # save_predictions(example_dir, image_filename, output)
     
        rgb_image = np.array(Image.open(rgb_dir / image_filename))

        # Visualize the predicted pose (in green)
        for i, pose in enumerate(output.poses):
            pose_matrix = pose.cpu().numpy()  # Get the pose as a numpy array
            rotation_matrix = pose_matrix[:3, :3]  # Extract the rotation part
            translation = pose_matrix[:3, 3] * 1000  # Extract the translation part (converted to mm)

            # Combine rotation and translation into a 3x4 matrix
            transformation_matrix = np.hstack((rotation_matrix, translation.reshape(3, 1)))

            # print(f"prediction R: \n {rotation_matrix} \n and T \n {translation}")

            # Save the 3x4 matrix to a txt file
            txt_output_path = prediction_output_dir / f"{image_filename}.txt"
            np.savetxt(txt_output_path, transformation_matrix, fmt="%.6f")

            # Project the mesh using the predicted pose
            projected_vertices_pred = project_mesh_to_image(camera_data, mesh_vertices, rotation_matrix, translation)
            rgb_image_with_mesh = draw_projected_mesh_on_image(rgb_image, projected_vertices_pred, color=(0, 255, 0))  # Green

        # Load the ground truth pose (3x4 matrix) from the .npy file
        # gt_pose_path = pose_dir / image_filename.replace('.png', '.npy')
        # if gt_pose_path.exists():
        #     gt_pose = np.load(gt_pose_path)
        #     gt_rotation_matrix = gt_pose[:3, :3]
        #     gt_translation = gt_pose[:3, 3]
            # print(f"gt R: \n{gt_rotation_matrix} \n and T \n{gt_translation}")

            # Project and visualize the mesh using the ground truth pose (in red)
            # projected_vertices_gt = project_mesh_to_image(camera_data, mesh_vertices, gt_rotation_matrix, gt_translation)
            # rgb_image_with_mesh = draw_projected_mesh_on_image(rgb_image_with_mesh, projected_vertices_gt, color=(255, 0, 0))  # Red

        # Save the image with both predicted and ground truth projected meshes
        # output_dir = Path("mega_output") / example_dir.name
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_image_path = output_dir / f"{image_filename}_with_mesh.png"
        # Image.fromarray(rgb_image_with_mesh).save(output_image_path)
        # logger.info(f"Wrote image with 3D mesh projection: {output_image_path}")

    return



if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name", type=str, help="Name of the example directory under LOCAL_DATA_DIR/examples")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name


    if args.run_inference:
        run_inference(example_dir, args.model)

