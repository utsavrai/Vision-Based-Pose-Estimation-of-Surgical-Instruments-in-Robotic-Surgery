# Data Preparation Pipeline

This repository contains a complete pipeline for calibrating stereo cameras, rectifying images, computing transformations between a marker and a tool, generating depth maps, converting data into BOP format, and optionally generating synthetic datasets for object pose estimation.

## Steps

### Step 1: Camera Calibration

**Goal**: Calibrate the stereo camera system using uncalibrated left and right images.

- **Input**: Stereo images located in the `Calibration` folder.
- **Tool**: MATLAB Stereo Camera Calibration App.

#### MATLAB Stereo Calibration Steps:

1. Open MATLAB and navigate to the **Stereo Camera Calibrator** app.
2. Load the left and right images from the `Calibration` folder.
3. Configure the calibration pattern (e.g., chessboard) in the app.
4. Perform stereo calibration to compute the intrinsic and extrinsic camera parameters.
5. Export the computed camera parameters for use in later steps.

### Step 2: Stereo Rectification

**Goal**: Use the camera parameters from MATLAB calibration to rectify stereo images.

- **Input**: Stereo images and calibration parameters saved in `config.yaml`.
- **Tool**: OpenCV via `rectification.py`.

#### Steps:

1. Define the `data_path` for the stereo images and folders for `rect_left` and `rect_right`.
2. Make sure the `config.yaml` contains the camera parameters computed from MATLAB stereo calibration.
3. Run the rectification script:

   ```bash
   python Rectification/rectification.py
   ```

This script will rectify the stereo images using the provided parameters and save the output in the specified directories.

### Step 3: Compute Transform Between Marker and Tool

**Goal**: Calculate the transformation between a marker and tool using rectified stereo images.

- **Input**: Rectified images, `config.yaml` with calibration parameters.
- **Tool**: `3d_adjust` via `main.py` and `pose_estimation.py`.

#### Steps:

1. Use `main.py` to load the `config.yaml` and the root path to the rectified image data.
   
   ```bash
   python 3d_adjust/main.py --root_path=<path_to_image_data>
   ```

2. The script will guide you in adjusting the transformation between the marker and tool. Once you finalize the transformation, save it to `config.yaml` under `pose_rot` (rotation) and `pose_trans` (translation).

3. Adjust the `model_folder` path in `pose_estimation.py`

### Step 4: Save Ground Truth Pose and Full Mask

**Goal**: Use the transformation data to generate ground truth poses and full masks.

- **Input**: Rectified images and transformation details stored in `config.yaml`.
- **Tool**: `data_generation_stereo.py`.

#### Steps:
1. Adjust the following path in this script `root_path` containing the rectified images and config.yaml file
2. Run the `data_generation_stereo.py` script to generate the ground truth pose and full mask.

   ```bash
   python Generator/data_generation_stereo.py 
   ```

This script will use the transformation to compute and save the ground truth poses for each image, as well as generate masks for the objects in the images.

### Step 5: Disparity and Depth Computation

**Goal**: Compute disparity and depth from stereo images.

- **Input**: Rectified stereo images (left and right).
- **Tools**: OpenCV (`depthmap.py`), RAFT-Stereo (`run_multiple.sh`), `generate_pointcloud.py`.

#### Steps:

1. **Disparity and Depth Using OpenCV**:
   - Use `depthmap.py` to compute the depth map and point cloud from the rectified images.
   
     ```bash
     python Utility/depthmap.py --left=rect_left --right=rect_right --config=config.yaml
     ```

2. **Disparity and Depth Using RAFT-Stereo** (Recommended):
   - Edit the `run_multiple.sh` script to set the directories for the left and right images, output, point cloud, and depth maps.
   - Run the script:

     ```bash
     bash RAFT-Stereo/run_multiple.sh
     ```

   - This will generate the disparity map in `.npy` format, depth maps, and point clouds for depth visualization.

3. **Verifying Disparity** (Optional):
   - Use `disparity_checker.py` to manually verify the disparity by selecting corresponding points in the left and right images. The script will output the distance between the two points in world coordinates.

   ```bash
   python Utility/disparity_checker.py
   ```

### Step 6: BOP Format Conversion

**Goal**: Convert the dataset into BOP format.

- **Input**: Depth maps, ground truth poses, camera parameters, etc.
- **Tool**: BOP Toolkit.

#### Dataset Structure:

Prepare your dataset according to the following structure:

```
DATASET_NAME
├─ camera[_TYPE].json
├─ dataset_info.json
├─ test_targets_bop19.json
├─ models[_MODELTYPE][_eval]
│  ├─ models_info.json
│  ├─ obj_OBJ_ID.ply
├─ train|val|test[_TYPE]|onboarding_static|onboarding_dynamic
│  ├─ SCENE_ID|OBJ_ID
│  │  ├─ scene_camera.json
│  │  ├─ scene_gt.json
│  │  ├─ scene_gt_info.json
│  │  ├─ depth
│  │  ├─ mask
│  │  ├─ mask_visib
│  │  ├─ rgb|gray
```

#### Steps:

1. Modify the BOP Toolkit files (`config.py` and `dataset_params.py`) to reflect the structure of your dataset.
2. Use the scripts in the `scripts` folder to generate the necessary metadata for your dataset:
   - `create_scene_camera_json.py`: Generates the `scene_camera.json` file with camera parameters for each image.
   - `create_scene_gt.py`: Takes input pose `.npy` files and generates the `scene_gt.json` file.
   - `calc_gt_info.py`, `calc_model_info.py`: Calculate ground truth and model information.
   - `calc_gt_mask.py` or `calc_visible_mask.py`: Extract visible masks for the objects.

### Step 7 (Optional): Synthetic Dataset Generation

**Goal**: Generate synthetic images for object pose estimation using PyBullet and HDRi lighting.

- **Input**: 3D object models and HDRi skybox folder.
- **Tool**: `single_video_pybullet.py` in `Synthetic/Deep_Object_Pose/data_generation`.

#### Steps:

1. Run the script to generate synthetic images:

   ```bash
   python single_video_pybullet.py --nb_frames 10 --scale 0.03 --skyboxes_folder <path_to_skybox_folder> --path_single_obj <model.obj>
   ```

   This command will generate 10 synthetic images using HDRi maps for lighting. Additional arguments like object scale, number of frames, number of distractor objects, and image quality can be adjusted as needed.

---

### Additional Notes

- Ensure that all necessary dependencies, including OpenCV, RAFT-Stereo, and the BOP Toolkit, are installed and properly configured.
- Refer to the documentation and comments within each script for further details on configuration options.

  You can install the dependencies by running:

  ```bash
  pip install -r requirements.txt
  ```