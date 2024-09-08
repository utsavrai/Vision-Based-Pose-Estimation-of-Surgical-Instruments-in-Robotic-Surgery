# Monocular Metric Depth Estimation Pipeline

This folder contains three methods for monocular depth estimation: **Depth Anything**, **UniDepth**, and **Metric3D**. Each method requires specific installation steps and data formats. Follow the instructions below for installation, inference, and visualization.

## Methods Overview
- All methods tested on CUDA 11.8
- **Depth Anything**: A flexible model that uses different transformer-based encoders to estimate depth from a single RGB image.
- **UniDepth**: A method designed for single and batch inference, producing monocular depth predictions.
- **Metric3D**: A model tested for in-the-wild inference using different backbone architectures (transformers or convnets) with multiple focal length settings.

---

## Depth Anything

### Installation

Follow the official installation instructions provided by the **Depth Anything** repository to set up the required environment.

### Running Inference

You can perform inference on either a single image or a directory of images using different transformer-based encoders (`vitl`, `vitb`, `vits`). Use the following command to run inference:

```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```

### Arguments:
- **--img-path**: This can point to:
  - An image directory containing multiple images.
  - A single image.
  - A text file listing paths of images for batch inference.
- **--pred-only**: Saves only the predicted depth map. By default, both the image and its depth map are visualized side-by-side.
- **--grayscale**: Saves the depth map as grayscale. Without this option, a color palette is applied to the depth map by default.

### Example Command:

```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

This will run inference on images in `assets/examples` and output the results to the `depth_vis` folder.

---

## UniDepth

### Installation

Follow the official installation guide for **UniDepth** to set up the necessary environment and dependencies.

### Running Inference

There are two scripts available for inference, depending on whether you're testing on a single image or multiple images:

1. **Single Image Inference**:
   Use the following script to perform inference on a single image:

   ```bash
   python Monocular_metric_depth_estimation/UniDepth/scripts/test.py
   ```

2. **Multiple Images Inference**:
   For batch inference, use the following script:

   ```bash
   python Monocular_metric_depth_estimation/UniDepth/scripts/test_multiple.py
   ```

Ensure that the image paths are correctly set in the respective scripts.

---

## Metric3D

### Installation

Follow the official installation guide for **Metric3D** to set up the environment. Once set up, ensure that the trained model checkpoint (`model.pth`) is placed in the `weight/` directory.

### Running Inference

1. **Prepare the Checkpoint**:
   Place the trained model checkpoint (`model.pth`) in the `weight/` directory.

2. **Modify the Data Path**:
   Open the `test.sh` script and change the `test_data_path` to point to your image folder.

3. **Run the Inference**:
   You can run inference with transformers or convnets depending on your model architecture.

   - **For transformers**:
     ```bash
     source test_vit.sh
     ```

   - **For convnets**:
     ```bash
     source test.sh
     ```

### In-the-Wild Mode:
- Since no camera intrinsics are provided, **Metric3D** uses default settings for 9 focal lengths during inference.

---

### Conclusion

This README provides step-by-step instructions for running inference with three monocular depth estimation methods: **Depth Anything**, **UniDepth**, and **Metric3D**. Make sure to follow the installation instructions and adapt the provided scripts to your input image paths for accurate results.