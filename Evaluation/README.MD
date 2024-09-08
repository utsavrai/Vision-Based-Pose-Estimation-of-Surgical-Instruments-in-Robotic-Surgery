# Evaluation Pipeline

This repository provides an evaluation pipeline for both unseen and seen object pose estimation models. The evaluation is conducted using two common metrics in the research community: **ADD (Average Distance of Model Points)** and **2D Projection Metric**. The scripts are organized into two pipelines: **unseen_pipeline** for unseen object evaluation and **seen_pipeline** for seen object evaluation.

## Evaluation Metrics

### 1. **ADD (Average Distance of Model Points)**
   - Measures the distance between the predicted and ground truth poses, specifically the alignment of corresponding 3D points on the model.
   - Evaluated over a range of 0% to 50% of the 3D model diameter of the LND tool.
   - The threshold commonly used in research is **ADD (10%)**, representing a 10% tolerance of the 3D model diameter.

### 2. **2D Projection Metric**
   - Measures the accuracy of the 2D projection of the 3D model when rendered from the predicted pose onto the image plane.
   - Evaluated over a range of 0 to 50 pixels, with **5 pixels** as the standard threshold for evaluation.

## Evaluation Pipelines

### Unseen Object Evaluation (Unseen Pipeline)

This pipeline is designed for evaluating models trained on unseen objects. The **unseen_pipeline** folder contains evaluation scripts for multiple pose estimation models. These scripts are compatible with the model-specific output formats.

#### Supported Models:
1. **FoundationPose**
2. **SAM-6D with Mask R-CNN**
3. **SAM-6D with SAM (Segment Anything Model)**
4. **OVE6D**
5. **Megapose**

Each script in this pipeline computes the ADD and 2D Projection metrics for the corresponding model’s predictions and ground truth poses. The ground truth and predicted poses should be provided in a compatible format specific to each model’s output.

#### Steps:
1. Run the respective evaluation script for the desired model to compute the ADD and 2D Projection metrics.
2. The evaluation is performed across a range of thresholds for both metrics.
3. Results are saved for further analysis and visualization.

### Seen Object Evaluation (Seen Pipeline)

The **seen_pipeline** contains scripts for evaluating models that are trained and tested on the same set of objects (i.e., seen objects). The following two pose estimation models are supported in this pipeline:

#### Supported Models:
1. **GDR-Net**
2. **PVNet**

Similar to the unseen pipeline, the ADD and 2D Projection metrics are computed across various thresholds, with **10% of the model diameter** and **5 pixels** being the standard thresholds.

#### Steps:
1. Run the respective script for **GDR-Net** or **PVNet** to compute the evaluation metrics.
2. The results are stored for visualization.

## Visualizing Results

Once the evaluation metrics are computed for all the models, a **plot_all.py** script is provided to generate visualizations of the evaluation results. This script aggregates the results from the different models and presents them in a comparative format for better understanding of model performance.

### How to Run:
1. After completing evaluations using the unseen and seen pipelines, ensure all results are collected.
2. Run the `plot_all.py` script to generate plots comparing ADD and 2D Projection metrics across models.

   ```bash
   python plot_all.py
   ```

### Example Output:
- Comparative bar charts or line plots showing model performance for both ADD and 2D Projection metrics across various thresholds.
- This provides a quick and clear visualization of how each model performs under different conditions.

---

## Summary of Files and Scripts

- **Unseen Pipeline:**
  - Evaluation scripts for:
    - FoundationPose
    - SAM-6D (Mask R-CNN)
    - SAM-6D (SAM)
    - OVE6D
    - Megapose

- **Seen Pipeline:**
  - Evaluation scripts for:
    - GDR-Net
    - PVNet

- **plot_all.py**: 
  - Script to plot evaluation results for both unseen and seen models.

---

### Notes:
- Ensure that the predictions and ground truth poses are provided in the correct format for each model.
- The default thresholds for ADD (10%) and 2D Projection (5 pixels) are the commonly accepted benchmarks but can be customized in the scripts.
