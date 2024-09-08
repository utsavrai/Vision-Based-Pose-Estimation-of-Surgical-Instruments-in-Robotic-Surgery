#!/bin/bash

# Directories
left_dir="../lnd2/filtered_left_rect_images"
right_dir="../lnd2/rect_right"
output_dir="demo_output/filtered_left_rect_images.npy"
pointcloud_dir="../lnd2/pointclouds_output"
depth_dir="../lnd2/depth_output"

# Create the output directories if they don't exist
mkdir -p "$pointcloud_dir"
mkdir -p "$depth_dir"

# Python script for generating point clouds
generate_pointcloud_script="generate_pointcloud.py"

# Iterate over all images in the left directory
for left_image_path in "$left_dir"/*.png; do
    # Extract the base name (e.g., "761.png")
    image_name=$(basename "$left_image_path")
    # Construct the corresponding right image path
    right_image_path="$right_dir/$image_name"
    
    # Check if the right image exists
    if [ -f "$right_image_path" ]; then
        echo "Processing $image_name"
        # Run the command
        python demo.py --restore_ckpt models/raftstereo-middlebury.pth --corr_implementation alt --mixed_precision -l="$left_image_path" -r="$right_image_path" --save_numpy
        
        # Define the output paths for the point cloud and depth map
        pointcloud_output_path="$pointcloud_dir/${image_name%.*}.ply"
        depth_output_path="$depth_dir/${image_name%.*}.png"

        # Run the point cloud generation script
        python $generate_pointcloud_script "$left_image_path" "$output_dir" "$pointcloud_output_path" "$depth_output_path"
    else
        echo "Right image for $image_name not found!"
    fi
done
