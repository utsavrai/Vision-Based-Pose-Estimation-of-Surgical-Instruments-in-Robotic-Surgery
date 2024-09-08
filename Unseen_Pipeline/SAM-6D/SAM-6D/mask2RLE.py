import os
import numpy as np
from skimage.io import imread
import json

# Function to calculate the bounding box from a binary mask
def bbox_from_binary_mask(binary_mask):
    """Returns the smallest bounding box containing all pixels marked '1' in the given image mask."""
    if not np.any(binary_mask):  # No object detected
        return [0, 0, 0, 0]  # Return a dummy bounding box
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    return [int(cmin), int(rmin), int(w), int(h)]

# Function to convert a binary mask to RLE format
def binary_mask_to_rle(binary_mask):
    """Converts a binary mask to COCO's run-length encoding (RLE) format."""
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    mask = binary_mask.ravel(order="F")
    if len(mask) > 0 and mask[0] == 1:
        counts.append(0)

    if len(mask) > 0:
        mask_changes = mask[:-1] != mask[1:]
        changes_indx = np.where(np.concatenate(([True], mask_changes, [True]), 0))[0]
        rle2 = np.diff(changes_indx)
        counts.extend(rle2.tolist())
    return rle

# Main function to process mask images and create a JSON file for each image
def process_mask_images(mask_dir, output_dir, category_id=1, scene_id=0):
    """
    Converts all mask images in a folder into separate JSON files.
    
    :param mask_dir: Directory containing the binary mask images.
    :param output_dir: Directory where the JSON files will be saved.
    :param category_id: Category ID to assign to all masks.
    :param scene_id: Scene ID to assign to all images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all mask images in the directory
    mask_images = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    for image_id, mask_filename in enumerate(mask_images):
        # Read the binary mask
        mask_path = os.path.join(mask_dir, mask_filename)
        binary_mask = imread(mask_path)
        
        # Ensure binary mask is boolean
        binary_mask = binary_mask > 0
        
        # Compute bounding_box
        bbox = bbox_from_binary_mask(binary_mask)
        
        # Convert mask to RLE format
        if np.any(binary_mask):
            segmentation = binary_mask_to_rle(binary_mask)
        else:
            segmentation = {"counts": [], "size": list(binary_mask.shape)}  # No segmentation
        
        # Create annotation info
        annotation = {
            "scene_id": scene_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": 1.0,  # Ground truth, so confidence is 100%
            "time": 0.0,
            "segmentation": segmentation
        }
        
        # Define output JSON file path
        output_json_path = os.path.join(output_dir, f"{os.path.splitext(mask_filename)[0]}.json")
        
        # Write to the JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump([annotation], json_file, indent=4)

        print(f"Annotation for {mask_filename} saved to {output_json_path}")

# Usage example
mask_dir = '../../data/dataset/lnd2/train/000001/mask_visib'  # Directory containing the mask images
output_dir = '../../data/dataset/lnd2/train/000001/rle_jsons'  # Directory where individual JSON files will be saved
process_mask_images(mask_dir, output_dir)
