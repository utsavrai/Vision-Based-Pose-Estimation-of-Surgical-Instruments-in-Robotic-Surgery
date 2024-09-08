import os
import random
import shutil

def select_and_copy_images(left_source_dir, right_source_dir, left_target_dir, right_target_dir, num_images):
    # Ensure target directories exist
    if not os.path.exists(left_target_dir):
        os.makedirs(left_target_dir)
    if not os.path.exists(right_target_dir):
        os.makedirs(right_target_dir)

    # List all files in the left source directory
    left_files = os.listdir(left_source_dir)
    right_files = os.listdir(right_source_dir)
    
    # Filter out only image files (you can add more image file extensions if needed)
    left_image_files = [f for f in left_files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    right_image_files = [f for f in right_files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # Find common image files in both directories
    common_files = list(set(left_image_files) & set(right_image_files))
    
    # Randomly select the specified number of images
    selected_files = random.sample(common_files, num_images)
    
    # Copy selected images to the target directories
    for file in selected_files:
        shutil.copy(os.path.join(left_source_dir, file), os.path.join(left_target_dir, file))
        shutil.copy(os.path.join(right_source_dir, file), os.path.join(right_target_dir, file))

# Define source and target directories
left_source_dir = 'left'
right_source_dir = 'right'
left_target_dir = 'left_random'
right_target_dir = 'right_random'

# Number of images to select
num_images = 50

# Select and copy images
select_and_copy_images(left_source_dir, right_source_dir, left_target_dir, right_target_dir, num_images)

print(f'Selected {num_images} common images from {left_source_dir} and {right_source_dir}, and saved to {left_target_dir} and {right_target_dir}')
