import numpy as np
import cv2

def load_image(file_path, dtype):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(dtype)

def compare_depth_images(img1, img2):
    if img1.shape != img2.shape:
        print("Images have different dimensions!")
        return False

    if not np.array_equal(img1, img2):
        diff = np.abs(img1 - img2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"Images are not identical. Max difference: {max_diff}, Mean difference: {mean_diff}")
        return False
    
    print("Images are identical.")
    return True

# Load the int32 and int16 depth images
depth_image_int32_path = '/home/utsav/IProject/data/lnd1/depth_in32/130.png'
depth_image_int16_path = '/home/utsav/IProject/data/lnd1/depth/130.png'

depth_image_int32 = load_image(depth_image_int32_path, np.int32)
depth_image_int16 = load_image(depth_image_int16_path, np.int16)

# Optionally convert int32 image to int16 format if needed
# Assuming the depth values are in the range where this conversion makes sense.
# This step might involve scaling depending on the depth value ranges.
# For direct comparison without scaling:
depth_image_int32_as_int16 = depth_image_int32.astype(np.int16)

# Compare the images
are_identical = compare_depth_images(depth_image_int32_as_int16, depth_image_int16)

if not are_identical:
    # Optionally, save the difference image for visual inspection
    diff_image = np.abs(depth_image_int32_as_int16 - depth_image_int16)
    cv2.imwrite('difference_image.png', diff_image)
    print("Difference image saved as difference_image.png")
