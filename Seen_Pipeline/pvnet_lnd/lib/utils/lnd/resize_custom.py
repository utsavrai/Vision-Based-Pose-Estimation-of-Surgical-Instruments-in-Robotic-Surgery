# from PIL import Image
# import os

# def resize_image(image_path, output_path, target_size):
#     with Image.open(image_path) as img:
#         resized_img = img.resize(target_size, Image.LANCZOS)
#         resized_img.save(output_path)

# def resize_images_in_directory(input_dir, output_dir, target_size):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for filename in os.listdir(input_dir):
#         if filename.endswith(('.jpg', '.png')):
#             input_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, filename)
#             resize_image(input_path, output_path, target_size)
#             print(f"Image {filename} resized successfully!")

# # Paths to your directories
# rgb_dir = '/home/utsav/IProject/clean-pvnet/data/custom/rgb'
# mask_dir = '/home/utsav/IProject/clean-pvnet/data/custom/mask'

# # Output directories for resized images
# resized_rgb_dir = '/home/utsav/IProject/clean-pvnet/data/custom/rgb'
# resized_mask_dir = '/home/utsav/IProject/clean-pvnet/data/custom/mask'

# # Desired size (width 960, height 544)
# new_size = (640, 480)

# # Resize the images in both directories
# resize_images_in_directory(rgb_dir, resized_rgb_dir, new_size)
# resize_images_in_directory(mask_dir, resized_mask_dir, new_size)

# print("All images resized successfully!")


from PIL import Image
import os

# def resize_image(image_path, output_path, target_size):
#     with Image.open(image_path) as img:
#         resized_img = img.resize(target_size, Image.LANCZOS)
#         resized_img.save(output_path)

def resize_image(image_path, output_path, target_size):
    with Image.open(image_path) as img:
        # Handle different image modes
        if img.mode == 'I;16':  # If depth image is in 16-bit integer format
            print(f"Converting depth image {image_path} from 'I;16' to 'F' mode for resizing")
            img = img.convert('F')  # Convert to 32-bit float for resizing
            
        resized_img = img.resize(target_size, Image.LANCZOS)
        
        # Save the resized depth image in TIFF format
        output_path_tiff = output_path.replace('.png', '.tiff')  # Change extension to .tiff
        resized_img.save(output_path_tiff, format='TIFF')
        print(f"Resized image saved as TIFF: {output_path_tiff}")

def resize_images_in_directory(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_image(input_path, output_path, target_size)
            print(f"Image {filename} resized successfully!")

# Paths to your directories
# rgb_dir = '/home/utsav/IProject/clean-pvnet/data/custom/rgb'
# mask_dir = '/home/utsav/IProject/clean-pvnet/data/custom/mask'
depth_dir = '/home/utsav/IProject/clean-pvnet/data/custom/depth'  # New depth directory

# Output directories for resized images
# resized_rgb_dir = '/home/utsav/IProject/clean-pvnet/data/custom/rgb'
# resized_mask_dir = '/home/utsav/IProject/clean-pvnet/data/custom/mask'
resized_depth_dir = '/home/utsav/IProject/clean-pvnet/data/custom/depth'  # New resized depth directory

# Desired size (width 960, height 544)
new_size = (640, 480)

# Resize the images in all directories
# resize_images_in_directory(rgb_dir, resized_rgb_dir, new_size)
# resize_images_in_directory(mask_dir, resized_mask_dir, new_size)
resize_images_in_directory(depth_dir, resized_depth_dir, new_size)  # Resize depth images

print("All images resized successfully!")
