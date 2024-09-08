import os
import shutil

def merge_folders(dataset1_path, dataset2_path, merged_dataset_path):
    folders = ['mask', 'rgb', 'pose']

    for folder in folders:
        # Define paths for the current folder in dataset1, dataset2, and the merged dataset
        dataset1_folder = os.path.join(dataset1_path, folder)
        dataset2_folder = os.path.join(dataset2_path, folder)
        merged_folder = os.path.join(merged_dataset_path, folder)

        # Create the merged folder if it doesn't exist
        os.makedirs(merged_folder, exist_ok=True)

        # Initialize a counter for naming files numerically
        file_counter = 1

        # Copy and rename files from dataset1
        for filename in os.listdir(dataset1_folder):
            src_file = os.path.join(dataset1_folder, filename)
            dst_file = os.path.join(merged_folder, f'{file_counter:06}.png')  # Assuming files are .png, adjust extension as needed
            shutil.copy(src_file, dst_file)
            file_counter += 1

        # Copy and rename files from dataset2
        for filename in os.listdir(dataset2_folder):
            src_file = os.path.join(dataset2_folder, filename)
            dst_file = os.path.join(merged_folder, f'{file_counter:06}.png')  # Adjust extension if necessary
            shutil.copy(src_file, dst_file)
            file_counter += 1

        print(f"Folder '{folder}' merged successfully with {file_counter - 1} files!")

# Define paths to the two datasets and the merged dataset
dataset1_path = '/home/utsav/Downloads/LND_TRAIN/TRAIN'
dataset2_path = '/home/utsav/IProject/data/dataset/lnd1/train/000001'
merged_dataset_path = '/home/utsav/Downloads/lnd_bop_train'

# Merge the folders
merge_folders(dataset1_path, dataset2_path, merged_dataset_path)
