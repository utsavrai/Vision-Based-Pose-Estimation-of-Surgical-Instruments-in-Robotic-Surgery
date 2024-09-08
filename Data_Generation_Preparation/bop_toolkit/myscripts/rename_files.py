import os

def rename_files(base_path):
    folders = ['rgb', 'mask', 'depth', 'pose']
    object_id_suffix = '_000001'  # Suffix for mask files

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        for filename in os.listdir(folder_path):
            name, ext = os.path.splitext(filename)
            if folder == 'pose' and ext == '.npy':
                # Special handling for pose files
                new_name = f"{int(name):06}{ext}"
            elif ext == '.png':
                if folder == 'mask':
                    # Rename mask files with object ID suffix
                    new_name = f"{int(name):06}{object_id_suffix}{ext}"
                else:
                    # Rename rgb and depth files
                    new_name = f"{int(name):06}{ext}"
            else:
                print(f"Unknown file type: {filename}. Skipping.")
                continue

            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")

# Example usage
base_path = "/home/utsav/IProject/data/dataset/lnd2/train/000001"  # Replace with the path to your base folder containing rgb, mask, depth, and pose folders
rename_files(base_path)
