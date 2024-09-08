import numpy as np

def update_intrinsics_for_resize(K, original_size, new_size):
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    K_new = K.copy()
    K_new[0, 0] *= scale_x  # f_x
    K_new[1, 1] *= scale_y  # f_y
    K_new[0, 2] *= scale_x  # c_x
    K_new[1, 2] *= scale_y  # c_y

    return K_new

def update_intrinsics_for_padding(K, original_size, new_size):
    padding_x = (new_size[0] - original_size[0]) / 2
    padding_y = (new_size[1] - original_size[1]) / 2

    K_new = K.copy()
    K_new[0, 2] += padding_x  # c_x
    K_new[1, 2] += padding_y  # c_y

    return K_new

def read_intrinsics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        K = np.array([list(map(float, line.split())) for line in lines])
    return K

def write_intrinsics(file_path, K):
    with open(file_path, 'w') as file:
        for row in K:
            file.write(' '.join(map(str, row)) + '\n')

# Example usage
intrinsics_file_path = '/home/utsav/IProject/data/for_pvnet/lnd/camera.txt'
output_resized_file_path = '/home/utsav/IProject/data/for_pvnet/lnd_resize/camera.txt'
# output_padded_file_path = 'path/to/your/padded_intrinsic.txt'

original_size = (960, 540)
new_size_resized = (640, 480)  # Example new size for resize
# new_size_padded = (960, 544)   # Example new size for padding

# Read intrinsic matrix from file
K = read_intrinsics(intrinsics_file_path)

# Update intrinsic matrix for resized and padded images
K_resized = update_intrinsics_for_resize(K, original_size, new_size_resized)
# K_padded = update_intrinsics_for_padding(K, original_size, new_size_padded)

# Write the updated intrinsic matrices to new files
write_intrinsics(output_resized_file_path, K_resized)
# write_intrinsics(output_padded_file_path, K_padded)

print("Original Intrinsics:\n", K)
print("Resized Intrinsics:\n", K_resized)
# print("Padded Intrinsics:\n", K_padded)
