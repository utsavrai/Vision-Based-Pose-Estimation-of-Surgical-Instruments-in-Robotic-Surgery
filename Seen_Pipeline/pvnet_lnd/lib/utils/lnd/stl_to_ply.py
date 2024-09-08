# # Example usage
stl_file_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/joint.stl'
# ply_file_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/lnd.ply'
# # convert_stl_to_ply(stl_file_path, ply_file_path)
# # print(f'Converted {stl_file_path} to {ply_file_path}')


# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh(stl_file_path)
# pointcloud = mesh.sample_points_poisson_disk(100000)

# # you can plot and check
# # o3d.visualization.draw_geometries([mesh])
# o3d.visualization.draw_geometries([pointcloud])



# import open3d as o3d
# import numpy as np

# def convert_stl_to_pointcloud_with_gray_color(stl_file_path, ply_file_path, num_points=100000):
#     # Load the STL mesh
#     mesh = o3d.io.read_triangle_mesh(stl_file_path)
    
#     # Sample points from the mesh to create a point cloud
#     pointcloud = mesh.sample_points_poisson_disk(num_points)
    
#     # Set all vertex colors to gray
#     gray_color = np.array([[128, 128, 128]] * num_points, dtype=np.float64) / 255.0  # Gray color in normalized RGB
#     pointcloud.colors = o3d.utility.Vector3dVector(gray_color)
    
#     # Save the point cloud to a PLY file
#     o3d.io.write_point_cloud(ply_file_path, pointcloud)
#     print(f'Saved point cloud with gray color to {ply_file_path}')

# # Example usage
# stl_file_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/joint.stl'
# ply_file_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/pointcloud_gray.ply'
# convert_stl_to_pointcloud_with_gray_color(stl_file_path, ply_file_path)


import open3d as o3d

def convert_pointcloud_to_mesh(pointcloud_path, mesh_output_path, depth=8):
    # Load the point cloud
    pointcloud = o3d.io.read_point_cloud(pointcloud_path)
    
    # Estimate normals
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointcloud.normalize_normals()
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud, depth=depth)
    
    # Save the mesh to a PLY file
    o3d.io.write_triangle_mesh(mesh_output_path, mesh, write_ascii=True)
    print(f'Saved reconstructed mesh to {mesh_output_path}')

# Example usage
pointcloud_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/pointcloud_gray.ply'
mesh_output_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/reconstructed_mesh.ply'
convert_pointcloud_to_mesh(pointcloud_path, mesh_output_path)

