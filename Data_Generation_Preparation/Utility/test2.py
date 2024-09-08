import open3d as o3d
import numpy as np

def center_mesh(mesh):
    mesh_center = mesh.get_center()
    print(f"Mesh center before centering: {mesh_center}")
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = -mesh_center
    mesh.transform(transformation_matrix)
    mesh_center_after = mesh.get_center()
    print(f"Mesh center after centering: {mesh_center_after}")
    return mesh

def draw_axes(vis, length=1.0):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=length, origin=[0, 0, 0])
    vis.add_geometry(axes)

def visualize_model_with_axes(cad_model_path):
    mesh = o3d.io.read_triangle_mesh(cad_model_path)
    
    # Center the CAD model
    mesh = center_mesh(mesh)
    
    # Visualize the model
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    
    # Draw the origin axes
    draw_axes(vis, length=10.0)  # Increase length for better visibility
    
    # Set the view control to include the model and the axes
    ctr = vis.get_view_control()
    ctr.set_up([0, -1, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()

# Path to the CAD model
cad_model_path = '/home/utsav/Downloads/Synapse_dataset/LND_TRAIN/TRAIN/joint.stl'

# Visualize the model with axes
visualize_model_with_axes(cad_model_path)
