import numpy as np
from plyfile import PlyData, PlyElement

def load_ply(filepath):
    """
    Loads a PLY file.
    
    :param filepath: Path to the PLY file.
    :return: Dictionary containing vertices, normals, texture coordinates, and faces of the PLY model.
    """
    plydata = PlyData.read(filepath)
    vertices = np.vstack([plydata['vertex'][dim] for dim in ('x', 'y', 'z')]).T
    normals = np.vstack([plydata['vertex'][dim] for dim in ('nx', 'ny', 'nz')]).T
    texture_coords = np.vstack([plydata['vertex'][dim] for dim in ('texture_u', 'texture_v')]).T
    faces = np.vstack(plydata['face'].data['vertex_indices'])
    return {'pts': vertices, 'normals': normals, 'texture_coords': texture_coords, 'faces': faces}

def save_ply(filepath, model):
    """
    Saves a PLY file.
    
    :param filepath: Path to save the PLY file.
    :param model: Dictionary containing vertices, normals, texture coordinates, and faces of the PLY model.
    """
    vertices = model['pts']
    normals = model['normals']
    texture_coords = model['texture_coords']
    faces = model['faces']

    vertex_elements = [tuple(vertex) + tuple(normal) + tuple(tex_coord) for vertex, normal, tex_coord in zip(vertices, normals, texture_coords)]
    face_elements = [(face.tolist(),) for face in faces]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('texture_u', 'f4'), ('texture_v', 'f4')]
    face_dtype = [('vertex_indices', 'i4', (3,))]
    
    vertex_array = np.array(vertex_elements, dtype=vertex_dtype)
    face_array = np.array(face_elements, dtype=face_dtype)
    
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    face_element = PlyElement.describe(face_array, 'face')
    
    with open(filepath, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment VCGLIB generated\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property float texture_u\n')
        f.write('property float texture_v\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        
        for vertex, normal, tex_coord in zip(vertices, normals, texture_coords):
            f.write(f'{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} {tex_coord[0]:.6f} {tex_coord[1]:.6f}\n')
        
        for face in faces:
            f.write(f'3 {" ".join(map(str, face))}\n')

def mirror_ply_model(input_ply_path, output_ply_path, axis='y'):
    """
    Mirrors a PLY model along the specified axis.
    
    :param input_ply_path: Path to the input PLY file.
    :param output_ply_path: Path to save the mirrored PLY file.
    :param axis: Axis to mirror the model along ('y' only).
    """
    if axis != 'y':
        raise ValueError("Only 'y' axis mirroring is supported.")

    # Load the PLY model
    model = load_ply(input_ply_path)
    
    # Mirror the vertices and normals along the y-axis
    model['pts'][:, 1] *= -1
    model['normals'][:, 1] *= -1
    
    # Save the mirrored model to a new PLY file
    save_ply(output_ply_path, model)

# Example usage
input_ply_path = '/home/utsav/IProject/data/dataset/lnd1/models/obj_000001.ply'
output_ply_path = '/home/utsav/IProject/data/dataset/lnd1/models/obj_000001_mirrored.ply'
mirror_ply_model(input_ply_path, output_ply_path, axis='y')
