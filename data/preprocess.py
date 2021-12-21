import pymeshlab
import numpy as np
from pathlib import Path
from rich.progress import track


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

if __name__ == '__main__':
    root = Path('dataset/Manifold40')
    new_root = Path('dataset/ModelNet40_processed')
    max_faces = 500
    shape_list = sorted(list(root.glob('*/*/*.obj')))
    ms = pymeshlab.MeshSet()

    for shape_dir in track(shape_list):
        out_dir = new_root / shape_dir.relative_to(root).with_suffix('.npz')
        # if out_dir.exists():
        #     continue
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        ms.clear()
        # load mesh
        ms.load_new_mesh(str(shape_dir))
        mesh = ms.current_mesh()
        
        # # clean up
        # mesh, _ = pymesh.remove_isolated_vertices(mesh)
        # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

        # get elements
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()

        if faces.shape[0] != max_faces:
            print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], out_dir))
            continue

        # move to center
        center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
        vertices -= center

        # normalize
        max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
        vertices /= np.sqrt(max_len)

        # get normal vector
        ms.clear()
        mesh = pymeshlab.Mesh(vertices, faces)
        ms.add_mesh(mesh)
        face_normal = ms.current_mesh().face_normal_matrix()

        # get neighbors
        faces_contain_this_vertex = []
        for i in range(len(vertices)):
            faces_contain_this_vertex.append(set([]))
        centers = []
        corners = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            x1, y1, z1 = vertices[v1]
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]
            centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
            corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        neighbors = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])

        centers = np.array(centers)
        corners = np.array(corners)
        faces = np.concatenate([centers, corners, face_normal], axis=1)
        neighbors = np.array(neighbors)

        np.savez(str(out_dir), faces=faces, neighbors=neighbors)
