import glob as glob
import numpy as np
import os
import pymesh


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


if __name__ == '__main__':

    root = 'ModelNet40_simplification'
    new_root = 'ModelNet40_MeshNet'

    for type in os.listdir(root):
        for phrase in ['train', 'test']:
            type_path = os.path.join(root, type)
            phrase_path = os.path.join(type_path, phrase)
            if not os.path.exists(type_path):
                os.mkdir(os.path.join(new_root, type))
            if not os.path.exists(phrase_path):
                os.mkdir(phrase)

            files = glob.glob(os.path.join(phrase_path, '*.off'))
            for file in files:
                # load mesh
                mesh = pymesh.load_mesh(file)

                # clean up
                mesh, _ = pymesh.remove_isolated_vertices(mesh)
                mesh, _ = pymesh.remove_duplicated_vertices(mesh)

                # get elements
                vertices = mesh.vertices.copy()
                faces = mesh.faces.copy()

                # move to center
                center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
                vertices -= center

                # normalize
                max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
                vertices /= np.sqrt(max_len)

                # get normal vector
                mesh = pymesh.form_mesh(vertices, faces)
                mesh.add_attribute('face_normal')
                face_normal = mesh.get_face_attribute('face_normal')

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

                _, filename = os.path.split(file)
                np.savez(new_root + type + '/' + phrase + '/' + filename[:-4] + '.npz',
                         faces=faces, neighbors=neighbors)

                print(file)
