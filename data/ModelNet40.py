import numpy as np
import os
import torch
import torch.utils.data as data
import pymeshlab
from data.preprocess import find_neighbor

type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}


class ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.max_faces = cfg['max_faces']
        self.part = part
        self.augment_data = cfg['augment_data']
        if self.augment_data:
            self.jitter_sigma = cfg['jitter_sigma']
            self.jitter_clip = cfg['jitter_clip']

        self.data = []
        for type in os.listdir(self.root):
            if type not in type_to_index_map.keys():
                continue
            type_index = type_to_index_map[type]
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz') or filename.endswith('.obj'):
                    self.data.append((os.path.join(type_root, filename), type_index))

    def __getitem__(self, i):
        path, type = self.data[i]
        if path.endswith('.npz'):
            data = np.load(path)
            face = data['faces']
            neighbor_index = data['neighbors']
        else:
            face, neighbor_index = process_mesh(path, self.max_faces)
            if face is None:
                return self.__getitem__(0)

        # data augmentation
        if self.augment_data and self.part == 'train':
            # jitter
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*face[:, :3].shape), -1 * self.jitter_clip, self.jitter_clip)
            face = np.concatenate((face[:, :3] + jittered_data, face[:, 3:]), 1)

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)


def process_mesh(path, max_faces):
    ms = pymeshlab.MeshSet()
    ms.clear()

    # load mesh
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    
    # # clean up
    # mesh, _ = pymesh.remove_isolated_vertices(mesh)
    # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

    # get elements
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    if faces.shape[0] != max_faces:     # only occur once in train set of Manifold40
        print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], path))
        return None, None

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

    return faces, neighbors
