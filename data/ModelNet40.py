import numpy as np
import os
import torch
import torch.utils.data as data


class ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.augment_data = cfg['augment_data']
        self.part = part

        self.data = []
        type_index = 0
        for type in os.listdir(self.root):
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), type_index))
            type_index += 1

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < 1024
        num_point = len(face)
        if num_point < 1024:
            fill_face = []
            fill_neighbor_index = []
            for i in range(1024 - num_point):
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
