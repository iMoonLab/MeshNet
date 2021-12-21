import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet
from utils.retrival import append_feature, calculate_map


cfg = get_test_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = ModelNet40(cfg=cfg['dataset'], part='test')
data_loader = data.DataLoader(data_set, batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)


def test_model(model):

    correct_num = 0
    ft_all, lbl_all = None, None

    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
            centers = centers.cuda()
            corners = corners.cuda()
            normals = normals.cuda()
            neighbor_index = neighbor_index.cuda()
            targets = targets.cuda()

            outputs, feas = model(centers, corners, normals, neighbor_index)
            _, preds = torch.max(outputs, 1)

            correct_num += (preds == targets).float().sum()

            if cfg['retrieval_on']:
                ft_all = append_feature(ft_all, feas.detach().cpu())
                lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))
    if cfg['retrieval_on']:
        print('mAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))


if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg['load_model']))
    model.eval()

    test_model(model)
