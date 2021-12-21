import os
import numpy as np
import scipy
import scipy.spatial


def append_feature(raw, data, flaten=False):
    data = np.array(data)
    if flaten:
        data = data.reshape(-1, 1)
    if raw is None:
        raw = np.array(data)
    else:
        raw = np.vstack((raw, data))
    return raw


def calculate_map(fts, lbls, dis_mat=None):
    return map_score(fts, fts, lbls, lbls)


def acc_score(y_true, y_pred, average="micro"):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if average == "micro": 
        # overall
        return np.mean(y_true == y_pred)
    elif average == "macro":
        # average of each class
        cls_acc = []
        for cls_idx in np.unique(y_true):
            cls_acc.append(np.mean(y_pred[y_true==cls_idx]==cls_idx))
        return np.mean(np.array(cls_acc))
    else:
        raise NotImplementedError

def cdist(fts_a, fts_b, metric):
    if metric == 'inner':
        return np.matmul(fts_a, fts_b.T)
    else:
        return scipy.spatial.distance.cdist(fts_a, fts_b, metric)

def map_score(fts_a, fts_b, lbl_a, lbl_b, metric='cosine'):
    dist = cdist(fts_a, fts_b, metric)
    res = map_from_dist(dist, lbl_a, lbl_b)
    return res


def map_from_dist(dist, lbl_a, lbl_b):
    n_a, n_b = dist.shape
    s_idx = dist.argsort()

    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)
