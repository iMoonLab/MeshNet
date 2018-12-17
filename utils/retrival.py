import os
import numpy as np
import matplotlib.pyplot as plt

# all 2468 shapes
top_k = 1000


def append_feature(raw, data, flaten=False):
    data = np.array(data)
    if flaten:
        data = data.reshape(-1, 1)
    if raw is None:
        raw = np.array(data)
    else:
        raw = np.vstack((raw, data))
    return raw


def Eu_dis_mat_fast(X):
    aa = np.sum(np.multiply(X, X), 1)
    ab = X*X.T
    D = aa+aa.T - 2*ab
    D[D<0] = 0
    D = np.sqrt(D)
    D = np.maximum(D, D.T)
    return D


def calculate_map(fts, lbls, dis_mat=None):
    if dis_mat is None:
        dis_mat = Eu_dis_mat_fast(np.mat(fts))
    num = len(lbls)
    mAP = 0
    for i in range(num):
        scores = dis_mat[:, i]
        targets = (lbls == lbls[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        sum = 0
        precision = []
        for j in range(top_k):
            if truth[j]:
                sum+=1
                precision.append(sum*1.0/(j + 1))
        if len(precision) == 0:
            ap = 0
        else:
            for ii in range(len(precision)):
                precision[ii] = max(precision[ii:])
            ap = np.array(precision).mean()
        mAP += ap
        # print(f'{i+1}/{num}\tap:{ap:.3f}\t')
    mAP = mAP/num
    return mAP


def cal_pr(cfg, des_mat, lbls, save=True, draw=False):
    num = len(lbls)
    precisions = []
    recalls = []
    ans = []
    for i in range(num):
        scores = des_mat[:, i]
        targets = (lbls == lbls[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        tmp = 0
        sum = truth[:top_k].sum()
        precision = []
        recall = []
        for j in range(top_k):
            if truth[j]:
                tmp+=1
                # precision.append(sum/(j + 1))
            recall.append(tmp*1.0/sum)
            precision.append(tmp*1.0/(j+1))
        precisions.append(precision)
        for j in range(len(precision)):
            precision[j] = max(precision[j:])
        recalls.append(recall)
        tmp = []
        for ii in range(11):
            min_des = 100
            val = 0
            for j in range(top_k):
                if abs(recall[j] - ii * 0.1) < min_des:
                    min_des = abs(recall[j] - ii * 0.1)
                    val = precision[j]
            tmp.append(val)
        print('%d/%d'%(i+1, num))
        ans.append(tmp)
    ans = np.array(ans).mean(0)
    if save:
        save_dir = os.path.join(cfg.result_sub_folder, 'pr.csv')
        np.savetxt(save_dir, np.array(ans), fmt='%.3f', delimiter=',')
    if draw:
        plt.plot(ans)
        plt.show()


def test():
    scores = [0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11, 0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.1, 0.23, 0.46, 0.08]
    gt_label = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    scores = np.array(scores)
    targets = np.array(gt_label).astype(np.uint8)
    sortind = np.argsort(scores, 0)[::-1]
    truth = targets[sortind]
    sum = 0
    precision = []
    for j in range(20):
        if truth[j]:
            sum += 1
            precision.append(sum / (j + 1))
    if len(precision) == 0:
        ap = 0
    else:
        for i in range(len(precision)):
            precision[i] = max(precision[i:])
        ap = np.array(precision).mean()
    print(ap)


if __name__ == '__main__':
    test()
