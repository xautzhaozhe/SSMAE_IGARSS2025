import pdb
import sys
from sklearn import metrics
import numpy as np

seed_dict = {'Pavia_100': 4521, 'abu-beach-1': 2369}


class ForkedPdb(pdb.Pdb):

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def get_params(net):
    params = []
    params += [x for x in net.parameters()]

    return params


def img2mask(img):
    img = img[0].sum(0)
    img = img - img.min()
    img = img / img.max()
    img = img.detach().cpu().numpy()

    return img


def Residual(contr_data, org_data):
    row, col, band = org_data.shape
    residual = np.square(org_data - contr_data)
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            R = np.mean(residual[i, j, :])
            result[i, j] = R

    return result


def ROC_AUC(target2d, groundtruth):
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    target2d = target2d.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    print('The AUC Value: ', auc)
    return auc


def Mahalanobis(data):
    row, col, band = data.shape
    data = data.reshape(row * col, band)
    mean_vector = np.mean(data, axis=0)
    mean_matrix = np.tile(mean_vector, (row * col, 1))
    re_matrix = data - mean_matrix
    matrix = np.dot(re_matrix.T, re_matrix) / (row * col - 1)
    variance_covariance = np.linalg.pinv(matrix)

    distances = np.zeros([row * col, 1])
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = np.dot(re_array, variance_covariance)
        distances[i] = np.dot(re_var, np.transpose(re_array))
    distances = distances.reshape(row, col)

    return distances
