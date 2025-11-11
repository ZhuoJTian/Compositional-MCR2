import os
import random
import numpy as np
from distutils.dir_util import copy_tree

import torch
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import scipy.spatial as sp
import matplotlib.pyplot as plt

def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def svm(train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test

def knn(train_features, train_labels, test_features, test_labels, topk=(1,)):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN
    
    """
    maxk = max(topk)
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(maxk, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc

def nearsub(train_features, train_labels, test_features, test_labels, n_comp=30):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = 10 # .max() + 1 # should be correct most of the time
    # print(label_cor) 
    features_sort, _ = sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                    num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = compute_accuracy(test_predict_svd, test_labels.numpy())
    # print('PCA: {}'.format(acc_pca))
    # print('SVD: {}'.format(acc_svd))
    return acc_pca, acc_svd

def accuracy_softmax(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        print(correct)
        correct_k = correct[:1].contiguous().view(-1).float().sum(0, keepdim=True)
        print(correct_k)
    return correct_k

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def accuracy_topk(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k
    Top-K accuracy is the percentage of times that the model includes the correct prediction among the top-K probabilities."""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res[-1]


def plot_corZ(Z_all, label_all, agent_all, path_fig):
    """
    Parameters:
        Z_all: numpy array, shape (N, d) 样本特征
        label_all: numpy array, shape (N,) 样本标签
        agent_all: numpy array, shape (N,) agent id
        path_fig: str 保存路径
    """

    num_samples = Z_all.shape[0]

    # 按照 label 排序
    sorted_id = np.argsort(label_all)
    Z_new = Z_all[sorted_id]
    label_new = label_all[sorted_id]
    agent_new = agent_all[sorted_id]

    # 计算余弦相似度矩阵
    result_matrix = 1 - sp.distance.cdist(Z_new, Z_new, metric="cosine")
    print("get result_matrix")

    # 检查 NaN
    diag_nan = np.isnan(np.diag(result_matrix))
    if np.any(diag_nan):
        bad_idx = np.where(diag_nan)[0]
        for i in bad_idx:
            print("0 results:", i, Z_new[i], label_new[i], agent_new[i])

    # 可视化
    plt.matshow(result_matrix, cmap=plt.cm.Blues)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(0, 1)
    plt.savefig(path_fig)
    plt.close()

    return 0


def visualize_samples(dataset, path_fig, num_samples=5):
    """
    可视化某个视角数据集里的若干样本
    """
    plt.figure(figsize=(15, 3))
    num = 0
    for i in np.random.choice(len(dataset), size=num_samples, replace=False):
        img, label = dataset[i]   # 从 Dataset 取数据
        # img 是 Tensor [3, H, W]，转回 numpy [H, W, 3]
        img = img.permute(1, 2, 0).numpy()
        num += 1
        plt.subplot(1, num_samples, num)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.savefig(path_fig)
    plt.close()