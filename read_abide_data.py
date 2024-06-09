from sklearn.model_selection import StratifiedKFold

from graph import ABIDEParser_1_HOFC_selected as Reader
# coding=utf-8
import os
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.io as sio
from torch_geometric.data import DataLoader, Data
from scipy.spatial import distance
import scipy.sparse as sp
from os import listdir
import os.path as osp
import torch
import numpy as np
from data import load_data, preprocess_features, preprocess_adj, chebyshev_polynomials


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data_source(root, name, rfe_selector):
    # 读取下标、特征、标签
    subject_IDs = np.genfromtxt(os.path.join(root, 'subject_IDs.txt'), dtype=str)
    # index = np.genfromtxt(os.path.join('G:/pytorch/UDAGCN-master/ABIDE/ABIDE_UM_1', 'subject_IDs.txt'), dtype=str)
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    # 制作标签
    num_classes = 2
    num_nodes = len(subject_IDs)
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=int)
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]]) - 1
        site[i] = unique.index(sites[subject_IDs[i]])

    # 特征集合
    all_networks = []
    for subject in subject_IDs:
        fl = os.path.join(root, subject,
                          subject + "_ho_correlation.mat")
        matrix = sio.loadmat(fl)['connectivity']
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    features = np.vstack(vec_networks)

    # features = np.delete(features, np.where(features == 0)[1], axis=1)
    features = features.astype(np.float32)

    # 特征选择方法
    selector = rfe_selector.fit(features, y.ravel())
    source_data_x = selector.transform(features)
    selected_feature_indices = rfe_selector.get_support(indices=True)
    selected_feature = set()
    selected_feature.update(selected_feature_indices)
    selected_feature_indices = sorted(list(selected_feature))

    # 获取并组合表型信息
    # index = subject_IDs.astype(str)
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    x_data = source_data_x
    graph_feat = graph
    graph = graph.astype(int)

    # 计算邻接矩阵
    distv = distance.pdist(x_data, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * sparse_graph
    adj = final_graph

    supports = chebyshev_polynomials(adj, 1)
    support = supports[1]
    # 从元组中提取非零元素的坐标和值
    nonzero_indices, nonzero_values, shape = support
    # 使用 COO 稀疏矩阵表示法创建稀疏矩阵
    sparse_matrix = sp.coo_matrix((nonzero_values, nonzero_indices.T), shape=shape)
    # 将稀疏矩阵转换为密集矩阵
    dense_matrix = sparse_matrix.toarray()
    adj = torch.from_numpy(dense_matrix).to(device)

    # 将x和y,权重转换为tensor
    x_data = torch.from_numpy(x_data).to(device)
    y = torch.from_numpy(y).to(device)
    y = y.to(torch.long)
    y = y.view(-1)
    return x_data, y, adj, selected_feature_indices


def read_data_target(root, name, selected_feature_indices):
    # 读取下标、特征、标签
    subject_IDs = np.genfromtxt(os.path.join(root, 'subject_IDs.txt'), dtype=str)
    # index = np.genfromtxt(os.path.join('G:/pytorch/UDAGCN-master/ABIDE/ABIDE_UM_1', 'subject_IDs.txt'), dtype=str)
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    # 制作标签
    num_classes = 2
    num_nodes = len(subject_IDs)
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=int)
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]]) - 1
        site[i] = unique.index(sites[subject_IDs[i]])

    # 特征集合

    all_networks = []
    for subject in subject_IDs:
        fl = os.path.join(root, subject,
                          subject + "_ho_correlation.mat")
        matrix = sio.loadmat(fl)['connectivity']
        all_networks.append(matrix)


    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    features = np.vstack(vec_networks)

    features = features.astype(np.float32)


    # 获取并组合表型信息
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    x_data = features[:, selected_feature_indices]
    graph_feat = graph
    graph = graph.astype(int)

    # 计算邻接矩阵
    distv = distance.pdist(x_data, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * sparse_graph
    adj = final_graph

    supports = chebyshev_polynomials(adj, 1)
    support = supports[1]
    # 从元组中提取非零元素的坐标和值
    nonzero_indices, nonzero_values, shape = support
    # 使用 COO 稀疏矩阵表示法创建稀疏矩阵
    sparse_matrix = sp.coo_matrix((nonzero_values, nonzero_indices.T), shape=shape)
    # 将稀疏矩阵转换为密集矩阵
    dense_matrix = sparse_matrix.toarray()
    adj = torch.from_numpy(dense_matrix).to(device)


    # 将x和y,权重转换为tensor
    x_data = torch.from_numpy(x_data).to(device)
    y = torch.from_numpy(y).to(device)
    y = y.to(torch.long)
    y = y.view(-1)
    return x_data, y, adj