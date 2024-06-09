
# coding=utf-8
import os
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.io as sio
# from imports.ABIDEDataset import ABIDEDataset
from torch_geometric.data import DataLoader, Data

from scipy.spatial import distance
import scipy.sparse as sp
from os import listdir
import os.path as osp
import torch
import numpy as np
from data import load_data, preprocess_features, preprocess_adj, chebyshev_polynomials



def contrast_adj(x_data):
    # graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    # x_data = x_data
    # graph_feat = graph
    # graph = graph.astype(int)

    # 计算邻接矩阵
    distv = distance.pdist(x_data.cpu().detach().numpy(), metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    # final_graph = graph_feat * sparse_graph
    final_graph = sparse_graph
    adj = final_graph

    supports = chebyshev_polynomials(adj, 2)
    support = supports[1]
    # 从元组中提取非零元素的坐标和值
    nonzero_indices, nonzero_values, shape = support
    # 使用 COO 稀疏矩阵表示法创建稀疏矩阵
    sparse_matrix = sp.coo_matrix((nonzero_values, nonzero_indices.T), shape=shape)
    # 将稀疏矩阵转换为密集矩阵
    dense_matrix = sparse_matrix.toarray()
    adj = torch.from_numpy(dense_matrix)
    return adj