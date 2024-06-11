import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from sklearn.model_selection import KFold

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim, device):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1. / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, ndomains):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)  # disable for digits
    self.ad_layer3 = nn.Linear(hidden_size, ndomains)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()  # disable for digits
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)  # disable for digits
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0
    self.ndomains = ndomains

  def output_num(self):
    return self.ndomains

  def get_parameters(self):
    return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

  def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
      return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

  def grl_hook(self, coeff):
      def fun1(grad):
          return -coeff * grad.clone()

      return fun1

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = self.calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.requires_grad_(True)
    if self.training:
        x.register_hook(self.grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)  # disable for digits
    x = self.relu2(x)  # disable for digits
    x = self.dropout2(x)  # disable for digits
    y = self.ad_layer3(x)
    return y




class mlp1(nn.Module):
    def __init__(self, networks_name, class_num=2, in_features=1500):
        super(mlp1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, 256),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )
        # self.linear.apply(init_weights)
        self.fc = nn.Linear(256, 2)
        # self.fc.apply(init_weights)
        self.__in_features = 256

    def forward(self, x):

        x = self.linear(x)
        # x = torch.sigmoid(x)
        y = self.fc(x)
        # y = torch.sigmoid(y)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [
            {'params': self.linear.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]

        return parameter_list


def Split_Sets_10_Fold(total_fold, data, args):
    # train_index,test_index用来存储train和test的index（索引）
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=args.random_state)
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index

def init_weights1(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.1) ## or simply use your layer.reset_parameters()
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(1 / m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(4 / m.in_channels))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def find_indices_in_graph(inputs_target, graph_nodes):
    indices_list = []
    for i in range(inputs_target.size(0)):
        # 获取当前目标域样本
        target_sample = inputs_target[i]
        # 在 graph_nodes 中查找匹配的行索引
        match_index = torch.where(torch.all(torch.eq(graph_nodes, target_sample.unsqueeze(0)), dim=1))[0]
        indices_list.extend(match_index.tolist())  # 将所有匹配的索引添加到列表中
    # 移除重复的索引
    indices_list = list(set(indices_list))
    return indices_list

