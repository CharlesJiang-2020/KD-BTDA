import os
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier, Lasso, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from trainer import evaluate
from sklearn.preprocessing import StandardScaler
import transfer_loss
import utils
import trainer
import networks
from torch_geometric.data import DataLoader, Data
from contrast_adj import contrast_adj
from read_abide_data import read_data_source as reader_source
from read_abide_data import read_data_target as reader_target
from models import GCN

MLP_Accuracy = []
target_1_acc = []
target_2_acc = []
target_3_acc = []
target_4_acc = []

for i in range(10):
    print('\n','**' * 10, '第', i + 1, '折', 'ing....', '**' * 10, '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Graph Curriculum Domain Adaptaion')
    # model args
    parser.add_argument('--method', type=str, default='CDAN')
    parser.add_argument('--encoder', type=str, default='mlp1')
    parser.add_argument('--rand_proj', type=int, default=1024, help='random projection dimension')

    parser.add_argument('--save_models', type=bool, default=True, help='whether to save models')
    parser.add_argument('--dataset', type=str, default='ABIDE', help='dataset used')
    parser.add_argument('--kfold', type=int, default=10, help='Cross validation')
    parser.add_argument('--estimator', type=str, default='RidgeClassifier()', choices=['RidgeClassifier()', 'LogisticRegression()'])
    parser.add_argument('--max_key_size', type=int, default=16384, help='Maximum number of key feature size computed in the model')
    parser.add_argument('--temperature', type=int, default=5, help='temperature of infoNCEloss')

    # ABIDE数据集
    parser.add_argument("--source", type=str, default='NYU')
    parser.add_argument('--target', nargs='+', default=['PITT', 'UCLA_1', 'USM', 'YALE'], help='names of target domains')
    parser.add_argument('--data_root', type=str, default='ABIDE/ABIDE-ho', help='path to dataset root')

    # training args
    parser.add_argument('--source_iters', type=int, default=500, help='number of source pre-train iters')
    parser.add_argument('--adapt_iters', type=int, default=500, help='number of iters for a curriculum adaptation')
    parser.add_argument('--test_interval', type=int, default=100, help='interval of two continuous test phase')
    parser.add_argument('--output_dir', type=str, default='test', help='output directory')
    parser.add_argument('--random_state', type=int, default=200)
    parser.add_argument('--ABIDE_in_features', type=int, default=1500, help='number of ABIDE_in_features')


    # optimization args
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--kd_lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--beta', type=float, default=0.1, help='weight of kd loss')
    parser.add_argument('--lambda_node', default=1, type=float, help='gnn loss weight')
    parser.add_argument('--lambda_adv', default=1, type=float, help='adversarial loss weight')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold for pseudo labels')
    parser.add_argument('--ds_threshold', type=float, default=0.92, help='ds threshold for pseudo labels')
    parser.add_argument('--seed', type=int, default=0, help='random seed for training')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    total_fold = args.kfold
    # fix random seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # create train configurations
    args.use_cgct_mask = True  # used in CGCT for pseudo label mask in target datasets


    config = utils.build_config(args)
    # feature selector
    estimator = RidgeClassifier()
    # estimator = LogisticRegression()
    rfe_selector = RFE(estimator, n_features_to_select=args.ABIDE_in_features, step=args.ABIDE_in_features/20, verbose=1)

    dset_loaders = {
        'target_train': {},
        'target_test': {},
    }
    data_config = config["data"]

    #读源域和目标域
    source_data_x, source_data_y, source_data_adj, selected_feature_indices = reader_source(os.path.join(args.data_root, args.source), args.source, rfe_selector)

    # create train and test datasets for a target domain
    target_data = {}
    for dset_name in sorted(data_config['target']['name']):
        target_data_x, target_data_y, target_data_adj = reader_target(os.path.join(args.data_root, dset_name), dset_name, selected_feature_indices)
        target_data[dset_name] = Data(x=target_data_x, y=target_data_y, edge_attr=target_data_adj, name=dset_name)

    source_data = Data(x=source_data_x.to(device), y=source_data_y, edge_attr=source_data_adj)

    dset_loaders['source'] = DataLoader(source_data,  shuffle=True,
                                        num_workers=config['num_workers'], drop_last=True, pin_memory=True)

    target_train = {}
    target_test = {}
    train_index = {}
    test_index = {}

    for dset_name, target_loader in target_data.items():
        [train_index[dset_name], test_index[dset_name]] = networks.Split_Sets_10_Fold(total_fold, target_data[dset_name].x, args)
        x = target_loader.x
        x_train = x[train_index[dset_name][i]]
        y_train = target_loader.y[train_index[dset_name][i]]
        target_train_adj = target_loader.edge_attr[train_index[dset_name][i]]
        x_test = x[test_index[dset_name][i]]
        y_test = target_loader.y[test_index[dset_name][i]]
        target_test_adj = target_loader.edge_attr[test_index[dset_name][i]]


        target_train[dset_name] = Data(x=x_train, y=y_train, edge_attr=target_train_adj, pseudo_labels=None)
        target_test[dset_name] = Data(x=x_test, y=y_test, edge_attr=target_test_adj)
        dset_loaders['target_train'][dset_name] = DataLoader(dataset=target_train[dset_name],
                                                                       shuffle=True,
                                                                       num_workers=config['num_workers'],
                                                                       drop_last=True)

        dset_loaders['target_test'][dset_name] = DataLoader(dataset=target_test[dset_name],
                                                                      num_workers=config['num_workers'],
                                                                      pin_memory=True)

    # 训练集构建一整张图，包含一个源域和所有目标域
    x_data = torch.cat((dset_loaders['source'].dataset.x, dset_loaders['target_train'][args.target[0]].dataset.x,
                        dset_loaders['target_train'][args.target[1]].dataset.x,
                        dset_loaders['target_train'][args.target[2]].dataset.x,
                        dset_loaders['target_train'][args.target[3]].dataset.x), dim=0)
    y_data = torch.cat((dset_loaders['source'].dataset.y, dset_loaders['target_train'][args.target[0]].dataset.y,
                        dset_loaders['target_train'][args.target[1]].dataset.y,
                        dset_loaders['target_train'][args.target[2]].dataset.y,
                        dset_loaders['target_train'][args.target[3]].dataset.y), dim=0)


    adj_all = contrast_adj(x_data)
    mask = torch.zeros(x_data.size(0), 1, dtype=torch.bool)
    source_targets_data = Data(x=x_data, y=y_data, edge_attr=adj_all, mask=mask)
    dset_loaders['source_targets'] = DataLoader(source_targets_data)
    # 测试集构建一整张图，所有目标域
    x_data_test = torch.cat((dset_loaders['target_test'][args.target[0]].dataset.x,
                             dset_loaders['target_test'][args.target[1]].dataset.x,
                             dset_loaders['target_test'][args.target[2]].dataset.x,
                             dset_loaders['target_test'][args.target[3]].dataset.x), dim=0)
    y_data_test = torch.cat((dset_loaders['target_test'][args.target[0]].dataset.y,
                             dset_loaders['target_test'][args.target[1]].dataset.y,
                             dset_loaders['target_test'][args.target[2]].dataset.y,
                             dset_loaders['target_test'][args.target[3]].dataset.y), dim=0)
    adj_all_test = contrast_adj(x_data_test)

    test_data = Data(x=x_data_test, y=y_data_test, edge_attr=adj_all_test)
    dset_loaders['test'] = DataLoader(test_data)

    # set base network
    net_config = config['encoder']
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.to(device)
    # print(base_network)

    # set GCN classifier
    classifier_gnn = GCN(nfeat=256,
                         nhid=args.hidden,
                         nclass=2,
                         dropout=args.dropout)

    classifier_gnn = classifier_gnn.to(device)

    # train on source domain and compute domain inheritability
    log_str = '==> Step 1: Pre-training on the source dataset ...'
    utils.write_logs(config, log_str)

    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)

    log_str = '==> Finished pre-training on source!\n'
    utils.write_logs(config, log_str)

    log_str = '==> Step 2: Curriculum learning ...'
    utils.write_logs(config, log_str)

    ######## Stage 1: find the closest target domain ##########
    temp_test_loaders = dict(dset_loaders['target_test'])

    log_str = '==> Starting the adaptation on {} ...'.format('all targets')
    utils.write_logs(config, log_str)
    ######## Stage 2: adapt to the chosen target domain having the maximum inheritance/similarity ##########
    base_network, classifier_gnn = trainer.adapt_target_kd(config, base_network, classifier_gnn,
                                                        dset_loaders, args)
    log_str = '==> Finishing the adaptation on {}!\n'.format('all targets')
    utils.write_logs(config, log_str)

    mlp_accuracy, target_acc = evaluate(i, config, base_network, dset_loaders['target_test'], dset_loaders)

    MLP_Accuracy.append(mlp_accuracy)
    # 目标域准确率
    target_1_acc.append(target_acc[0])
    target_2_acc.append(target_acc[1])
    target_3_acc.append(target_acc[2])
    target_4_acc.append(target_acc[3])

    log_str = 'Finished training and evaluation!'
    utils.write_logs(config, log_str)

    # save models
    if args.save_models:
        torch.save(base_network.cpu().state_dict(), os.path.join(config['output_path'], 'base_network.pth.tar'))
        torch.save(classifier_gnn.cpu().state_dict(), os.path.join(config['output_path'], 'classifier_gnn.pth.tar'))

    for layer in base_network.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    for layer in classifier_gnn.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

MLP_Accuracy_Avg = sum(MLP_Accuracy) / len(MLP_Accuracy)

target_1_acc = sum(target_1_acc)/len(target_1_acc)
target_2_acc = sum(target_2_acc)/len(target_2_acc)
target_3_acc = sum(target_3_acc)/len(target_3_acc)
target_4_acc = sum(target_4_acc)/len(target_4_acc)


log_str = '10_fold Avg Accuracy MLP Classifier: %.4f'\
          % (MLP_Accuracy_Avg * 100.)

config['out_file'].write(log_str + '\n')
config['out_file'].flush()
print(log_str)
