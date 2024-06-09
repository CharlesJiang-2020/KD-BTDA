import os
import torch
from torch.utils.data import DataLoader

import networks

from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader, Data

def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.01, weight_decay=0.005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i += 1
    return optimizer


def build_config(args):
    config = {
        'method': args.method,
        'ndomains': 2,
        'in_features': args.ABIDE_in_features,
        'output_path': 'results/' + args.output_dir,
        'threshold': args.threshold,
        'ds_threshold': args.ds_threshold,
        'source_iters': args.source_iters,
        'adapt_iters': args.adapt_iters,
        'test_interval': args.test_interval,
        'num_workers': args.num_workers,
        'lambda_node': args.lambda_node,
        'lambda_adv': args.lambda_adv,
        'random_dim': args.rand_proj,
        'use_cgct_mask': args.use_cgct_mask if 'use_cgct_mask' in args else False,
    }
    # backbone params
    config['encoder'] = {
        'name': networks.mlp1,
        'params': {'networks_name': args.encoder,
                   'in_features': args.ABIDE_in_features,
                   },
    }
    # optimizer params
    config['optimizer'] = {
        'type': torch.optim.SGD,
        'optim_params': {
            'lr': args.lr,
             'momentum': 0.9,
             'weight_decay': args.wd,
             'nesterov': True,
            # 'eps': 1e-08
             },
        'lr_type': 'inv',
        'lr_param': {
            'lr': args.lr,
            'gamma': 0.01,
            # 'gamma': 0.1,
            'power': 0.75,
        },
    }
    # kd optimizer params
    config['optimizer_kd'] = {
        'type': torch.optim.SGD,
        'optim_params': {
            'lr': args.kd_lr,
             'momentum': 0.9,
             'weight_decay': args.wd,
             'nesterov': True,
            # 'eps': 1e-08
             },
        'lr_type': 'inv',
        'lr_param': {
            'lr': args.lr,
            'gamma': 0.01,
            # 'gamma': 0.1,
            'power': 0.75,
        },
    }
    # dataset params
    config['dataset'] = args.dataset
    config['data_root'] = args.data_root
    config['data'] = {
        'source': {
            'name': args.source,
            'selected_features': args.ABIDE_in_features,
        },
        'target': {
            'name': args.target,
            'selected_features': args.ABIDE_in_features,
        },
        'test': {
            'name': args.target,
            'selected_features': args.ABIDE_in_features,
        },
    }

    if config['dataset'] == 'ABIDE':
        config['encoder']['params']['class_num'] = 2
        config['data']['series_list_root'] = './code/ABIDE/ABIDE-ho'
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    # create output folder and log file
    if not os.path.exists(config['output_path']):
        os.system('mkdir -p '+config['output_path'])
    config['out_file'] = open(os.path.join(config['output_path'], f'results_ds_pseudo.txt'), 'w')

    # print pout config values
    config['out_file'].write(str(config)+'\n')
    config['out_file'].flush()

    return config

def write_logs(config, log_str):
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
