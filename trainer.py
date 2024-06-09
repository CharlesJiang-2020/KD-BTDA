import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
import transfer_loss
import utils
from KD.base_kd import hinton_distillation, hinton_distillation_sw, hinton_distillation_wo_ce

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(i, config, base_network, target_test_dset_dict, dset_loaders):
    base_network.eval()
    mlp_accuracy_list = []
    for dset_name, test_loader in target_test_dset_dict.items():
        test_res = eval_domain(config, test_loader, base_network, dset_loaders)
        mlp_accuracy = test_res['mlp_accuracy']
        mlp_accuracy_list.append(mlp_accuracy)
        # print out test accuracy for domain
        log_str = 'Dataset:%s\tTest Accuracy mlp %.4f'\
                  % (dset_name, mlp_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)

    # print out domains averaged accuracy
    mlp_accuracy_avg = sum(mlp_accuracy_list) / len(mlp_accuracy_list)

    log_str = 'iter: %d, Avg Accuracy MLP Classifier: %.4f'\
              % (i, mlp_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
    base_network.train()

    return mlp_accuracy_avg, mlp_accuracy_list


def eval_domain(config, test_loader, base_network, dset_loaders):
    graph_nodes = dset_loaders['test'].dataset.x.to(DEVICE)
    indices_test = networks.find_indices_in_graph(test_loader.dataset.x, graph_nodes)
    with torch.no_grad():
        inputs = graph_nodes.to(DEVICE)
        feature, logits_mlp = base_network(inputs)
        labels = test_loader.dataset.y.cpu()

    # predict class labels
    _, predict_mlp = torch.max(logits_mlp[indices_test].cpu(), 1)
    # 准确率(Accuracy)
    mlp_accuracy = torch.sum(predict_mlp == labels).item() / len(labels)

    out = {
        'mlp_accuracy': mlp_accuracy
    }
    return out


def train_source(config, base_network, classifier_gnn, dset_loaders):
    torch.autograd.set_detect_anomaly(True)
    # define loss functions
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() +\
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    classifier_gnn.train()
    # len_train_source = len(dset_loaders["source"])
    for i in range(config['source_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()


        graph_nodes = dset_loaders['source_targets'].dataset.x.to(DEVICE)
        graph_edge = dset_loaders['source_targets'].dataset.edge_attr.to(DEVICE)
        # 找到 inputs_target 在graph中的索引  存储每个目标域样本在 graph_nodes 中的索引
        indices_source = networks.find_indices_in_graph(dset_loaders['source'].dataset.x, graph_nodes)
        # 创建一个掩码，将要暂时不使用的部分标记为False
        mask = dset_loaders['source_targets'].dataset.mask.clone()
        mask[indices_source] = True
        features, logits_mlp = base_network(graph_nodes)

        # 避免就地修改 features 张量
        features = features.clone()
        features[~mask.squeeze(1)] = 0.0

        # get input data
        #所有数据
        labels_source = dset_loaders['source'].dataset.y.to(DEVICE) #源域标签

        logits_gnn = classifier_gnn(features, graph_edge)
        gnn_loss = ce_criterion(logits_gnn[indices_source], labels_source)#取前175即源域数据计算损失
        mlp_loss = ce_criterion(logits_mlp[indices_source], labels_source)

        # total loss and backpropagation
        loss = mlp_loss + gnn_loss
        loss.backward()
        optimizer.step()

        # printout train loss
        if i % 20 == 0 or i == config['source_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tGCN loss:%.4f\t MLP loss:%.4f' % (i,
                  config['source_iters'], gnn_loss.item(), mlp_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, dset_loaders['target_test'], dset_loaders)


    return base_network, classifier_gnn


def adapt_target_kd(config, base_network, classifier_gnn, dset_loaders, args):
    # define loss functions
    ce_criterion = nn.CrossEntropyLoss()

    # add random layer and adversarial network
    class_num = config['encoder']['params']['class_num']
    random_layer = networks.RandomLayer([base_network.output_num(), class_num], config['random_dim'], DEVICE)

    adv_net = networks.AdversarialNetwork(config['random_dim'], config['random_dim'], config['ndomains'])

    random_layer.to(DEVICE)
    adv_net = adv_net.to(DEVICE)

    # configure optimizer
    optimizer_config = config['optimizer']
    optimizer_kd = config['optimizer_kd']
    parameter_list = base_network.get_parameters() + adv_net.get_parameters() \
                     + [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    optimizer_kd = optimizer_kd['type'](parameter_list, **(optimizer_kd['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    classifier_gnn.train()
    adv_net.train()
    random_layer.train()

    for i in range(config['adapt_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        optimizer_kd = utils.inv_lr_scheduler(optimizer_kd, i, **schedule_param)
        optimizer_kd.zero_grad()
        # get input data
        batch_target = dset_loaders['target_train']
        # get input data
        inputs_source = dset_loaders['source'].dataset.x.to(DEVICE)
        inputs_target = torch.cat((batch_target[args.target[0]].dataset.x, batch_target[args.target[1]].dataset.x,
                                   batch_target[args.target[2]].dataset.x,  batch_target[args.target[3]].dataset.x), dim=0)
        labels_source = dset_loaders['source'].dataset.y.to(DEVICE)
        domain_source = torch.zeros(dset_loaders['source'].dataset.y.size()).to(DEVICE)

        domain_target = torch.ones(inputs_target.size()[0]).to(DEVICE)
        domain_input = torch.cat([domain_source, domain_target], dim=0)

        # 整个graph
        graph_nodes = dset_loaders['source_targets'].dataset.x.to(DEVICE)
        graph_edge = dset_loaders['source_targets'].dataset.edge_attr.to(DEVICE)

        # 创建一个掩码，将使用的部分标记为True
        mask = dset_loaders['source_targets'].dataset.mask.clone()

        # 找到 inputs_target 在graph中的索引  存储每个目标域样本在 graph_nodes 中的索引
        indices_source = networks.find_indices_in_graph(inputs_source, graph_nodes)
        indices_target = networks.find_indices_in_graph(inputs_target, graph_nodes)

        # 创建一个和 mask 大小相同的索引张量
        indices_tensor = torch.tensor(indices_source + indices_target, dtype=torch.long)
        mask[indices_tensor] = True

        # 大graph
        features, logits_mlp = base_network(graph_nodes)
        features = features.clone()
        features[~mask.squeeze(1)] = 0.0

        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn = classifier_gnn(features, graph_edge)
        gnn_loss = ce_criterion(logits_gnn[indices_source], labels_source)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp_source = base_network(inputs_source)
        features_target, logits_mlp_target = base_network(inputs_target)

        features = torch.cat((features_source, features_target), dim=0)

        # mlp_loss = ce_criterion(logits_mlp_source, labels_source)
        trans_loss = transfer_loss.CDAN(config['ndomains'], [features[mask.any(dim=1)], F.softmax(logits_gnn)[mask.any(dim=1)]],
                                        adv_net, None, None, random_layer, domain_input)

        # kd loss
        alpha = args.alpha
        beta = torch.tensor(args.beta).unsqueeze(0).to(DEVICE)
        T = 20
        teacher_source_logits = logits_gnn[indices_source]
        teacher_target_logits = logits_gnn[indices_target]

        student_source_logits = logits_mlp_source
        student_target_logits = logits_mlp_target


        source_kd_loss = hinton_distillation_sw(teacher_source_logits, student_source_logits, labels_source, T,
                                                alpha).abs()
        target_kd_loss = hinton_distillation_wo_ce(teacher_target_logits, student_target_logits, T).abs()

        kd_loss = beta * (target_kd_loss + source_kd_loss)
        # total loss and backpropagation
        loss = config['lambda_adv'] * trans_loss + config['lambda_node'] * gnn_loss + kd_loss

        loss.backward()
        optimizer.step()
        optimizer_kd.step()

        # printout train loss
        if i % 20 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\t GNN Loss: %.4f\t Transfer loss:%.4f\t KD loss:%.4f' % (
                i, config["adapt_iters"],  config['lambda_node'] * gnn_loss.item(),
                config['lambda_adv'] * trans_loss.item(),
                kd_loss.item()
            )

            utils.write_logs(config, log_str)

        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, dset_loaders['target_test'], dset_loaders)

    return base_network, classifier_gnn

