import argparse
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import avg_pool_neighbor_x
from torch_geometric.utils import degree, train_test_split_edges


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score


def eval_node_cls(nc_logits, labels):
    """ evaluate node classification results """
    if len(labels.size()) == 2:
        preds = torch.round(torch.sigmoid(nc_logits))
        tp = len(torch.nonzero(preds * labels))
        tn = len(torch.nonzero((1 - preds) * (1 - labels)))
        fp = len(torch.nonzero(preds * (1 - labels)))
        fn = len(torch.nonzero((1 - preds) * labels))
        pre, rec, f1 = 0., 0., 0.
        if tp + fp > 0:
            pre = tp / (tp + fp)
        if tp + fn > 0:
            rec = tp / (tp + fn)
        if pre + rec > 0:
            f_measure = (2 * pre * rec) / (pre + rec)
    else:
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        f_measure = correct.item() / len(labels)
    return f_measure


def loss_fn(hidden1, summary1):
    r"""Computes the margin objective."""
    marginloss_fn = nn.MarginRankingLoss(0.5, reduction='none')
    shuf_index = torch.randperm(summary1.size(0))

    hidden2 = hidden1[shuf_index]
    summary2 = summary1[shuf_index]

    logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
    logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
    logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
    logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))
    TotalLoss = 0.0
    ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
    TotalLoss += marginloss_fn(logits_aa, logits_ba, ones)
    TotalLoss += marginloss_fn(logits_bb, logits_ab, ones)

    return TotalLoss


def loss_fn(hidden1, data):
    r"""Computes the margin objective."""
    datac = copy.copy(data)
    datac.x = hidden1
    summary1 = avg_pool_neighbor_x(datac).x

    marginloss_fn = nn.MarginRankingLoss(0.5, reduction='none')
    shuf_index = torch.randperm(summary1.size(0))

    hidden2 = hidden1[shuf_index]
    summary2 = summary1[shuf_index]

    logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
    logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
    logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
    logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))
    TotalLoss = 0.0
    ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
    TotalLoss += marginloss_fn(logits_aa, logits_ba, ones)
    TotalLoss += marginloss_fn(logits_bb, logits_ab, ones)

    return TotalLoss


def loss_cal(x, x_aug):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.clamp(torch.einsum('i,j->ij', x_abs, x_aug_abs), min=1e-5)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


def triplet_margin_loss(anchor, positive, negative, w=0.5, margin=1.0, p=2, eps=1e-06, swap=False, reduction='mean'):
    # 计算两两之间的距离
    d_ap = torch.norm(anchor - positive, p=p, dim=-1)  # (N,)
    d_an = torch.norm(anchor - negative, p=p, dim=-1)  # (N,)
    d_pn = torch.norm(positive - negative, p=p, dim=-1)  # (N,)
    # 如果swap为True，交换d_an和d_pn中的最小值
    if swap:
        d_an = torch.min(d_an, d_pn)
    # 计算损失函数
    loss = torch.clamp(d_ap - w * d_an + margin, min=0.0)  # (N,)
    # 根据reduction参数返回结果x
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError('Invalid reduction: {}'.format(reduction))


def loss_cal2(x, x_aug, temperature=0.2, sym=True):
    # x and x_aug shape -> Batch x proj_hidden_dim

    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - torch.log(loss_1).mean()
        return loss_1

    return loss


def loss_com(summary, label):
    label = torch.tensor(label).to(device=summary.device)
    x = summary[label == 0]
    # 选择属于第二类的tensor
    x_aug = summary[label == 1]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    T = 0.2
    batch_size, _ = x.size()
    sim_matrix = torch.einsum('ik,jk->ij', x, x) / torch.clamp(torch.einsum('i,j->ij', x_abs, x_abs), min=1e-5)
    sim_matrix_neg = torch.einsum('ik,jk->ij', x, x_aug) / torch.clamp(torch.einsum('i,j->ij', x_abs, x_aug_abs),
                                                                       min=1e-5)
    sim_matrix = torch.exp(sim_matrix / T)
    sim_matrix_neg = torch.exp(sim_matrix_neg / T)
    # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss_pos = sim_matrix.mean(dim=1) / (sim_matrix_neg.sum(dim=1))
    loss_pos = - torch.log(loss_pos).mean()

    batch_size, _ = x_aug.size()
    sim_matrix = torch.einsum('ik,jk->ij', x_aug, x_aug) / torch.clamp(torch.einsum('i,j->ij', x_aug_abs, x_aug_abs),
                                                                       min=1e-5)
    sim_matrix_neg = torch.einsum('ik,jk->ij', x_aug, x) / torch.clamp(torch.einsum('i,j->ij', x_aug_abs, x_abs),
                                                                       min=1e-5)
    sim_matrix = torch.exp(sim_matrix / T)
    sim_matrix_neg = torch.exp(sim_matrix_neg / T)
    # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss_neg = sim_matrix.mean(dim=1) / (sim_matrix_neg.sum(dim=1))
    loss_neg = - torch.log(loss_neg).mean()

    loss = loss_pos + loss_neg

    return loss


def loss_compare(summary, label):
    # datac = copy.copy(data)
    # datac.x = summary
    # summary = avg_pool_neighbor_x(datac).x
    # 选择属于第一类的tensor
    label = torch.tensor(label).to(device=summary.device)
    # x1 = torch.masked_select(summary, label == 0).reshape(-1, 128)
    x1 = summary[label == 0].mean(dim=-1).unsqueeze(-1)
    # 选择属于第二类的tensor
    x2 = summary[label == 1].mean(dim=-1).unsqueeze(-1)

    pos_p = torch.exp(torch.mm(x1, x1.transpose(0, 1))).mean(dim=-1)
    pos_n = torch.exp(torch.mm(x1, x2.transpose(0, 1))).mean(dim=-1)
    loss_pos = torch.log(pos_p / pos_n).mean()
    neg_p = torch.exp(torch.mm(x2, x2.transpose(0, 1))).mean(dim=-1)
    neg_n = torch.exp(torch.mm(x2, x1.transpose(0, 1))).mean(dim=-1)
    loss_neg = torch.log(neg_p / neg_n).mean()
    loss = (loss_pos + loss_neg) * label.size(0)
    # mse = F.mse_loss
    # tr_loss = F.triplet_margin_loss
    # sff_index_a = torch.randperm(x1.size(0)).to(device=summary.device)
    # sff_index_b = torch.randperm(x2.size(0)).to(device=summary.device)
    #
    # sff_index_c = torch.randint(0, x1.size(0), (x2.size(0),)).to(device=summary.device)
    # sff_index_d = torch.randint(0, x2.size(0), (x1.size(0),)).to(device=summary.device)
    #
    # y1 = x1[sff_index_a]
    # y2 = x2[sff_index_b]
    #
    # z1 = x1[sff_index_c]
    # z2 = x2[sff_index_d]
    # x_pos = torch.cat((y1, y2), dim=0)
    # x_neg = torch.cat((z2, z1), dim=0)
    # pdist = nn.PairwiseDistance(p=2)
    # sim_pos = torch.exp(pdist(x_new, x_pos))
    # sim_neg = torch.exp(pdist(x_new, x_neg))
    # loss = torch.log(sim_pos/sim_neg).mean()
    # loss = tr_loss(x_new, x_pos, x_neg, reduction='mean')
    # loss = mse(x_new, x_pos)
    return loss


def loss_means(summary, label_all, num_classes):
    r"""Computes the margin objective."""
    std = 0
    for i in range(num_classes):
        node = summary[label_all == i]
        if node.shape[0] != 0:
            std += node.std()
    return std


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    """ schedule the learning rate with the sigmoid function.
    The learning rate will start with near zero and end with near lr """
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    # range the factors to [0, 1]
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule


def compute_loss_para(adj):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
            adj.shape[0]
            * adj.shape[0]
            / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_ep_data(data, args):
    # new_label = None
    # a = data.edge_index
    if args.task:
        train_rate = 0.85
        val_ratio = (1 - train_rate) / 3
        test_ratio = (1 - train_rate) / 3 * 2
        train_edge = train_test_split_edges(copy.copy(data), val_ratio, test_ratio)
        adj = train_edge.train_pos_edge_index.cuda()
        adj_m = np.zeros((data.x.size(0), data.x.size(0)))
        for i in range(adj.size(1)):
            adj_m[adj[0][i]][adj[1][i]] = 1
            adj_m[adj[1][i]][adj[0][i]] = 1
        adj_m = torch.from_numpy(adj_m)
        # pos_weight, norm_w = compute_loss_para(adj_m)
        norm_w = adj_m.shape[0] ** 2 / float((adj_m.shape[0] ** 2 - adj_m.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_m.shape[0] ** 2 - adj_m.sum()) / adj_m.sum()])
        new_adj = adj_m.to_sparse().indices()
        # if args.task == 1:
        degree_count = degree(new_adj[0]).numpy()
        flag = np.median(degree_count)
        # 将 degree_count 中与 flag 的关系进行值替换的位置替换为随机数组中的对应值
        # new_label = np.where(degree_count == flag, random_values, degree_count)
        # new_label = np.where(degree_count == flag, np.random.choice([0, 1], size=len(degree_count)),
        #                      np.where(degree_count < flag, 0, 1))
        new_label = np.where(degree_count < flag, 0, 1)
        return new_label, adj_m, norm_w, pos_weight, train_edge
    else:
        degree_count = degree(data.edge_index[0], data.num_nodes).numpy()
        # unique_elements, counts = np.unique(degree_count, return_counts=True)
        # print(f'avg degree:{np.mean(unique_elements)}')
        # print(f'unique_elements:{unique_elements},counts:{counts}')
        flag = np.median(degree_count)
        # 将 degree_count 中与 flag 的关系进行值替换的位置替换为随机数组中的对应值
        # new_label = np.where(degree_count == flag, random_values, degree_count)
        # new_label = np.where(degree_count == flag, np.random.choice([0, 1], size=len(degree_count)),
        #                      np.where(degree_count < flag, 0, 1))
        new_label = np.where(degree_count < flag, 0, 1)
        return new_label


