import copy
import gc
import os
import pickle

import scipy.sparse as sp
from scipy.sparse import csr_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import statistics
import argparse

import optuna
from torch.utils.hipify.hipify_python import str2bool
from torch_geometric.utils import train_test_split_edges, degree
from torch_geometric.data import Data
from lib import eval_node_cls, setup_seed, loss_means, get_lr_schedule_by_sigmoid, get_ep_data, eval_edge_pred, loss_fn, \
    triplet_margin_loss
from models_ACGA import GCN_Net
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from spliter import louvain, random_spliter, origin, rand_walk

import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset', help='Cora, Citeseer or Pubmed. Default=Cora', default='Cora')
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=200)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=128)
    parser.add_argument('--emb_size', type=int, help='gae/vgae embedding size', default=32)
    parser.add_argument('--spliter', help='spliter method.Default=random_spliter (rand_walk)', default="random_spliter")
    parser.add_argument('--gae', type=str2bool, help='whether use GAE ', default=True)
    parser.add_argument('--use_bns', type=str2bool, help='whether use bns for GNN layer ', default=True)
    parser.add_argument('--task', type=int, help='node cls = 0, edge predict = 1', default=0)

    return parser


def train_rep(model, data, num_classes, alpha=0.5, beta=3, gamma=2, train_edge=None, new_label=None):
    model.train()
    batch = data.to(device)
    if isinstance(new_label, np.ndarray):
        label_all = new_label
    else:
        label_all = batch.y
    # alpha,beta,gamma,epoch,acc
    alpha = alpha  # MAX: Cora: 0.5,3,2,300,83.6,    0.58,  0.65,  3.4000,
    beta = beta  # 2
    gamma = gamma  # 2
    if train_edge is not None:
        adj = train_edge.train_pos_edge_index
        summary, summary_pos, summary_neg, loss_co = model.train_present(batch, label_all, adj)
    else:
        summary, summary_pos, summary_neg, loss_co = model.train_present(batch, label=label_all)
    loss_s = loss_means(summary, label_all, num_classes)
    loss_cl = torch.nn.functional.triplet_margin_loss(summary, summary_pos, summary_neg, reduction='mean')
    loss = beta * ((1 - alpha) * loss_cl + alpha * loss_co) + loss_s * gamma

    return loss


def train_cls(model, data, args, criterion, optimizer, epoch):
    model.train()
    batch = data.to(device)
    label_all = batch.y
    predict = model(batch)
    # predict = model.forward_emb(batch, embedding)
    if data.x.size(0) == args.subgraph_size:
        loss_nc = criterion(predict, label_all)
    else:
        predict = predict[batch.train_mask]
        label = batch.y[batch.train_mask]
        loss_nc = criterion(predict, label)

    return loss_nc


def test_cls(model, data):
    r"""Evaluates latent space quality via a logistic regression downstream task."""
    model.eval()
    # criterion = F.CrossEntropyLoss()
    batch = data
    predict = model(batch)
    # predict = model.test(batch)
    gcn_val_z = predict[batch.val_mask]
    gcn_test_z = predict[batch.test_mask]
    val_y = batch.y[batch.val_mask]
    test_y = batch.y[batch.test_mask]
    gcn_val_acc = eval_node_cls(gcn_val_z, val_y)
    gcn_test_acc = eval_node_cls(gcn_test_z, test_y)
    return gcn_val_acc, gcn_test_acc


def train_ep(model, data, train_edge, adj_m, norm_w, pos_weight, optimizer, args, wight):
    model.train()
    kl_divergence = 0
    batch = data.to(device)
    adj = train_edge.train_pos_edge_index
    adj_logit = model(batch, edge=adj)
    loss_nc = norm_w * F.binary_cross_entropy_with_logits(adj_logit.view(-1), adj_m.view(-1), pos_weight=pos_weight)
    loss_nc = loss_nc - kl_divergence

    return loss_nc


def test_ep(model, data, train_edge):
    model.eval()
    adj = train_edge.train_pos_edge_index
    adj_logit = model(data, adj)

    val_edges = torch.cat((train_edge.val_pos_edge_index, train_edge.val_neg_edge_index), axis=1).cpu().numpy()
    val_edge_labels = np.concatenate(
        [np.ones(train_edge.val_pos_edge_index.size(1)), np.zeros(train_edge.val_neg_edge_index.size(1))])

    test_edges = torch.cat((train_edge.test_pos_edge_index, train_edge.test_neg_edge_index), axis=1).cpu().numpy()
    test_edge_labels = np.concatenate(
        [np.ones(train_edge.test_pos_edge_index.size(1)), np.zeros(train_edge.test_neg_edge_index.size(1))])

    adj_pred = adj_logit.cpu()
    ep_auc, ep_ap = eval_edge_pred(adj_pred, val_edges, val_edge_labels)
    # print(f'EPNet train: auc {ep_auc:.4f}, ap {ep_ap:.4f}')

    ep_auc_test, ep_ap_test = eval_edge_pred(adj_pred, test_edges, test_edge_labels)
    # print(f'EPNet train,Final: auc {ep_auc_test:.4f}, ap {ep_ap:.4f}')

    return ep_auc, ep_ap, ep_auc_test, ep_ap_test


spliter_dict = {
    'louvain': louvain,
    'random_spliter': random_spliter,
    'origin': origin,
    'rand_walk': rand_walk,
    'nothing': lambda x: x  # Identity function
}


def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx


def main(trial=None, train_edge=None):
    parser = get_parser()
    try:
        args = parser.parse_args()
    except 'parser error':
        exit()
    print(args)
    setup_seed(1024)
    tvt_nids = pickle.load(open(f'./data_new/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'./data_new/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'./data_new/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'./data_new/graphs/{args.dataset}_labels.pkl', 'rb'))
    if not isinstance(features, torch.Tensor):
        if isinstance(features, csr_matrix):
            features = features.toarray()
        features = torch.from_numpy(features).type(torch.float32)

    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).type(torch.int64)
    setup_seed(1024)

    if not isinstance(adj_orig, sp.coo_matrix):
        adj_orig = sp.coo_matrix(adj_orig)
    adj_orig.setdiag(1)
    adj_orig = scipysp_to_pytorchsp(adj_orig).to_dense()
    adj_orig = adj_orig.to_sparse().indices()
    num_classes = torch.unique(labels).size(0)
    data = Data(x=features, edge_index=adj_orig, y=labels, train_mask=tvt_nids[0], val_mask=tvt_nids[1],
                test_mask=tvt_nids[2], num_classes=num_classes)
    # data = Planetoid(root='./dataset/' + args.dataset, name=args.dataset)
    data.train_mask, data.val_mask, data.test_mask = tvt_nids
    num_classes = data.num_classes
    feature_size = data.x.size(1)
    data = data.to(device)
    print('Start training !!!')
    val_acc_list, test_acc_list = [], []
    n_epochs = 10000
    if num_classes == 2:
        nc_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        nc_criterion = torch.nn.CrossEntropyLoss()
    num = 11
    model = GCN_Net(feature_size,
                    num_classes,
                    hidden=args.hidden_size,
                    emb_size=args.emb_size,
                    dropout=0.5,
                    gae=args.gae,
                    use_bns=args.use_bns,
                    task=args.task).to(device)
    new_label, adj_m, norm_w, pos_weight, train_edge = get_ep_data(data.cpu(), args)
    if args.task == 1:
        adj_m, pos_weight, train_edge = [x.to(device) for x in [adj_m, pos_weight, train_edge]]
        val_ap_list, test_ap_list = [], []
    if trial is not None:
        alpha = trial.suggest_discrete_uniform('alpha', 0, 1, 0.01)
        beta = trial.suggest_discrete_uniform('beta', 0, 10, 0.05)
        gamma = trial.suggest_discrete_uniform('gama', 0, 5, 0.05)
    else:
        alpha, beta, gamma = 0.69, 8.85, 3.6
    for weight in range(1, num):
        if args.task == 0:
            lr, weight_decay = 1e-2, 5e-4  # , 5e-4  # , 5e-4
            best_val_acc, last_test_acc, early_stop, patience = 0, 0, 0, 200  # cora 100, cite-seer 250
            model.reset_parameters()

            optimizer_cls = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            for epoch in range(n_epochs):  # n_epochs,800
                rep_loss = train_rep(model, data, 2, alpha=alpha, beta=beta, gamma=gamma, new_label=new_label)
                # rep_loss = 0
                cls_loss = train_cls(model, data, args, nc_criterion, optimizer_cls, epoch)
                loss = rep_loss + cls_loss
                loss.backward()
                optimizer_cls.step()
                optimizer_cls.zero_grad()
                with torch.no_grad():
                    val_acc, test_acc = test_cls(model, data)
                    # print(f"epoch {epoch} val acc = {val_acc}, test_acc = {test_acc}")
                if val_acc > best_val_acc:
                    best_val_acc, last_test_acc, early_stop = val_acc, test_acc, 0
                else:
                    early_stop += 1
                if early_stop >= patience:
                    break
            print(f'num = {weight}, best_val_acc = {best_val_acc * 100:.1f}%, '
                  f'last_test_acc = {last_test_acc * 100:.1f}%')
        else:
            lr, weight_decay = 1e-2, 5e-4  # 5e-4
            best_val_acc, best_val_ap, last_test_acc, last_test_ap, early_stop, patience = 0, 0, 0, 0, 0, 200
            # setup_seed(1024)
            model.reset_parameters()
            optimizer_ep = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            # node pre 128,32,edge pre 32,16
            for epoch in range(n_epochs):
                rep_loss = train_rep(model, data, num_classes, alpha=alpha, beta=beta, gamma=gamma,
                                     train_edge=train_edge, new_label=new_label)
                ep_loss = train_ep(model, data, train_edge, adj_m, norm_w, pos_weight, optimizer_ep, args, weight)
                loss = rep_loss + ep_loss
                loss.backward()
                optimizer_ep.step()
                optimizer_ep.zero_grad()
                with torch.no_grad():
                    val_acc, val_ap, test_acc, test_ap = test_ep(model, data, train_edge)
                if val_acc > best_val_acc:
                    best_val_acc, best_val_ap, last_test_acc, last_test_ap, early_stop, = val_acc, val_ap, test_acc, \
                                                                                          test_ap, 0
                else:
                    early_stop += 1
                if early_stop >= patience:
                    break
            print(f'wight = {weight}, best_val_auc = {best_val_acc * 100:.1f}%, best_val_ap = {best_val_ap * 100:.1f}% '
                  f'last_test_auc = {last_test_acc * 100:.1f}%, last_test_ap = {last_test_ap * 100:.1f}%')
        val_acc_list.append(best_val_acc)
        test_acc_list.append(last_test_acc)
        if args.task == 1:
            val_ap_list.append(best_val_ap)
            test_ap_list.append(last_test_ap)
    avg_val_acc = statistics.mean(val_acc_list)
    avg_test_acc = statistics.mean(test_acc_list)
    std_acc = np.std(np.array(test_acc_list))
    if args.task == 1:
        avg_val_ap = statistics.mean(val_ap_list)
        avg_test_ap = statistics.mean(test_ap_list)
        std_ap = np.std(np.array(test_ap_list))
        print(f'train num:{num - 1}, avg val auc: {avg_val_acc * 100:.1f}%, avg val ap:{avg_val_ap * 100:.1f}%, '
              f'avg test auc: {avg_test_acc * 100:.1f}%, avg test ap:{avg_test_ap * 100:.1f} '
              f'std auc:{std_acc * 100:.2f}, std ap:{std_ap * 100:.2f},')
    else:
        print(f'train num:{num - 1}, avg val acc: {avg_val_acc * 100:.1f}%, '
              f'avg test acc: {avg_test_acc * 100:.1f}%, std acc:{std_acc * 100:.2f},')
    del data, model
    gc.collect()
    torch.cuda.empty_cache()
    return avg_test_acc


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=500)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # main()
