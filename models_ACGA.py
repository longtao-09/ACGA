import random
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import Linear
from torch import Tensor
from scipy.spatial.distance import cdist
import numpy as np
from itertools import combinations
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
import scipy.sparse as sp
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor, fill_diag, matmul, mul

from lib import eval_edge_pred, setup_seed


class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 emb_size=32,
                 dropout=.0,
                 gae=True,
                 use_bns=True,
                 task=0):
        super(GCN_Net, self).__init__()
        self.gae = gae
        self.use_bns = use_bns
        self.task = task
        self.cons = GCNLayer(input_dim=in_channels, output_dim=hidden, activation=F.relu, dropout=dropout)
        self.conv_pos = VGAE(input_dim=in_channels, output_dim=emb_size, dim_z=hidden, dropout=dropout, gae=self.gae)
        self.conv_neg = VGAE(input_dim=in_channels, output_dim=emb_size, dim_z=hidden, dropout=dropout, gae=self.gae)
        self.bns = torch.nn.BatchNorm1d(hidden)
        self.dropout = dropout
        self.graph_pos_edge, self.graph_neg_edge = None, None
        self.pos_adj_norm, self.neg_adj_norm = None, None
        self.pos_weight, self.neg_weight = None, None
        self.adj_pos, self.adj_neg, self.adj = None, None, None
        if self.task == 0:
            self.cls = GCNLayer(input_dim=hidden, output_dim=out_channels, activation=0, dropout=0)
        else:
            self.ep = GCNLayer(input_dim=hidden, output_dim=emb_size, activation=0, dropout=0, ep=self.task)

    def reset_parameters(self):
        self.cons.init_params()
        self.conv_neg.init_params()
        self.conv_pos.init_params()

        if self.task == 0:
            self.cls.init_params()
        else:
            self.ep.init_params()

        self.bns.reset_parameters()

    def train_present(self, data, label=None, edge=None, kl_beta=None):
        if edge is not None:
            input_x, edge_index, label = data.x, edge, label
        elif isinstance(data, Data):
            input_x, edge_index, label = data.x, data.edge_index, label
        elif isinstance(data, tuple):
            input_x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        if self.graph_pos_edge is None:
            x_embedding = input_x
            self.graph_pos_edge, self.graph_neg_edge = self.generate_graph_smi_new(x_embedding, label)
            self.adj_pos = self.preprocess_graph(self.graph_pos_edge, input_x.size(0))
            self.adj_neg = self.preprocess_graph(self.graph_neg_edge, input_x.size(0))

        if self.adj is None:
            self.adj = self.preprocess_graph_new(edge_index, input_x.size(0))

        x_orig = input_x
        x_orig = self.cons(x_orig, self.adj)
        if self.use_bns:
            x_orig = self.bns(x_orig)
            x_orig = F.relu(F.dropout(x_orig, p=self.dropout, training=self.training))

        x_pos = input_x
        x_pos_a = self.conv_pos(x_pos, self.adj)
        x_pos_b = self.conv_pos(x_pos, self.adj_pos)
        x_pos = torch.cat((x_pos_a, x_pos_b), dim=-1)
        if self.use_bns:
            x_pos = self.bns(x_pos)
            x_pos = F.relu(F.dropout(x_pos, p=self.dropout, training=self.training))

        x_neg = input_x
        x_neg_a = self.conv_neg(x_neg, self.adj)
        x_neg_b = self.conv_neg(x_neg, self.adj_neg)
        x_neg = torch.cat((x_neg_a, x_neg_b), dim=-1)

        if self.use_bns:
            x_neg = self.bns(x_neg)

            x_neg = F.relu(F.dropout(x_neg, p=self.dropout, training=self.training))

        loss_co, x_pos_new, x_neg_new = self.loss_co(input_x, x_pos, x_neg, kl_beta=kl_beta)
        return x_orig, x_pos, x_neg, loss_co

    def forward(self, data, edge=None):
        if edge is not None:
            input_x, edge_index = data.x, edge
        elif isinstance(data, Data):
            input_x, edge_index, label = data.x, data.edge_index, data.y
        elif isinstance(data, tuple):
            input_x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        if self.adj is None:
            # self.adj = self.preprocess_graph(edge_index, input_x.size(0))
            self.adj = self.preprocess_graph_new(edge_index, input_x.size(0))
        x_orig = input_x
        x_orig = self.cons(x_orig, self.adj)
        # x_orig = F.relu(F.dropout(x_orig, p=self.dropout, training=self.training))
        if self.task == 0:
            cls_x = self.cls(x_orig, self.adj)
        else:
            cls_x = self.ep(x_orig, self.adj)
        return cls_x

    def forward_emb(self, data, embedding):
        if isinstance(data, Data):
            input_x, edge_index = embedding, data.edge_index
        elif isinstance(data, tuple):
            input_x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        cls_x = self.cls(input_x, edge_index)
        return cls_x  # .log_softmax(dim=-1)

    def loss_co(self, input_x, x_pos, x_neg, pos_weight=None, neg_weight=None, kl_beta=None):
        smi_pos = torch.mm(x_pos, x_pos.transpose(0, 1))
        supports_pos = F.softmax(F.relu(smi_pos), dim=1)
        a_pos = self.cons(input_x, supports_pos, model=2)
        if self.use_bns:
            a_pos = self.bns(a_pos)
        # a_pos = F.dropout(F.relu(a_pos), p=self.dropout, training=self.training)
        a_pos = F.relu(F.dropout(a_pos, p=self.dropout, training=self.training))

        if pos_weight is None:
            pos_weight, pos_norm_w = self.get_pos_weight(self.graph_pos_edge)
        pos_weight, pos_norm_w = pos_weight.to(a_pos.device), pos_norm_w
        loss_pos = F.binary_cross_entropy_with_logits(smi_pos.view(-1), self.graph_pos_edge.to(a_pos.device).view(-1),
                                                      pos_weight=pos_weight)
        if not self.gae:
            kl_pos = self._kl_divergence(self.conv_pos)

        smi_neg = torch.mm(x_neg, x_neg.transpose(0, 1))
        supports_neg = F.softmax(F.relu(smi_neg), dim=1)
        a_neg = self.cons(input_x, supports_neg, model=2)
        if self.use_bns:
            a_neg = self.bns(a_neg)
        # a_neg = F.dropout(F.relu(a_neg), p=self.dropout, training=self.training)
        a_neg = F.relu(F.dropout(a_neg, p=self.dropout, training=self.training))

        if neg_weight is None:
            neg_weight, neg_norm_w = self.get_pos_weight(self.graph_neg_edge)
        neg_weight, neg_norm_w = neg_weight.to(a_neg.device), neg_norm_w
        loss_neg = F.binary_cross_entropy_with_logits(smi_neg.view(-1), self.graph_neg_edge.to(a_neg.device).view(-1),
                                                      pos_weight=neg_weight)
        loss = pos_norm_w * loss_pos + neg_norm_w * loss_neg

        if not self.gae:
            kl_neg = self._kl_divergence(self.conv_neg)
            kl_divergence = kl_pos + kl_neg
            if kl_beta is not None:
                loss = loss - kl_beta * kl_divergence
            else:
                loss = loss - kl_divergence
        return loss, a_pos, a_neg

    @staticmethod
    def get_pos_weight(adj):
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm_w = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor, norm_w
        # return pos_weight, norm_w

    @staticmethod
    def _kl_divergence(model):
        mu = model.means
        lgstd = model.logists
        num_node = mu.size(0)
        kl_divergence = 0.5 / num_node * (1 + 2 * lgstd - mu ** 2 - torch.exp(lgstd) ** 2).sum(1).mean()
        model.means = None
        model.logists = None
        return kl_divergence

    def generate_graph_random(self, x):
        pos_mat = torch.rand(x.shape[0], x.shape[0]).cuda()
        neg_mat = torch.rand(x.shape[0], x.shape[0]).cuda()

        deg = torch.sum(pos_mat, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch.mul(pos_mat, deg_inv_sqrt.view(-1, 1))
        adj_pos = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))

        deg = torch.sum(neg_mat, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch.mul(neg_mat, deg_inv_sqrt.view(-1, 1))
        adj_neg = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return pos_mat, neg_mat, adj_pos.to(x.device), adj_neg.to(x.device)

    @staticmethod
    def generate_graph_smi_new(x, y):
        node_feature = x.cpu().detach().numpy()
        simi_mat = cdist(node_feature, node_feature, 'euclidean')
        np.fill_diagonal(simi_mat, np.max(simi_mat) + 1)
        node_nns = np.argsort(simi_mat, axis=1)

        number = 1
        num_nodes = node_feature.shape[0]
        pos_graph = torch.eye(num_nodes, num_nodes).cuda()
        neg_graph = torch.eye(num_nodes, num_nodes).cuda()
        for i in range(node_feature.shape[0]):
            j = 0
            flag = 0
            while flag < number:
                if y[node_nns[i][j]] == y[i]:
                    pos_graph[i][node_nns[i][j]] = 1
                    pos_graph[node_nns[i][j]][i] = 1
                    flag += 1
                j += 1
            j = node_feature.shape[0] - 2
            flag = 0
            while flag < number:
                if y[node_nns[i][j]] != y[i]:
                    neg_graph[i][node_nns[i][j]] = 1
                    neg_graph[node_nns[i][j]][i] = 1
                    flag += 1
                j -= 1

        return pos_graph, neg_graph  # , pos_graph.to_sparse().indices(), neg_graph.to_sparse().indices()

    def preprocess_graph(self, adj, num_nodes):
        adj_norm = torch.eye(num_nodes, num_nodes)
        for row, col in adj.transpose(0, 1):
            adj_norm[row][col] = 1
        adj_norm = sp.coo_matrix(adj_norm)
        row_sum = np.array(adj_norm.sum(1))  # row sum的shape=(节点数,1)，对于cora数据集来说就是(2078,1)，sum(1)求每一行的和
        degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())  # 计算D^{-0.5}
        adj_normalized = adj_norm.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.scipysp_to_pytorchsp(adj_normalized).to(adj.device)

    @staticmethod
    def preprocess_graph_new(adj, num_nodes):
        adj_norm = torch.eye(num_nodes, num_nodes)
        for row, col in adj.transpose(0, 1):
            adj_norm[row][col] = 1
        deg = torch.sum(adj_norm, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch.mul(adj_norm, deg_inv_sqrt.view(-1, 1))
        adj_t = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t.to(adj.device)

    @staticmethod
    def preprocess_graph(adj_norm, num_nodes):
        deg = torch.sum(adj_norm, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch.mul(adj_norm, deg_inv_sqrt.view(-1, 1))
        adj_t = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t.to(adj_t.device)

    @staticmethod
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


class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """

    def __init__(self, input_dim, output_dim, dim_z, dropout, gae=True):
        super(VGAE, self).__init__()
        self.mean = None
        self.logist = None
        self.means = None
        self.logists = None
        self.gae = gae
        # F.relu
        self.gcn_base = GCNLayer(input_dim=input_dim, output_dim=dim_z, activation=0, dropout=0,
                                 bias=False)
        self.gcn_mean = GCNLayer(input_dim=dim_z, output_dim=dim_z // 2, activation=0, dropout=0,
                                 bias=False)
        self.gcn_logist = GCNLayer(input_dim=dim_z, output_dim=dim_z // 2, activation=0, dropout=0,
                                   bias=False)

    def forward(self, features, adj):
        # GCN encoder
        hidden = self.gcn_base(features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        if self.gae:
            # GAE (no sampling at bottleneck)
            x = self.mean
        else:
            # VGAE
            if self.means is None:
                self.means = self.mean
            else:
                self.means = torch.cat((self.means, self.mean), dim=-1)
            self.logist = self.gcn_logist(hidden, adj)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_z = gaussian_noise * torch.exp(self.logist) + self.mean
            x = sampled_z
            if self.logists is None:
                self.logists = self.logist
            else:
                self.logists = torch.cat((self.logists, self.logist), dim=-1)
        # inner product decoder
        # adj_logit = x @ x.T
        return x

    def init_params(self):
        self.gcn_base.init_params()
        self.gcn_mean.init_params()
        self.gcn_logist.init_params()


class GCNLayer(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True, ep=False):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.ep = ep
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj, model=1):  # model=1 表示正常模式，model=0表示生成图
        if model == 1:
            if self.dropout:
                h = self.dropout(h)
            x = h @ self.W
            x = adj @ x
            if self.b is not None:
                x = x + self.b
            if self.activation:
                x = self.activation(x)
            if self.ep:
                x = x @ x.T
        else:
            w_t = self.W.data
            if self.dropout:
                h = self.dropout(h)
            x = h @ w_t
            x = adj @ x
            if self.b is not None:
                b_t = self.b.data
                x = x + b_t
            if self.activation:
                x = self.activation(x)
        return x
