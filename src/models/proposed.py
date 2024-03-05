import numpy as np
import gc
from collections import defaultdict
import itertools
import copy

import torch
from torch import nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler

from models.utils import normalize_adj, to_adj_matrix
from models.cm import compatibility
from models.nui import compute_netinfo
import rwcpp


def make_structural_features(edge_index, num_nodes, num_features):
    adj = to_adj_matrix(edge_index.cpu(), num_nodes, num_nodes)
    u, s, _ = torch.svd_lowrank(adj, int(num_features))
    return u * s.sqrt().unsqueeze(0)


def make_ppr_features(edge_index, num_nodes, num_features, T=200, k=3, threshold=1):
    row, col = edge_index.cpu().numpy()
    row, col = row[row < col], col[row < col]

    neighbors_dict = defaultdict(list)
    for n1, n2 in zip(row, col):
        neighbors_dict[n1].append(n2)
        neighbors_dict[n2].append(n1)
    walks_arr = rwcpp.random_walks(neighbors_dict, k, T, threshold)
    edge_index_ppr = np.array(list(itertools.chain.from_iterable(walks_arr.keys()))).reshape(-1, 2).T
    values_ppr = np.array(list(walks_arr.values()))
  
    adj_ppr = torch.sparse_coo_tensor(torch.tensor(edge_index_ppr), 
                                      torch.tensor(values_ppr).float(), (num_nodes, num_nodes))
    u, s, _ = torch.svd_lowrank(adj_ppr, int(num_features))
    return u * s.sqrt().unsqueeze(0)


def propagate(x, edge_index, num_layers=2, direction='row', self_loops=True):
    adj = normalize_adj(edge_index.long(), len(x), direction=direction, self_loops=self_loops)
    for _ in range(num_layers):
        x = torch.spmm(adj, x)
    return x


def preprocess(x_t):
    x_t = StandardScaler(with_mean=False).fit_transform(x_t)
    x_t = torch.tensor(Normalizer(norm='l2').fit_transform(x_t)).float()
    return x_t


class NetInfoF(nn.Module):
    def __init__(self, args, g, use_U=True, use_R=True, use_F=True, use_P=True, use_S=True, seed=0):
        super().__init__()

        self.input_dim = g.ndata['feat'].shape[1]
        self.num_nodes = g.num_nodes()
        self.edge_index = torch.stack(g.edges()).cpu()
        self.x = g.ndata['feat']
        self.hidden_size = min(self.x.shape[1], args.hidden)

        self.xs_t, self.xs_c, self.names = [], [], []
        if use_U:
            x_t = preprocess(make_structural_features(self.edge_index, self.num_nodes, self.hidden_size).numpy())
            x_c = compatibility(x_t, self.edge_index)
            self.xs_t.append(x_t)
            self.xs_c.append(x_c)
            self.names.append('U')

        if use_R:
            x_t = preprocess(make_ppr_features(self.edge_index, self.num_nodes, self.hidden_size, T=args.T).numpy())
            x_c = compatibility(x_t, self.edge_index)
            self.xs_t.append(x_t)
            self.xs_c.append(x_c)
            self.names.append('R')
        
        if use_F:
            x_t = preprocess(PCA(n_components=self.hidden_size, random_state=seed).fit_transform(self.x.cpu().numpy()))
            x_c = compatibility(x_t, self.edge_index)
            self.xs_t.append(x_t)
            self.xs_c.append(x_c)
            self.names.append('F')
        
        if use_P:
            x_t = propagate(self.x, self.edge_index, num_layers=2, direction='row', self_loops=False)
            x_t = preprocess(PCA(n_components=self.hidden_size, random_state=seed).fit_transform(x_t.cpu().numpy()))
            x_c = compatibility(x_t, self.edge_index)
            self.xs_t.append(x_t)
            self.xs_c.append(x_c)
            self.names.append('P')
                
        if use_S:
            x_t = propagate(self.x, self.edge_index, num_layers=2, direction='row', self_loops=True)
            x_t = preprocess(PCA(n_components=self.hidden_size, random_state=seed).fit_transform(x_t.cpu().numpy()))
            x_c = compatibility(x_t, self.edge_index)
            self.xs_t.append(x_t)
            self.xs_c.append(x_c)
            self.names.append('S')
            
        torch.cuda.empty_cache()
        gc.collect()

        self.n_components = len(self.xs_t)
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(self.n_components)])
        self.score_arr, self.sim_arr = [], []

    def probe(self, edge_index_valid, y_valid):
        for x_c, x_t, name in zip(self.xs_c, self.xs_t, self.names):
            score, sim = compute_netinfo(x_c, x_t, self.edge_index, edge_index_valid, y_valid)
            self.score_arr.append(score)
            self.sim_arr.append(sim)

        for score, name in zip(self.score_arr, self.names):
            print('%s: %.3f' % (name, score))
        print()

    def to(self, device):
        self.fcs = self.fcs.to(device)
        for i in range(self.n_components):
            self.xs_c[i] = self.xs_c[i].to(device)
            self.xs_t[i] = self.xs_t[i].to(device)
        return self

    def forward(self, edge_index):
        out = []
        for x_c, x_t, fc in zip(self.xs_c, self.xs_t, self.fcs):
            x = fc(torch.multiply(x_c[edge_index[0]], x_t[edge_index[1]]))
            out.append(x)
        return torch.stack(out).sum(0)
