"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import numpy as np

import torch
from torch import spmm

import dgl

import rwcpp
from models.utils import normalize_adj, to_adj_matrix


def make_intra_edges_block(nodes, density):
    edge_index, visited_nodes = set(), np.zeros(len(nodes))
    max_edge_num = len(nodes) * (len(nodes) - 1) / 2
    p = np.ones(len(nodes))
    while True:
        clique_size = np.random.randint(4, 9)
        rand_node_idx = np.random.choice(nodes, clique_size, replace=False, p=p/p.sum()).astype(int)

        for i, nid_src in enumerate(rand_node_idx):
            visited_nodes[nodes == nid_src] = 1
            for nid_dst in rand_node_idx[i+1:]:
                edge_index.add((nid_src, nid_dst))

        if len(edge_index) / max_edge_num > density and visited_nodes.sum() == len(nodes):
            break
        p[visited_nodes == 1] = 0.1
    edge_index = np.array([list(e) for e in edge_index]).T

    row = torch.LongTensor(edge_index[0])
    col = torch.LongTensor(edge_index[1])
    row_out = torch.cat([row, col])
    col_out = torch.cat([col, row])
    return torch.stack([row_out, col_out]).long()


def make_intra_edges_butterfly(nodes1, nodes2, density):
    edge_index, visited_nodes1, visited_nodes2 = set(), np.zeros(len(nodes1)), np.zeros(len(nodes2))
    max_edge_num = len(nodes1) * len(nodes2)
    p1, p2 = np.ones(len(nodes1)), np.ones(len(nodes2))
    while True:
        clique_size1 = np.random.randint(4, 9)
        clique_size2 = np.random.randint(4, 9)
        rand_node_idx1 = np.random.choice(nodes1, clique_size1, replace=False, p=p1/p1.sum()).astype(int)
        rand_node_idx2 = np.random.choice(nodes2, clique_size2, replace=False, p=p2/p2.sum()).astype(int)

        for nid_src in rand_node_idx1:
            visited_nodes1[nodes1 == nid_src] = 1
            for nid_dst in rand_node_idx2:
                visited_nodes2[nodes2 == nid_dst] = 1
                edge_index.add((nid_src, nid_dst))

        if len(edge_index) / max_edge_num > density and visited_nodes1.sum() == len(nodes1) and visited_nodes2.sum() == len(nodes2):
            break
        p1[visited_nodes1 == 1] = 0.1
        p2[visited_nodes2 == 1] = 0.1
    edge_index = np.array([list(e) for e in edge_index]).T

    row = torch.LongTensor(edge_index[0])
    col = torch.LongTensor(edge_index[1])
    row_out = torch.cat([row, col])
    col_out = torch.cat([col, row])
    return torch.stack([row_out, col_out]).long()


def make_inter_edges(nodes1, nodes2, density):
    row, col = [], []
    for nid in nodes1:
        dst_idx = nodes2[torch.nonzero(torch.rand(len(nodes2)) < density, as_tuple=True)]
        dst_idx = dst_idx[dst_idx != nid]
        row.append(torch.full((len(dst_idx),), fill_value=nid))
        col.append(dst_idx)
    row = torch.cat(row)
    col = torch.cat(col)
    row_out = torch.cat([row, col])
    col_out = torch.cat([col, row])
    return torch.stack([row_out, col_out])


def make_clustered_edges(clusters, pairs, density):
    def select_nodes(c):
        return torch.nonzero(clusters == c, as_tuple=True)[0]
    
    edge_list = []
    num_clusters = (clusters.max() + 1).item()
    for c1 in range(num_clusters):
        for c2 in range(c1, num_clusters):
            if (c1, c2) in pairs:
                if c1 == c2:
                    edges = make_intra_edges_block(select_nodes(c1), density)
                else:
                    edges = make_intra_edges_butterfly(select_nodes(c1), select_nodes(c2), density)
                edge_list.append(edges)
    return torch.cat(edge_list, dim=1)


def make_noisy_edges(edge_index, clusters, pairs, noise):
    def select_nodes(c):
        return torch.nonzero(clusters == c, as_tuple=True)[0]

    edge_list = []
    num_clusters = (clusters.max() + 1).item()
    for c1 in range(num_clusters):
        for c2 in range(c1, num_clusters):
            if (c1, c2) not in pairs:
                edges = make_inter_edges(select_nodes(c1), select_nodes(c2), noise)
                edge_list.append(edges)
    noisy_edge_list = torch.cat(edge_list, dim=1)
    return torch.cat([edge_index, noisy_edge_list], dim=1)


def assign_clusters(num_nodes, num_clusters):
    clusters = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_clusters):
        clusters[int(i * num_nodes / num_clusters):
                 int((i + 1) * num_nodes / num_clusters)] = i
    return clusters


def propagate_features(init_features, edge_index, num_nodes, num_steps=2):
    adj = normalize_adj(edge_index, num_nodes, direction='row')
    out = init_features
    for _ in range(num_steps):
        out = spmm(adj, out)
    return out


class Synthetic():
    def __init__(self, num_nodes=4000, num_features=800, num_classes=4,
                 x_type='global', e_type='diag',
                 edge_density=0.01, edge_noise=0., feature_noise=0.):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes_ = num_classes
        self.edge_density = edge_density
        self.edge_noise = edge_noise
        self.feature_noise = feature_noise

        self.x_type = x_type
        self.e_type = e_type
        if self.e_type == 'diag':
            self.pairs = [(i, i) for i in range(self.num_classes)]
        elif self.e_type == 'offdiag':
            self.pairs = self.pick_class_pairs()
        else:
            raise NotImplementedError
        self.clusters = assign_clusters(self.num_nodes, self.num_classes)

        edge_index = self.make_adjacency()
        edge_index = make_noisy_edges(edge_index, self.clusters, self.pairs, self.edge_noise)

        y = self.make_labels()
        x = self.make_features(edge_index, self.num_features)

        self.g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        self.g.ndata['feat'] = x
        self.g.ndata['label'] = y

    @property
    def num_classes(self):
        return self.num_classes_

    def pick_class_pairs(self):
        assert self.num_classes % 2 == 0
        return [(2 * i, 2 * i + 1) for i in range(self.num_classes // 2)]

    def make_adjacency(self):
        if self.e_type == 'uniform':
            nodes = torch.arange(self.num_nodes)
            return make_inter_edges(nodes, nodes, self.edge_density)
        elif self.e_type in ['diag', 'offdiag']:
            return make_clustered_edges(self.clusters, self.pairs, self.edge_density)
        else:
            raise ValueError(self.e_type)

    def make_labels(self):
        if self.e_type == 'uniform':
            return torch.randint(self.num_classes, (self.num_nodes,))
        elif self.e_type == 'diag':
            return assign_clusters(self.num_nodes, self.num_classes)
        elif self.e_type == 'offdiag':
            return assign_clusters(self.num_nodes, self.num_classes)
        else:
            raise ValueError(self.e_type, self.y_type)

    def make_features(self, edge_index, num_features):
        def make_structural_features():
            adj = to_adj_matrix(edge_index, self.num_nodes, self.num_nodes)
            u, s, _ = torch.svd_lowrank(adj, num_features)
            return u * s.sqrt().unsqueeze(0)
        
        def make_global_features():
            row, col = edge_index.cpu().numpy()
            neighbors_dict = {n: col[row == n] for n in range(self.num_nodes)}
            walks_arr = rwcpp.random_walks(neighbors_dict, 3, 1000)
            out = torch.sparse_coo_tensor(torch.tensor([[i, j] for i, j in walks_arr.keys()]).T, 
                                  torch.tensor(list(walks_arr.values())).float(), (self.num_nodes, self.num_nodes))
            return out

        if self.x_type == 'random':
            out = torch.randint(0, 2, (self.num_nodes, num_features)).float()
        elif self.x_type == 'global':
            u, s, _ = torch.svd_lowrank(make_global_features(), num_features)
            u = u * s.sqrt().unsqueeze(0)
            u -= u.min()
            out = u
        elif self.x_type == 'local':
            f_num = int(self.num_features / self.num_classes_)
            u, s, _ = torch.svd_lowrank(make_global_features(), f_num)
            u = u * s.sqrt().unsqueeze(0)
            u -= u.min()

            out = torch.zeros((self.num_nodes, self.num_features))
            for c, i in enumerate(range(0, self.num_features, f_num)):
                out[self.clusters == c, i:i + f_num] = u[self.clusters == c]
        else:
            raise ValueError(self.x_type)

        if self.feature_noise > 0 and self.x_type != 'random':
            mask = torch.rand_like(out) < self.feature_noise
            out[mask] = torch.rand_like(out)[mask]

        return out
