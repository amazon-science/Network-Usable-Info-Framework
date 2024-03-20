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

import numpy as np
import os
import scipy
import gdown
import pandas as pd

from sklearn.model_selection import train_test_split

import torch

import dgl
from dgl.transforms import RowFeatNormalizer
from ogb.nodeproppred import DglNodePropPredDataset

from utils import ROOT
from data.synthetic import Synthetic

dataset_drive_url = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
}


def uniform_negative_sampling(g, num_edges, exact=True):
    num_nodes = g.num_nodes()

    if exact:
        pos_row, pos_col = g.edges()
        neighbors = {n: set(pos_col[pos_row == n]) for n in range(num_nodes)}
        neg_edge_index = []
        while True:
            src, dst = np.random.randint(num_nodes, size=2)
            if dst not in neighbors[src] and src != dst:
                neg_edge_index.append([src, dst])
            if len(neg_edge_index) == num_edges:
                break
        neg_edge_index = torch.tensor(neg_edge_index).int().to(g.device).T
    else:
        neg_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=g.device)

    return neg_edge_index[0], neg_edge_index[1]


def construct_dgl_graph(edge_index, num_nodes, features, symmetric=True):
    if symmetric:
        row, col = edge_index
        g = dgl.graph((torch.cat([row, col]), torch.cat([col, row])), num_nodes=num_nodes)
    else:
        g = dgl.graph(edge_index, num_nodes=num_nodes)
    g.ndata['feat'] = features
    return g


def split_edges(g, ratio, threshold=1e6, seed=None):
    assert len(ratio) == 3 and sum(ratio) == 1

    features = g.ndata['feat']
    num_nodes = g.number_of_nodes()
    row, col = g.edges()
    row, col = row[row < col], col[row < col]
    num_edges = len(row)

    train_idx, test_idx = train_test_split(np.arange(num_edges),
                                           test_size=ratio[2],
                                           random_state=seed)
    train_idx, valid_idx = train_test_split(train_idx,
                                            test_size=ratio[1] / (ratio[0] + ratio[1]),
                                            random_state=seed)

    train_g = construct_dgl_graph((row[train_idx], col[train_idx]), num_nodes, features)
    test_g = construct_dgl_graph((row[train_idx], col[train_idx]), num_nodes, features)

    exact = True if num_edges < threshold else False
    edge_index_valid = torch.cat([torch.stack((row[valid_idx], col[valid_idx])), 
                                  torch.stack(uniform_negative_sampling(g, len(valid_idx), exact=exact))], dim=-1)
    edge_index_test = torch.cat([torch.stack((row[test_idx], col[test_idx])), 
                                 torch.stack(uniform_negative_sampling(g, len(test_idx), exact=exact))], dim=-1)
    
    y_valid = torch.cat([torch.ones(len(valid_idx)), torch.zeros(len(valid_idx))]).to(g.device)
    y_test = torch.cat([torch.ones(len(test_idx)), torch.zeros(len(test_idx))]).to(g.device)

    return train_g, test_g, (edge_index_valid, y_valid), (edge_index_test, y_test)


def load_data(name, root=os.path.join(ROOT, 'data')):
    if not os.path.exists(root):
        os.makedirs(root)

    if name == 'cora':
        g = dgl.data.CoraGraphDataset(raw_dir=root, verbose=False)[0]
    elif name == 'citeseer':
        g = dgl.data.CiteseerGraphDataset(raw_dir=root, verbose=False)[0]
    elif name == 'pubmed':
        g = dgl.data.PubmedGraphDataset(raw_dir=root, verbose=False)[0]
    elif name == 'computers':
        g = dgl.data.AmazonCoBuyComputerDataset(raw_dir=root, verbose=False)[0]
    elif name == 'photo':
        g = dgl.data.AmazonCoBuyPhotoDataset(raw_dir=root, verbose=False)[0]
    elif name in ['chameleon', 'squirrel', 'actor']:
        data = scipy.io.loadmat(f'{root}/{name}.mat')
        row, col = torch.tensor(data['edge_index']).long()
        row, col = row[row != col], col[row != col]
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])])
        features = torch.tensor(data['node_feat']).float()
        num_nodes = features.shape[0]
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        g.ndata['feat'] = features
    elif name == 'twitch':
        if not os.path.exists(f'{root}/twitch-gamer_feat.csv'):
            gdown.download(id=dataset_drive_url['twitch-gamer_feat'], output=f'{root}/twitch-gamer_feat.csv', quiet=False)
        if not os.path.exists(f'{root}/twitch-gamer_edges.csv'):
            gdown.download(id=dataset_drive_url['twitch-gamer_edges'], output=f'{root}/twitch-gamer_edges.csv', quiet=False)
        edges = pd.read_csv(f'{root}/twitch-gamer_edges.csv')
        nodes = pd.read_csv(f'{root}/twitch-gamer_feat.csv')
        edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
        num_nodes = len(nodes)
        nodes = nodes.drop('numeric_id', axis=1)
        nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
        nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
        one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
        lang_encoding = [one_hot[lang] for lang in nodes['language']]
        nodes['language'] = lang_encoding
        task = 'mature'
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        g.ndata['feat'] = torch.tensor(features).float()
    elif name == 'pokec':
        if not os.path.exists(f'{root}/pokec.mat'):
            gdown.download(id=dataset_drive_url['pokec'], output=f'{root}/pokec.mat', quiet=False)
        fulldata = scipy.io.loadmat(f'{root}/pokec.mat')
        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        features = torch.tensor(fulldata['node_feat']).float()
        num_nodes = int(fulldata['num_nodes'])
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        g.ndata['feat'] = features
    elif name == 'arxiv':
        data = DglNodePropPredDataset(name='ogbn-arxiv', root=root)
        g = data[0][0]
    elif name == 'products':
        data = DglNodePropPredDataset(name='ogbn-products', root=root)
        g = data[0][0]
    elif name.startswith('synthetic'):
        if name + '.dgl' in os.listdir(root):
            g = dgl.load_graphs(os.path.join(root, name + '.dgl'))[0][0]
            print('Load Synthetic Graph!')
        else:
            keys = name.split('-')[1:]
            if keys[1] == 'diag':
                data = Synthetic(x_type=keys[0], e_type=keys[1],
                                edge_density=0.01, edge_noise=0.0002, feature_noise=0.002)
            elif keys[1] == 'offdiag':
                data = Synthetic(x_type=keys[0], e_type=keys[1],
                                edge_density=0.01, edge_noise=0.0001, feature_noise=0.002)
            else:
                data = Synthetic(x_type=keys[0], e_type=keys[1])
            g = data.g
            dgl.save_graphs(os.path.join(root, name + '.dgl'), [g])
            print('Save Synthetic Graph!')

    if not name.startswith('synthetic'):
        g = RowFeatNormalizer(subtract_min=True, node_feat_names=['feat'])(g)
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g, copy_ndata=True)

    return g

    
