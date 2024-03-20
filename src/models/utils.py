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

import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add


def to_adj_matrix(edge_index, num_nodes_1, num_nodes_2):
    return torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes_1, num_nodes_2))


def normalize_adj(edge_index, num_nodes=None, edge_weight=None, direction='sym', self_loops=True):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    if self_loops:
        fill_value = 1.
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if direction == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif direction == 'row':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif direction == 'col':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]
    else:
        raise ValueError()

    return torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
