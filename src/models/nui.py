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
import torch
from sklearn.preprocessing import KBinsDiscretizer


def conditional_entropy(X):
    n = X.sum()
    m, c = X.shape
    ce = 0
    for i in range(m):
        pi = X[i].sum() / n
        for j in range(c):
            pij = X[i][j] / n
            if pij == 0:
                continue
            ce += - pij * np.log2(pij / pi)
    return ce


def accuracy(X):
    n = X.sum()
    acc = 0
    for i in range(len(X)):
        s = X[i].sum()
        if s != 0:
            pi = s / n
            acc += pi * (np.max(X[i]) / s)
    return acc


def compute_netinfo(x_c, x_t, edge_index, edge_index_valid, y_valid, bins=8):
    n, d = x_t.shape
    pos_row, pos_col = edge_index.cpu().numpy()
    pos_row, pos_col = pos_row[pos_row < pos_col], pos_col[pos_row < pos_col]

    num_sampled_edges = int(len(pos_row)) * 2
    neg_row, neg_col = np.random.randint(0, n, num_sampled_edges), np.random.randint(0, n, num_sampled_edges)

    similarity_train = []
    for row, col in [[pos_row, pos_col], [neg_row, neg_col]]:
        for i, j in zip(row, col):
            similarity_train.append(torch.dot(x_c[i], x_t[j]).item())
    y_train = np.concatenate([np.zeros(len(neg_row)), np.zeros(len(pos_row))])
    
    kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile', 
                           subsample=len(similarity_train), random_state=0)
    kbd.fit(np.array(similarity_train).reshape(-1, 1))

    similarity_valid = []
    for i, j in zip(*edge_index_valid):
        similarity_valid.append(torch.dot(x_c[i], x_t[j]).item())
    similarity_valid = kbd.transform(np.array(similarity_valid).reshape(-1, 1)).astype(int).reshape(-1)
    
    table = np.zeros((bins, 2))
    for s, y in zip(similarity_valid, y_valid.cpu().numpy()):
        table[s, int(y)] += 1
    ce, acc = conditional_entropy(table), accuracy(table)
    netinfo_valid = 2 ** (-ce)

    return netinfo_valid, similarity_valid
