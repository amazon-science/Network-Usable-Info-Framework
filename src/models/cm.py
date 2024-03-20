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
import gc
from collections import Counter

import torch

from scipy.sparse.linalg import lsqr
from sklearn.linear_model import Ridge


def compatibility_cheap(features, row, col):
    X, y = features[row], features[col]
    reg = Ridge(fit_intercept=False).fit(X, y)
    return reg.coef_


def compatibility(features, edge_index, two_core=True, sample_num=int(2e5)):
    n, d = features.shape
    pos_row, pos_col = edge_index.cpu().numpy()
    cheap_cm = compatibility_cheap(features.cpu().numpy(), pos_row, pos_col)

    if two_core:
        deg = Counter(pos_row)
        core_2 = set([k for k, v in deg.items() if v >= 2])
        del_idx = np.concatenate([[k for k, v in enumerate(pos_row) if v not in core_2],
                                  [k for k, v in enumerate(pos_col) if v not in core_2]]).astype(int)
        pos_row, pos_col = np.delete(pos_row, del_idx), np.delete(pos_col, del_idx)
    pos_row, pos_col = pos_row[pos_row < pos_col], pos_col[pos_row < pos_col]

    if len(pos_row) > sample_num:
        ridx = np.random.choice(np.arange(len(pos_row)), sample_num, replace=False)
        pos_row, pos_col = pos_row[ridx], pos_col[ridx]
        
    num_sampled_edges = int(len(pos_row)) * 2
    neg_row, neg_col = np.random.randint(0, n, num_sampled_edges), np.random.randint(0, n, num_sampled_edges)

    r0, c0 = np.triu_indices(d, k=0)
    dia_idx = np.array([np.arange(d, d-i, -1).sum() for i in range(d)])
    x0 = cheap_cm[(r0, c0)]

    v_sort = np.sort(np.abs(x0))
    epsilon = v_sort[np.where(np.cumsum(v_sort) / np.sum(v_sort) < 0.05)[0][-1]]
    del_idx = np.where(np.abs(x0) < epsilon)[0]
    r0_d, c0_d = np.delete(r0, del_idx), np.delete(c0, del_idx)
    dia_idx = np.array([i - len(del_idx[del_idx < i]) for i in dia_idx if i not in del_idx])

    features_rc = []
    for f in features.cpu().numpy():
        features_rc.append([f[r0_d], f[c0_d]])
    
    X, y = [], np.concatenate([np.ones(len(pos_row)), np.zeros(len(neg_row))]).astype(np.float32)
    for row, col in [[pos_row, pos_col], [neg_row, neg_col]]:
        for i, j in zip(row, col):
            mtr = np.multiply(features_rc[i][0], features_rc[j][1]) + np.multiply(features_rc[i][1], features_rc[j][0])
            mtr[dia_idx] /= 2
            X.append(mtr)
    X = np.array(X)

    coef_lsqr = lsqr(X, y, damp=1, atol=1e-2, btol=1e-2, x0=np.delete(x0, del_idx))[0]
    coef = np.zeros(len(x0))
    coef[np.abs(x0) >= epsilon] = coef_lsqr

    comp_matrix = torch.zeros((d, d)).float()
    comp_matrix[(r0, c0)] = torch.tensor(coef).float()
    comp_matrix = comp_matrix + comp_matrix.T - torch.diag(comp_matrix.diagonal())

    del features_rc, X, y
    gc.collect()

    return features @ comp_matrix
