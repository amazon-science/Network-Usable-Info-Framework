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

import argparse
import io
from distutils.util import strtobool
import numpy as np
import itertools
from tqdm import tqdm
import time

import torch
from torch import nn, optim

from utils import set_seed
from models import load_model
from data import load_data, split_edges

torch.set_num_threads(8)
    

def to_regularizer(model, lambda_1, lambda_2, device):
    out_1, out_2 = torch.zeros(model.n_components).to(device), torch.zeros(model.n_components).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith('weight'):
            i = int(name.split('.')[1])
            out_1[i] += torch.abs(param).sum()
            out_2[i] += torch.pow(param, 2).sum()
    return lambda_1 * out_1.sum() + lambda_2 * out_2.sqrt().sum()


def train_model(args, model, edge_index, edge_index_valid, edge_index_test, y_valid, y_test, lambda_1, lambda_2, device):
    model = model.to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(model.parameters(), lr=args.lr)

    row, col = edge_index
    pos_edge_index = torch.stack((row[row < col], col[row < col]))
    num_edges = pos_edge_index[0].shape[0]
    num_nodes = train_g.num_nodes()

    y_train = torch.cat([torch.ones(num_edges), torch.zeros(num_edges)]).to(device)
    y_valid = y_valid.to(device)
    y_test = y_test.to(device)

    @torch.no_grad()
    def evaluate(edge_index_eval, y_eval, K=args.hitK):
        model.eval()
        y_pred = model(edge_index_eval).reshape(-1)
        loss = loss_func(y_pred, y_eval).item()
        hr = torch.where(y_pred[y_eval == 1] > torch.topk(y_pred[y_eval == 0], K)[0][-1], 1, 0).float().mean().item()
        return loss, hr

    def closure():
        optimizer.zero_grad()

        neg_edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_index_train = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
     
        y_pred = model(edge_index_train).reshape(-1)
        reg_loss = to_regularizer(model, lambda_1, lambda_2, device)
        loss = loss_func(y_pred, y_train) + reg_loss
        loss.backward()

        return loss

    best_epoch, best_hr, best_model = -1, 0, io.BytesIO()
    for epoch in range(args.max_epochs + 1):
        model.train()
        optimizer.step(closure)
        val_loss, val_hr = evaluate(edge_index_valid, y_valid)

        if val_hr > best_hr:
            best_epoch = epoch
            best_hr = val_hr
            torch.save(model.state_dict(), best_model)
            best_model.seek(0)
        elif epoch >= best_epoch + args.patience or val_loss > 1000:
            break

    try:
        model.load_state_dict(torch.load(best_model))
        _, test_hr = evaluate(edge_index_test, y_test)
    except:
        best_hr, test_hr = 0, 0, 0

    return best_hr, test_hr


def parse_args():
    def str2bool(x):
        return bool(strtobool(x))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--hitK', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=200, help='number of random walk trials for neighborhood embedding')
    parser.add_argument('--print_probe', type=str2bool, default=False, help='print the results of NetInfoF_Probe')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.gpu) if torch.cuda.is_available() else torch.device('cpu')

    g = load_data(args.dataset)
    print(args.dataset, '\n')

    results_arr, time_arr = [], []
    for seed in range(5):
        train_g, test_g, valid_data, test_data = split_edges(g, ratio=(0.7, 0.1, 0.2), seed=seed)

        edge_index = torch.stack(train_g.edges())
        edge_index_valid, y_valid = valid_data
        edge_index_test, y_test = test_data

        start_time = time.time()
        model = load_model(args, train_g, 'NetInfoF', use_U=True, use_R=True, use_F=True, use_P=True, use_S=True)
        cons_time = time.time() - start_time

        if args.print_probe:
            model.probe(edge_index_valid, y_valid)

        hyperparameter_search = {'lambda_1': [1e-4, 1e-5], 'lambda_2': [1e-3, 1e-4, 1e-5, 1e-6]}
        results = []
        for lambda_1, lambda_2 in tqdm(itertools.product(*hyperparameter_search.values())):
            start_time = time.time()
            best_hr, test_hr = train_model(args, model, edge_index, edge_index_valid, edge_index_test, y_valid, y_test, 
                                           lambda_1=lambda_1, lambda_2=lambda_2, device=device)
            results.append([best_hr, test_hr, time.time() - start_time])
        
        results = np.array(results)
        best_idx = np.argmax(results[:, 0])
        results_arr.append(results[best_idx][1])
        time_arr.append(cons_time + results[best_idx][-1])

    results_arr = np.array(results_arr) * 100
    print('NetInfoF - HR@%d: %.1f +- %.1f' % (args.hitK, results_arr.mean(), results_arr.std()))
    print('Run Time: %.1f' % np.mean(time_arr))
    print()
