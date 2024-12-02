import argparse
import os
import random
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm.auto import trange


class MLP(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, data):
        return self.net(data)


class Evaluator:
    def __init__(self, train_embeddings, test_embeddings, train_labels, test_labels, seed=None, verbose=False):
        if seed is None:
            seed = np.random.randint(0, 10000)
        self.size = train_embeddings.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(train_embeddings.shape[1], train_labels.shape[1]).to(self.device)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)

        torch.manual_seed(seed)
        self.train_index, self.val_index = train_test_split(np.arange(self.size), test_size=0.2, random_state=seed)

        self.train_data = torch.from_numpy(train_embeddings[self.train_index]).float().to(self.device)
        self.train_target = torch.from_numpy(train_labels[self.train_index]).float().to(self.device)
        self.val_data = torch.from_numpy(train_embeddings[self.val_index]).float().to(self.device)
        self.val_target = torch.from_numpy(train_labels[self.val_index]).float().to(self.device)
        self.test_data = torch.from_numpy(test_embeddings).float().to(self.device)
        self.test_target = torch.from_numpy(test_labels).float().to(self.device)
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.train_data, self.train_target),
            batch_size=64, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.val_data, self.val_target),
                                                      batch_size=64, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.test_data, self.test_target),
                                                       batch_size=64, shuffle=False)
        self.verbose = verbose

    def label_smoothing(self, predictions, targets, epsilon=1e-9):
        epsilon = 1e-9  # A small value to avoid zeros

        # Apply smoothing to both predictions and targets
        predictions = predictions.clamp(min=epsilon)  # Avoid zero in predictions
        targets = targets.clamp(min=epsilon)          # Avoid zero in targets

        # Normalize again after clamping
        predictions = predictions / predictions.sum(dim=1, keepdim=True)
        targets = targets / targets.sum(dim=1, keepdim=True)

        return predictions, targets

    def train(self, epochs):
        lowest_val_loss = float('inf')
        best_test_kl_div = 0
        best_test_l1 = 0
        best_test_cos = 0
        for epoch in range(epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                train_loss = self.loss(output.log(), target)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
            # self.scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_size = 0
                val_total_loss = 0
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)

                    output, target = self.label_smoothing(output, target)

                    val_total_loss += self.loss(output.log(), target).item() * data.shape[0]
                    val_size += data.shape[0]
                val_loss = val_total_loss / val_size
                test_size = 0
                test_kl_div_total = 0
                test_l1_total = 0
                test_cos_total = 0
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    output, target = self.label_smoothing(output, target)
                    test_kl_div_total += F.kl_div(output.log(), target, reduction='batchmean').item() * data.shape[0]
                    test_l1_total += torch.sum(torch.abs(output - target)).item()
                    test_cos_total += torch.sum(torch.cosine_similarity(output, target, dim=1)).item()
                    test_size += data.shape[0]
                test_kl_div = test_kl_div_total / test_size
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_test_kl_div = test_kl_div
                    best_test_l1 = test_l1_total / test_size
                    best_test_cos = test_cos_total / test_size
        if self.verbose:
            print('Best test_loss: {:.4f}, test_l1: {:.4f}, test_cos: {:.4f}'.format(best_test_kl_div, best_test_l1,
                                                                                     best_test_cos))
        return best_test_kl_div, best_test_l1, best_test_cos

    def predict(self, epochs):
        lowest_val_loss = float('inf')
        best_test_l1 = 0
        best_test_l1_array = None
        for epoch in range(epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                output, target = self.label_smoothing(output, target)
                train_loss = self.loss(output.log(), target)
                train_loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_size = 0
                val_total_loss = 0
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    output, target = self.label_smoothing(output, target)
                    val_total_loss += self.loss(output.log(), target).item() * data.shape[0]
                    val_size += data.shape[0]
                val_loss = val_total_loss / val_size
                test_l1_list = []
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_l1_list.append((torch.abs(output - target).sum(dim=-1).cpu().numpy()))
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_test_l1 = np.concatenate(test_l1_list).mean()
                    best_test_l1_array = np.concatenate(test_l1_list)
        return best_test_l1, best_test_l1_array
    
def land_use_inference(baseline_embeddings, raw_labels, split, repeat, verbose=False):
    embeddings = baseline_embeddings
    labels = raw_labels

    kl_div, l1, cos = [], [], []
    idxs = np.arange(len(embeddings))
    if not verbose:
        pbar = trange(1, repeat + 1)
    else:
        pbar = range(1, repeat + 1)
    for i in pbar:
        train_index, test_index = train_test_split(idxs, test_size=split[2] / (split[0] + split[1] + split[2]),
                                                   random_state=i)
        # make sure the result is fully reproducible by fix seed = epoch number
        evaluator = Evaluator(train_embeddings=embeddings[train_index], train_labels=labels[train_index],
                              test_embeddings=embeddings[test_index], test_labels=labels[test_index],
                              seed=i, verbose=verbose)
        test_kl_div, test_l1, test_cos = evaluator.train(epochs=150)
        kl_div.append(test_kl_div)
        l1.append(test_l1)
        cos.append(test_cos)
    if not verbose:
        pbar.close()
    average_l1 = np.mean(l1)
    std_l1 = np.std(l1)
    average_kl_div = np.mean(kl_div)
    std_kl_div = np.std(kl_div)
    average_cos = np.mean(cos)
    std_cos = np.std(cos)
    if verbose:
        # print('Result for ')
        # print('L1\t\tstd\t\tKL-Div\t\tstd\t\tCosine\t\tstd')
        print(f"L1: {average_l1:.4f} +- {std_l1:.4f}\nKL-Div: {average_kl_div:.4f} +- {std_kl_div:.4f}\nCosine: {average_cos:.4f} +- {std_cos:.4f}")
        # print(f'{average_l1}\t{std_l1}\t{average_kl_div}\t{std_kl_div}\t{average_cos}\t{std_cos}')
    return [average_l1, std_l1, average_kl_div, std_kl_div, average_cos, std_cos]

if __name__ == '__main__':
    import pandas as pd

    # baseline_path = '/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/h3-embeddings/data/ny_geovex_emb-res-8.csv'
    baseline_path = '/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/region-embedding-chicago.csv'
    # baseline_path = '/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/GAE/notebooks/new-york-GAE.csv'
    # baseline_path = '/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/HGI/data/region_embedding.csv'

    ground_truth_path = "/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/downstream/functionalZone/data/chicago-landuse-area-9-classes-geoid.csv"

    column = 'GEOID'
    
    baseline_df = pd.read_csv(baseline_path).drop_duplicates(subset=column)
    ground_truth_df = pd.read_csv(ground_truth_path).dropna()

    boroughs = baseline_df[column].unique().tolist()
    ground_truth_df = ground_truth_df[ground_truth_df[column].isin(boroughs)].sort_values(column)

    boroughs = ground_truth_df[column].unique().tolist()
    baseline_df = baseline_df[baseline_df[column].isin(boroughs)].sort_values(column)

    baseline_embeddings = baseline_df.iloc[:, 0:64].values
    raw_labels = ground_truth_df.iloc[:, 1:].values

    print(raw_labels.shape)

    # print(list(zip(baseline_df['BoroCT2020'].values, ground_truth_df['BoroCT2020'].values)))

    split = [0.6, 0.2, 0.2]
    result = land_use_inference(baseline_embeddings, raw_labels, split, 10, verbose=True)


### Chicago

## Our Model
# L1: 0.4750 +- 0.0123
# KL-Div: 0.2898 +- 0.0125
# Cosine: 0.9040 +- 0.0063

## GAE
# L1: 0.5904 +- 0.0108
# KL-Div: 0.3847 +- 0.0166
# Cosine: 0.8749 +- 0.0104

## DGI
# L1: 0.5594 +- 0.0118
# KL-Div: 0.3559 +- 0.0150
# Cosine: 0.8841 +- 0.0097

## Poi-Encoder
# L1: 0.5146 +- 0.0168
# KL-Div: 0.3008 +- 0.0227
# Cosine: 0.8938 +- 0.0135

## HGI
# L1: 0.5489 +- 0.0136
# KL-Div: 0.3433 +- 0.0237
# Cosine: 0.8876 +- 0.0132


## RegionDCL
# L1: 0.4842 +- 0.0247
# KL-Div: 0.2974 +- 0.0295
# Cosine: 0.9025 +- 0.0135


### New York

## Our Model
# L1: 0.6655 +- 0.0170
# KL-Div: 0.4550 +- 0.0344
# Cosine: 0.8222 +- 0.0099

## DGI
# L1: 0.7200 +- 0.0189
# KL-Div: 0.5042 +- 0.0306
# Cosine: 0.8010 +- 0.0100

## GAE
# L1: 0.8559 +- 0.0125
# KL-Div: 0.6471 +- 0.0267
# Cosine: 0.7288 +- 0.0064

## Poi-Encoder
# L1: 0.7225 +- 0.0175
# KL-Div: 0.5101 +- 0.0233
# Cosine: 0.7948 +- 0.0081

## HGI
# L1: 0.7465 +- 0.0148
# KL-Div: 0.5619 +- 0.0307
# Cosine: 0.7860 +- 0.0095

## RegionDCL
# L1: 0.5991 +- 0.0136
# KL-Div: 0.3956 +- 0.0235
# Cosine: 0.8508 +- 0.0093