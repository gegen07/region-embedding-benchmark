import argparse
import os
import random
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
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
                self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_size = 0
                val_total_loss = 0
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
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
        print('Result for ')
        print('L1\t\tstd\t\tKL-Div\t\tstd\t\tCosine\t\tstd')
        print(f'{average_l1}\t{std_l1}\t{average_kl_div}\t{std_kl_div}\t{average_cos}\t{std_cos}')
    return f'{average_kl_div} +- {std_kl_div}'


def train_crime(df):
    y = df['crime_number'].values
    X = df.drop(columns=['crime_number', 'BoroCT2020']).values

    model = Lasso(random_state=0, max_iter=10000)

    grid = dict()
    grid['alpha'] = np.arange(0.1, 20, 0.01)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    model = Lasso(alpha=results.best_params_['alpha'], random_state=0)

    rmse_list = []
    mae_list = []
    r2_list = []
    rmse_train_list = []

    for train_index, test_index in cv.split(X):
        # print(train_index, test_index)
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        rmse = root_mean_squared_error(y_test_fold, y_pred)
        mae = mean_absolute_error(y_test_fold, y_pred)

        y_train_pred = model.predict(X_train_fold)

        r2 = r2_score(y_train_fold, y_train_pred)
        rmse_train = root_mean_squared_error(y_train_fold, y_train_pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        rmse_train_list.append(rmse_train)
    # print(f'Config: {results.best_params_}')

    # print(f'RMSE: {sum(rmse_list)/len(rmse_list)} +- {np.std(np.array(rmse_list))}')
    # print(f'MAE: {sum(mae_list)/len(mae_list)} +- {np.std(np.array(mae_list))}')
    # print(f'RMSE Train: {sum(rmse_train_list)/len(rmse_train_list)} +- {np.std(np.array(rmse_train_list))}')
    # print(f'R2: {sum(r2_list)/len(r2_list)} +- {np.std(np.array(r2_list))}')
    return f'{sum(rmse_train_list)/len(rmse_train_list)} +- {np.std(np.array(rmse_train_list))}'

def train_population(df):
    y = df['pop'].values
    X = df.drop(columns=['pop', 'BoroCT2020']).values

    model = Lasso(random_state=0, max_iter=10000)

    grid = dict()
    grid['alpha'] = np.arange(0.1, 50, 0.1)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    model = Lasso(alpha=results.best_params_['alpha'], random_state=0)
    rmse_list = []
    mae_list = []
    r2_list = []
    rmse_train_list = []
    for train_index, test_index in cv.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        rmse = root_mean_squared_error(y_test_fold, y_pred)
        mae = mean_absolute_error(y_test_fold, y_pred)

        y_train_pred = model.predict(X_train_fold)

        r2 = r2_score(y_train_fold, y_train_pred)
        rmse_train = root_mean_squared_error(y_train_fold, y_train_pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        rmse_train_list.append(rmse_train)

    return f'{sum(mae_list)/len(mae_list)} +- {np.std(np.array(mae_list))}'

def read_embeddings(filename):
    embeddings = torch.load(filename).numpy()
    cta = pd.read_csv('/home/gegen07/dev/projects/master-region-embedding/src/data/geo-entities.csv')

    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [str(i) for i in range(64)]
    emb_df['BoroCT2020'] = cta['BoroCT2020']
    emb_df['geometry'] = cta['geometry']

    return emb_df

def lambda_experiment(filename):
    import warnings
    warnings.filterwarnings('ignore')

    my_model_df = read_embeddings(filename) 

    ground_truth_path = "/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/downstream/functionalZone/data/nyc-landuse-area-5-classes.csv"
    column = 'BoroCT2020'
    ground_truth_df = pd.read_csv(ground_truth_path)
    boroughs = my_model_df[column].unique().tolist()
    ground_truth_df = ground_truth_df[ground_truth_df[column].isin(boroughs)].sort_values('BoroCT2020')
    boroughs = ground_truth_df[column].unique().tolist()
    baseline_df = my_model_df[my_model_df[column].isin(boroughs)].sort_values('BoroCT2020')
    baseline_embeddings = baseline_df.iloc[:, 0:64].values
    raw_labels = ground_truth_df.iloc[:, 1:].values
    split = [0.6, 0.2, 0.2]
    land_use_result = land_use_inference(baseline_embeddings, raw_labels, split, 10, verbose=False)


    crime = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/downstream/crime-prediction/data/crime_number-manhattan.csv')
    embedding = my_model_df.copy().drop(columns=['geometry'])
    crime = crime.merge(embedding, on='BoroCT2020', how='inner')
    crime_result = train_crime(crime)

    pop = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/downstream/population-density/data/pop_cta_nyc.csv')
    embedding = my_model_df.copy().drop(columns=['geometry'])
    pop = pop.merge(embedding, on='BoroCT2020', how='inner')
    population_result = train_population(pop)

    return land_use_result, crime_result, population_result


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    data = {}
    with open('lambda_experiment.pkl', 'rb') as f:
        data = pkl.load(f)

    for lambda_1 in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for lambda_2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if lambda_1 == 0.4 and lambda_2 == 0.4:
                continue
            print(f'Running for lambda_1={lambda_1} and lambda_2={lambda_2}')
            filename = f'/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/lambda-benchmark/graph-transformer-multiple_graph-{lambda_1}-{lambda_2}.torch'

            land_use_result, crime_result, population_result = lambda_experiment(filename)

            data[str(lambda_1)+'-'+str(lambda_2)] = {
                'land_use': land_use_result,
                'crime': crime_result,
                'population': population_result
            }
    with open('lambda_experiment.pkl', 'wb') as f:
        pkl.dump(data, f)