import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import pandas as pd
from scipy.spatial.distance import canberra, chebyshev


class FunctionZoneDataset(Dataset):

    def __init__(self, csv_file_features, csv_file_labels):
        self.features = pd.read_csv(csv_file_features)
        self.labels = pd.read_csv(csv_file_labels)

        boroughs = self.labels['BoroCT2020'].unique().tolist()

        self.features = self.features[self.features['BoroCT2020'].isin(boroughs)]


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features.iloc[idx, 1:].values
        labels = self.labels.iloc[idx, 1:].values

        return {'features': torch.from_numpy(features), 'labels': torch.from_numpy(labels)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

zone_list = []

ground_truth_file = "./data/nyc-landuse.csv"
region_embedding_file = '../../baselines/poi-encoder/data/region_embedding.csv'

function_zone_dataset = FunctionZoneDataset(region_embedding_file, ground_truth_file)

embedding_size = 64

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(embedding_size, 32)
        self.lin1 = torch.nn.Linear(32, 11)

    def forward(self, x):
        out = torch.tanh(self.lin0(x.float())).float()
        out = F.softmax(self.lin1(out), -1).float()
        return out.view(-1).float()


def train():
    model.train()
    loss_all = 0
    p_a = 0
    for data in train_loader:
        data = data
        optimizer.zero_grad()

        features = torch.Tensor(data['features'])
        label = torch.Tensor(data['labels'])

        y_estimated = model(features)

        loss = F.kl_div(torch.log(y_estimated.reshape((-1, 11))), label.reshape((-1, 11)).float(), 
                        reduction='batchmean').float()
        p_a += (y_estimated - label.reshape(-1)).abs().sum()
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset), p_a / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0
    canberra_distance = 0
    chebyshev_distance = 0
    kl_dist = 0
    cos_dist = 0
    for i, data in enumerate(loader):
        # print(i)
        features = data['features']
        label = data['labels']

        y_estimated = model(features)

        error += ((y_estimated - label).abs().sum())
        
        canberra_distance += canberra(y_estimated.cpu().detach().numpy(), label.reshape(-1).cpu().detach().numpy())
        kl_dist += F.kl_div(torch.log(y_estimated.reshape((-1, 11))), label.reshape((-1, 11)).float(),
                            reduction='batchmean').float()
        chebyshev_distance += chebyshev(y_estimated.cpu().detach().numpy(), label.reshape(-1).cpu().detach().numpy())
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dist += cos(y_estimated, label.reshape(-1))
    return error/len(loader.dataset), canberra_distance/len(loader.dataset), kl_dist/len(loader.dataset), \
           chebyshev_distance/len(loader.dataset), cos_dist/len(loader.dataset)


training_batch_size = 8

train_len = int(len(function_zone_dataset) // 10 * 8)
test_len = len(function_zone_dataset) - train_len
train_set, test_set = random_split(function_zone_dataset, [train_len, test_len])

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
train_loader = DataLoader(train_set, batch_size=training_batch_size, shuffle=False, num_workers=0)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):

    loss, p_a = train()
    test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist = test(test_loader)

    if (epoch % 5 == 0) or (epoch == 1) or (epoch == 99):
        print('Epoch: {:03d}, p_a: {:7f}, Loss: {:.7f}, '
              'Test MAE: {:.7f}, canberra_dist:{:.7f}, kl_dist:{:.7f}, chebyshev_distance:{:.7f}, cos_distance:{:.7f}'.
              format(epoch, p_a, loss, test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist))


