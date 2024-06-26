import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import pandas as pd
from scipy.spatial.distance import canberra, chebyshev


class FunctionZoneDataset(Dataset):

    def __init__(self, csv_file_features, csv_file_labels, column_id):
        self.features = pd.read_csv(csv_file_features)
        self.labels = pd.read_csv(csv_file_labels)

        boroughs = self.labels[column_id].unique().tolist()

        self.features = self.features[self.features[column_id].isin(boroughs)]


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features.iloc[idx, 0:64].values.flatten().astype('float32')
        labels = self.labels.iloc[idx, 1:].values.flatten().astype('float32')

        return {'features': torch.from_numpy(features), 'labels': torch.from_numpy(labels)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

zone_list = []

ground_truth_file = "./data/nyc-landuse-5-classes.csv"
region_embedding_file = '/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/HGI/data/region_embedding.csv'
# region_embedding_file = '/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/data/region_embedding-info-nce.csv'

column_id = 'BoroCT2020'
# column_id = 'region_id'

function_zone_dataset = FunctionZoneDataset(region_embedding_file, ground_truth_file, column_id)

embedding_size = 64

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 5),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.net(x.float())
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

        loss = F.kl_div(torch.log(y_estimated.reshape((-1, 5))), label.reshape((-1, 5)).float(), 
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
        kl_dist += F.kl_div(torch.log(y_estimated.reshape((-1, 5))), label.reshape((-1, 5)).float(),
                            reduction='batchmean').float()
        chebyshev_distance += chebyshev(y_estimated.cpu().detach().numpy(), label.reshape(-1).cpu().detach().numpy())
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dist += cos(y_estimated, label.reshape(-1))
    return error/len(loader.dataset), canberra_distance/len(loader.dataset), kl_dist/len(loader.dataset), \
           chebyshev_distance/len(loader.dataset), cos_dist/len(loader.dataset)


training_batch_size = 16

train_len = int(len(function_zone_dataset) // 10 * 8)
test_len = len(function_zone_dataset) - train_len
train_set, test_set = random_split(function_zone_dataset, [train_len, test_len])

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
train_loader = DataLoader(train_set, batch_size=training_batch_size, shuffle=False, num_workers=0)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(200):

    loss, p_a = train()
    test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist = test(test_loader)

    if (epoch % 5 == 0) or (epoch == 1) or (epoch == 99):
        print('Epoch: {:03d}, p_a: {:7f}, Loss: {:.7f}, '
              'Test MAE: {:.7f}, canberra_dist:{:.7f}, kl_dist:{:.7f}, chebyshev_distance:{:.7f}, cos_distance:{:.7f}'.
              format(epoch, p_a, loss, test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist))


