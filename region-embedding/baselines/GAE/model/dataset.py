from torch.utils.data import Dataset
class GraphDataset(Dataset):
    def __init__(self, buildings_data_list, pois_data_list):
        self.building_data = buildings_data_list
        self.pois_data = pois_data_list

    def __getitem__(self, index):
        return self.building_data[index], self.pois_data[index]

    def __len__(self):
        return len(self.building_data)