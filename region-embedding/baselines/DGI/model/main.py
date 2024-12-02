import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import trange
from tqdm.auto import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

data_list_buildings, data_list_pois, region_data = read_data()
# data_list_buildings = get_structural_emb(data_list_buildings)
# data_list_pois = get_structural_emb(data_list_pois)

embedding_size = 64
cta_embedding_model = CTAEmbeddingModel(embedding_size)
cta_embedding_model.to(device)
weight_decay = 1e-4
learning_rate = 1e-3
optimizer = torch.optim.SGD(cta_embedding_model.parameters(), lr=learning_rate)

t = trange(1, 5 + 1)

# dataset = GraphDataset(data_list_buildings, data_list_pois)
# loader = DataLoader(dataset, batch_size=1, shuffle=True)

# for e in t:
#   optimizer.zero_grad()
#   torch.autograd.set_detect_anomaly(True)
#   losses = []
#   for index, data in tqdm(enumerate(loader)):
#     # print(graph_copy)
#     building_data, pois_data = copy.copy(data)
#     building_data.to(device)
#     pois_data.to(device)

#     feature_build_orig, feature_build_diff, graph_build_orig, feature_pois_orig, feature_pois_diff, graph_pois_orig = cta_embedding_model(building_data, pois_data)
#     loss = cta_embedding_model.loss(feature_build_orig, feature_build_diff, graph_build_orig, feature_pois_orig, feature_pois_diff, graph_pois_orig)
#     loss.backward()

#     torch.nn.utils.clip_grad_norm_(cta_embedding_model.parameters(), max_norm=5.0)
#     optimizer.step()

#     loss = loss.item()
#     losses.append(loss)

#     # t.set_postfix(loss='{:.4f}'.format(loss), refresh=True)

#   # print(losses)

#   print('Epoch: {:03d}, Loss: {:.4f}'.format(e, np.mean(losses)))
