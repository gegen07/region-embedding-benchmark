import torch
from torch_geometric.loader import DataLoader

data_list_buildings, pois_data_list, region_data = read_data()
graph_dataset = GraphDataset(data_list_buildings, pois_data_list)
data_loader = DataLoader(graph_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cta_embedding_model = CTAEmbeddingModel(64)
weight_decay = 5e-4
learning_rate = 1e-5
optimizer = torch.optim.Adam(cta_embedding_model.parameters(), lr=learning_rate)

for epoch in range(5):
    optimizer.zero_grad()
    total_loss = 0
    for data in data_loader:
        buildings, pois = data[0], data[1]
        buildings = buildings.to(device)
        pois = pois.to(device)

        z_building, adj_pred_building, z_pois, adj_pred_pois = cta_embedding_model(buildings, pois)

        # adj_true_building = to_dense_adj(buildings.edge_index, batch=buildings.batch)
        # adj_true_pois = to_dense_adj(pois.edge_index, batch=pois.batch)
        loss = cta_embedding_model.loss(adj_pred_building, buildings.edge_index, adj_pred_pois, pois.edge_index)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}')