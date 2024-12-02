import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.models import GAE
from torch_geometric.utils import negative_sampling


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GAEEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(GAEEmbeddingModel, self).__init__()

        self.embedding_size = embedding_size

        self.building_category_projector = nn.Sequential(
            nn.Linear(110, 512),
            nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
        )
        self.building_descriptor = nn.Sequential(
            nn.Linear(13, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
        )
        self.building_features_linear = nn.Sequential(
            nn.Linear(96, 512),
            nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
        )

        self.pois_category_projector = nn.Sequential(
            nn.Linear(401, 512),
            nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
        )

        # Use PyTorch Geometric's GAE with GCN encoders
        self.model_building = GAE(encoder=GCNEncoder(in_channels=64, hidden_channels=64, out_channels=32))
        self.model_pois = GAE(encoder=GCNEncoder(in_channels=64, hidden_channels=64, out_channels=32))

    def forward(self, buildings, pois):
        # Project and encode building data
        buildings.building_categories = self.building_category_projector(buildings.building_categories)
        buildings.features_descriptors = self.building_descriptor(buildings.features_descriptors)
        buildings.x = torch.cat((buildings.building_categories, buildings.features_descriptors), dim=1)
        buildings.x = self.building_features_linear(buildings.x)

        z_building = self.model_building.encode(buildings.x, buildings.edge_index)
        adj_pred_building = self.model_building.decode(z_building, buildings.edge_index)

        # Project and encode POI data
        pois.x = self.pois_category_projector(pois.categories)
        z_pois = self.model_pois.encode(pois.x, pois.edge_index)
        adj_pred_pois = self.model_pois.decode(z_pois, pois.edge_index)

        return z_building, adj_pred_building, z_pois, adj_pred_pois

    def input_embedding(self, buildings, pois):
      building_emb, _, pois_emb, _ = self.forward(buildings, pois)

      graph_embedding_buildings = global_mean_pool(building_emb, buildings.batch)
      graph_embedding_pois = global_mean_pool(pois_emb, pois.batch)
      
      return graph_embedding_buildings+graph_embedding_pois

    def loss(self, adj_pred_building, edge_index_building, adj_pred_pois, edge_index_pois):
      # Positive edge labels
      pos_labels_building = torch.ones(edge_index_building.size(1), device=z_building.device)
      pos_labels_pois = torch.ones(edge_index_pois.size(1), device=z_pois.device)

      # Generate negative edges
      neg_edge_index_building = negative_sampling(edge_index_building, num_nodes=z_building.size(0))
      neg_edge_index_pois = negative_sampling(edge_index_pois, num_nodes=z_pois.size(0))

      # Negative edge labels
      neg_labels_building = torch.zeros(neg_edge_index_building.size(1), device=z_building.device)
      neg_labels_pois = torch.zeros(neg_edge_index_pois.size(1), device=z_pois.device)

      # Combine positive and negative labels
      labels_building = torch.cat([pos_labels_building, neg_labels_building], dim=0)
      labels_pois = torch.cat([pos_labels_pois, neg_labels_pois], dim=0)

      # Get the edge scores (logits) from the model
      pos_pred_building = self.model_building.decode(z_building, edge_index_building)
      neg_pred_building = self.model_building.decode(z_building, neg_edge_index_building)
      preds_building = torch.cat([pos_pred_building, neg_pred_building], dim=0)

      pos_pred_pois = self.model_pois.decode(z_pois, edge_index_pois)
      neg_pred_pois = self.model_pois.decode(z_pois, neg_edge_index_pois)
      preds_pois = torch.cat([pos_pred_pois, neg_pred_pois], dim=0)

      # Binary cross-entropy loss with logits
      loss1 = F.binary_cross_entropy_with_logits(preds_building, labels_building)
      loss2 = F.binary_cross_entropy_with_logits(preds_pois, labels_pois)

      return loss1 + loss2
