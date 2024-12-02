import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.nn.pool import global_mean_pool

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()

        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        X = self.gcn1(data.x, data.edge_index)
        X = self.norm1(X)
        X = self.prelu(X)
        X = self.gcn2(X, data.edge_index)
        X = self.norm2(X)
        X = self.prelu(X)

        return X
    
class DGIEmbeddingCTA(nn.Module):
    def __init__(self, embedding_size):
      super(DGIEmbeddingCTA, self).__init__()

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

      self.feature_building_encoder = DeepGraphInfomax(
          hidden_channels=128,
          encoder=Encoder(64, 128),
          summary=self._summary_fn,
          corruption=self.corruption,
      )

      self.pois_category_projector = nn.Sequential(
          nn.Linear(401, 512),
          nn.PReLU(),
          nn.Linear(512, 64),
          nn.PReLU(),
      )
      self.feature_pois_encoder = DeepGraphInfomax(
          hidden_channels=128,
          encoder=Encoder(64, 128),
          summary=self._summary_fn,
          corruption=self.corruption,
      )

    def reset_parameters(self):
      torch.nn.init.uniform_(self.weight_building)
      torch.nn.init.uniform_(self.weight_pois)

    def corruption(self, data):

      clone_data = data.clone()
      x = clone_data.x[torch.randperm(clone_data.x.size(0))]
      x += 0.1 * torch.randn_like(x)
      data.x = x

      return data

    def _summary_fn(self, z, data):
        s = global_mean_pool(z, data.batch)
        return s

    def forward(self, building_data, pois_data):
      category_features = self.building_category_projector(building_data.building_categories)
      descriptor_features = self.building_descriptor(building_data.features_descriptors)
      building_data.x = torch.cat((category_features, descriptor_features), dim=1)
      building_data.x = self.building_features_linear(building_data.x)

      pos_feat_build, neg_feat_build, graph_build_orig = self.feature_building_encoder(building_data)

      poi_category_features = self.pois_category_projector(pois_data.categories)
      pois_data.x = poi_category_features

      pos_feat_pois, neg_feat_pois, graph_pois_orig = self.feature_pois_encoder(pois_data)

      return pos_feat_build, neg_feat_build, graph_build_orig, pos_feat_pois, neg_feat_pois, graph_pois_orig

    def loss(self, pos_feat_build, neg_feat_build, graph_build_orig, pos_feat_pois, neg_feat_pois, graph_pois_orig):
      build_loss = self.feature_building_encoder.loss(pos_feat_build, neg_feat_build, graph_build_orig)
      pois_loss = self.feature_pois_encoder.loss(pos_feat_pois, neg_feat_pois, graph_pois_orig)
      return build_loss + pois_loss

    def input_embedding(self, buildings_data, pois_data):
      _, _, build_emb, _, _, pois_emb = self.forward(buildings_data, pois_data)
      return build_emb+pois_emb