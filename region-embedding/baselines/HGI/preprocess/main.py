import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from shapely import wkt
import networkx as nx

from sklearn.preprocessing import LabelEncoder
import torch

import numpy as np
import pickle as pkl
import os

class Preprocess():
    def __init__(self, pois_filename, boroughs_filename, edges_filename, emb_filename) -> None:
        self.pois_filename = pois_filename
        self.boroughs_filename = boroughs_filename
        self.edges_filename = edges_filename
        self.embedding_filename = emb_filename


    def _read_poi_data(self):
        self.pois = pd.read_csv(self.pois_filename)
        self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
        self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
        self.pois["geometry"] = self.pois["geometry"].apply(lambda x: x if x.geom_type == "Point" else x.centroid)
    
    def _read_boroughs_data(self):
        self.boroughs = pd.read_csv(self.boroughs_filename)
        self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

        self.n_regions = len(self.boroughs)

    def _read_embedding(self):
        emb = torch.load(self.embedding_filename)['in_embed.weight'] ## TODO: change this in_embed.weight to general way of reading the embedding
        self.pois['embedding'] = self.pois['fclass'].apply(lambda x: list(np.array(emb[x])))
        self.embedding_array = self.pois['embedding'].values.tolist()
        # print(self.embedding_array)

    def _read_edges(self):
        self.edges = pd.read_csv(self.edges_filename)
    
    def _get_region_adjacency(self):
        import libpysal

        polygons_boroughs = self.boroughs.geometry
        adj = libpysal.weights.fuzzy_contiguity(polygons_boroughs)
        self.adj_list = adj.to_adjlist(remove_symmetric=False)

    
    def _get_coarse_region_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity

        if os.path.exists('../data/region_coarse_similarity.npy'):
            self.region_coarse_similarity = np.load('../data/region_coarse_similarity.npy')
            return

        onehot = self.pois[['index_right', 'fclass']]
        onehot = pd.concat([onehot[['index_right']],pd.get_dummies(onehot['fclass'], dtype=int)], axis=1)

        self.region_coarse_similarity = np.zeros((self.n_regions, self.n_regions))
        bor = np.unique(onehot.index).tolist()
        for x in range(self.n_regions):
            for y in range(self.n_regions):
                if (x not in bor) or (y not in bor):
                    sim = 0
                elif x != y:
                    sim = cosine_similarity([onehot[onehot.index == x].values[0]], [onehot[onehot.index == y].values[0]])
                    sim = sim[0][0]
                else:
                    sim = 1
                self.region_coarse_similarity[x, y] = sim

        with open('../data/region_coarse_similarity.npy', 'wb') as f:
            np.save(f, self.region_coarse_similarity)
    
    def get_data_torch(self):
        print("reading poi data")
        self._read_poi_data()
        
        print("reading boroughs data")
        self._read_boroughs_data()
        
        print("edges from graph")
        self._read_edges()
        
        print("reading embedding")
        self._read_embedding()
        
        print("creating region adjacency")
        self._get_region_adjacency()

        print("creating region similarity by cosine similarity of embeddings")
        self._get_coarse_region_similarity()
        
        data = {}
        data['node_features'] = self.embedding_array
        data['edge_index'] = self.edges[["source", "target"]].T.values
        data['edge_weight'] = self.edges["weight"].values
        data['region_id'] = self.adj_list['focal'].values
        data['coarse_region_similarity'] = self.region_coarse_similarity
        data['region_area'] = ((self.boroughs.to_crs(3857).area)/(10**6)).values
        data['region_adjacency'] = self.adj_list[['focal', 'neighbor']].T.values
        return data
    
if __name__ == "__main__":
    pois_filename = "../../poi-encoder/data/pois.csv"
    boroughs_filename = "../../../data/cta_nyc.csv"
    edges_filename = "../../poi-encoder/data/edges.csv"
    emb_filename = "../../poi-encoder/data/poi-encoder.tensor"
    pre = Preprocess(pois_filename, boroughs_filename, edges_filename, emb_filename)
    data = pre.get_data_torch()
    print(data)

    with open("../data/ny_hgi_data.pkl", "wb") as f:
        pkl.dump(data, f)

    print("Data saved to ../data/ny_hgi_data.pkl")

