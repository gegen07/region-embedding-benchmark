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

from libpysal import weights
from libpysal.cg import voronoi_frames

class Util:
    def __init__(self) -> None:
        pass

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        
        All args must be of equal length.    
        
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6378137 * c
        return km

    @staticmethod
    def diagonal_length_min_box(min_box):
        x1, y1, x2, y2 = min_box
        pt1 = (x1, y1)
        pt2 = (x2, y1)
        pt4 = (x1, y2)

        dist12 = scipy.spatial.distance.euclidean(pt1, pt2)
        dist23 = scipy.spatial.distance.euclidean(pt1, pt4)
    
        return np.sqrt(dist12**2 + dist23**2)

    @staticmethod
    def intra_inter_region_transition(poi1, poi2):
        if poi1["GEOID"] == poi2["GEOID"]:
            return 1
        else:
            return 0.4

class Preprocess():
    def __init__(self, pois_filename, boroughs_filename, emb_filename) -> None:
        self.pois_filename = pois_filename
        self.boroughs_filename = boroughs_filename
        self.embedding_filename = emb_filename


    def _read_poi_data(self):
        self.pois = pd.read_csv(self.pois_filename)
        self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
        self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
        self.pois["geometry"] = self.pois["geometry"].apply(lambda x: x if x.geom_type == "Point" else x.centroid)
        self.regions_unique_index = self.pois.index_right.unique().tolist()
    
    def _read_boroughs_data(self):
        self.boroughs = pd.read_csv(self.boroughs_filename)
        self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

        self.boroughs = self.boroughs[self.boroughs.index.isin(self.regions_unique_index)]
        self.boroughs = self.boroughs.reset_index(drop=True)

        self.pois = self.pois[['feature_id', 'category', 'fclass', 'geometry']].sjoin(self.boroughs, how='inner', predicate='intersects')

        self.n_regions = len(self.boroughs)

    def _read_embedding(self):
        emb = torch.load(self.embedding_filename)['in_embed.weight'] ## TODO: change this in_embed.weight to general way of reading the embedding
        self.pois['embedding'] = self.pois['fclass'].apply(lambda x: list(np.array(emb[x])))
        self.embedding_array = self.pois['embedding'].values.tolist()

    def _create_graph(self):
        if os.path.exists('../data/edges.csv'):
            self.edges = pd.read_csv('../data/edges.csv')
            return

        D = Util.diagonal_length_min_box(self.pois.geometry.unary_union.envelope.bounds)

        coordinates = np.column_stack((self.pois.geometry.x, self.pois.geometry.y))
        cells, generators = voronoi_frames(coordinates, clip="extent")
        delaunay = weights.Rook.from_dataframe(cells)
        G = delaunay.to_networkx()
        positions = dict(zip(G.nodes, coordinates))
       
        for edges in G.edges:
            x, y = edges
            dist = Util.haversine_np(*positions[x], *positions[y])
            w1 = np.log((1+D**(3/2))/(1+dist**(3/2)))
            w2 = Util.intra_inter_region_transition(self.pois.iloc[x], self.pois.iloc[y])
            G[x][y]['weight'] = w1*w2
        
        self.edges = nx.to_pandas_edgelist(G)
        mi = self.edges['weight'].min()
        ma = self.edges['weight'].max()
        self.edges['weight'] = self.edges['weight'].apply(lambda x: (x-mi)/(ma-mi))

        self.edges.to_csv('../data/edges.csv', index=False)
    
    def _get_region_adjacency(self):
        import libpysal

        polygons_boroughs = self.boroughs.geometry
        adj = libpysal.weights.fuzzy_contiguity(polygons_boroughs)
        self.adj_list = adj.to_adjlist(remove_symmetric=False)
    
    def _get_region_id(self):
        self.region_id = self.pois['index_right'].values.tolist()
        # print(max(self.region_id))

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

        print("creating graph")
        self._create_graph()

        print("get region ids")
        self._get_region_id()

        print("reading embedding")
        self._read_embedding()
        
        print("creating region adjacency")
        self._get_region_adjacency()

        print("creating region similarity by cosine similarity of embeddings")
        self._get_coarse_region_similarity()

        print("finishing preprocessing")
        
        data = {}
        data['node_features'] = self.embedding_array
        data['edge_index'] = self.edges[["source", "target"]].T.values
        data['edge_weight'] = self.edges["weight"].values
        data['region_id'] = self.region_id
        data['coarse_region_similarity'] = self.region_coarse_similarity
        data['region_area'] = ((self.boroughs.to_crs(3857).area)/(10**6)).values
        data['region_adjacency'] = self.adj_list[['focal', 'neighbor']].T.values

        return data
    
if __name__ == "__main__":
    pois_filename = "../../poi-encoder/data/pois.csv"
    boroughs_filename = "../../../data/cta_nyc.csv"
    # edges_filename = "../../poi-encoder/data/edges.csv"
    emb_filename = "../../poi-encoder/data/poi-encoder.tensor"
    pre = Preprocess(pois_filename, boroughs_filename, emb_filename)
    data = pre.get_data_torch()
    # print(data)

    with open("../data/ny_hgi_data.pkl", "wb") as f:
        pkl.dump(data, f)

    print("Data saved to ../data/ny_hgi_data.pkl")

