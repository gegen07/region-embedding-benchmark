import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
from shapely import wkt
import networkx as nx

import sys
import os
import pickle as pkl

import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

from sklearn.preprocessing import LabelEncoder
import geo_functions as geo

from libpysal import weights
from libpysal.cg import voronoi_frames


COLUMN_INDEX = "GEOID"

class Util:
    def __init__(self) -> None:
        pass

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
    def intra_inter_region_transition(poi1, poi2, column=COLUMN_INDEX):
        if poi1[column] == poi2[column]:
            return 1
        else:
            return 0.5
        

class PreProcess:
    def __init__(self, filename_pois, filename_boroughs, h3=False):
        self.filename_pois = filename_pois
        self.filename_boroughs = filename_boroughs
        self.h3 = h3

    def read_poi_data(self):
        self.pois = pd.read_csv(self.filename_pois)
        self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
        self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
        self.pois["geometry"] = self.pois["geometry"].apply(lambda x: x if x.geom_type == "Point" else x.centroid)

        self.pois = gpd.sjoin(self.pois, self.boroughs, how="inner", predicate='intersects')

    def read_boroughs_data(self):
        self.boroughs = pd.read_csv(self.filename_boroughs)
        self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

        if self.h3:
            self.boroughs = geo.H3Interpolation(self.boroughs).interpolate(9)
        print(self.boroughs)

    def encode_categories(self):
        first_level = LabelEncoder()
        self.second_level = LabelEncoder()
        self.pois["category"] = first_level.fit_transform(self.pois["category"].values)
        self.pois["fclass"] = self.second_level.fit_transform(self.pois["fclass"].values)

    def create_graph(self):

        column = 'BoroCT2020'
        if self.h3:
            column = "h3"
        
        points = np.array(self.pois.geometry.apply(lambda x: [x.x, x.y]).tolist())
        D = Util.diagonal_length_min_box(self.pois.geometry.unary_union.envelope.bounds)
        print(D)

        triangles = scipy.spatial.Delaunay(points, qhull_options="QJ QbB Pp").simplices

        G = nx.Graph()
        G.add_nodes_from(range(len(points)))

        from itertools import combinations

        for simplex in triangles:
            comb = combinations(simplex, 2)
            for x, y in comb:
                if not G.has_edge(x, y):
                    dist = geo.haversine_np(*points[x], *points[y])
                    w1 = np.log((1+D**(3/2))/(1+dist**(3/2)))
                    w2 = Util.intra_inter_region_transition(
                        self.pois.iloc[x], 
                        self.pois.iloc[y],
                        column=column
                    )
                    G.add_edge(x, y, weight=w1*w2)

        self.edges = nx.to_pandas_edgelist(G)
        mi = self.edges['weight'].min()
        ma = self.edges['weight'].max()
        self.edges['weight'] = self.edges['weight'].apply(lambda x: (x-mi)/(ma-mi))
    
    def save_data(self):
        self.pois.to_csv('./data/pois.csv', index=False)
        self.edges.to_csv('./data/edges.csv', index=False)

    def run(self):
        self.read_boroughs_data()
        self.read_poi_data()
        self.encode_categories()
        self.create_graph()
        self.save_data()
    
class POI2Vec:
    def __init__(self):
        if os.path.exists("./data/pois.csv") and os.path.exists("./data/edges.csv"):
            self.pois = pd.read_csv("./data/pois.csv")
            self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
            self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")

            self.edges = pd.read_csv("./data/edges.csv")
            print(self.edges)
        else:
            raise FileNotFoundError("Files not found. Run Preprocess first.")
        
        points = np.array(self.pois.geometry.apply(lambda x: [x.x, x.y]).tolist())

        self.data = Data(x=torch.tensor(points, dtype=torch.float), 
                         edge_index=torch.tensor(self.edges[["source", "target"]].T.values, 
                                                 dtype=torch.long), 
                         edge_weight=torch.tensor(self.edges["weight"].values, dtype=torch.float))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Node2Vec(
            self.data.edge_index,
            embedding_dim=64,
            walk_length=10,
            context_size=5,
            walks_per_node=5,
            num_negative_samples=2,
            p=0.5,
            q=0.5,
            sparse=True,
        ).to(self.device)

        self.num_workers = 4 if sys.platform == 'linux' else 0

        self.second_class_number = self.pois["fclass"].nunique()        

    def train(self):
        loader = self.model.loader(batch_size=128, shuffle=True, num_workers=self.num_workers)

        self.second_class_walks = []
        for idx, (pos_rw, neg_rw) in enumerate(loader):
            for walk in pos_rw:
                self.second_class_walks.append([])
                for poi_idx in walk.tolist():
                    second_class = self.pois.iloc[poi_idx]["fclass"]
                    self.second_class_walks[-1].append(second_class)
    
    def save_walks(self):
        pkl.dump(self.second_class_walks, open("./data/second_class_walks.pkl", "wb"))
    
    def read_walks(self):
        self.second_class_walks = pkl.load(open("./data/second_class_walks.pkl", "rb"))
    
    def get_global_second_class_walks(self):
        self.global_second_class_walks = []
        for i_temp in range(self.second_class_number):
            self.global_second_class_walks.append([])
        for second_class_walk in self.second_class_walks:
            self.global_second_class_walks[int(second_class_walk[0])].extend(second_class_walk[1:])
        for i_temp in range(self.second_class_number):
            self.global_second_class_walks[i_temp] = list(set(self.global_second_class_walks[i_temp]))
