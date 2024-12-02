import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from shapely import wkt
import networkx as nx
import h3
from h3ronpy.arrow import cells_to_string, grid_disk
from h3ronpy.arrow.vector import ContainmentMode, cells_to_wkb_polygons, wkb_to_cells
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from collections.abc import Iterable

from sklearn.preprocessing import LabelEncoder
import torch

import numpy as np
import pickle as pkl
import os

COLUMN_INDEX = "GEOID"

class H3Interpolation():

    def __init__(self, gdf) -> None:
        self.gdf = gdf 
    
    def _shapely_geometry_to_h3(
        self,
        geometry,
        h3_resolution: int,
        buffer: bool = True,
    ) -> list[str]:
        if not (0 <= h3_resolution <= 15):
            raise ValueError(f"Resolution {h3_resolution} is not between 0 and 15.")

        wkb = []
        if isinstance(geometry, gpd.GeoSeries):
            wkb = geometry.to_wkb()
        elif isinstance(geometry, gpd.GeoDataFrame):
            wkb = geometry['geometry'].to_wkb()
        elif isinstance(geometry, Iterable):
            wkb = [sub_geometry.wkb for sub_geometry in geometry]
        else:
            wkb = [geometry.wkb]

        containment_mode = (
            ContainmentMode.IntersectsBoundary if buffer else ContainmentMode.ContainsCentroid
        )
        h3_indexes = wkb_to_cells(
            wkb, resolution=h3_resolution, containment_mode=containment_mode, flatten=True
        ).unique()

        return [h3.int_to_str(h3_index) for h3_index in h3_indexes.tolist()]

    def _h3_to_geoseries(self, h3_index):
        if isinstance(h3_index, (str, int)):
            return self.h3_to_geoseries([h3_index])
        else:
            h3_int_indexes = (
                h3_cell if isinstance(h3_cell, int) else h3.str_to_int(h3_cell) for h3_cell in h3_index
            )
            return gpd.GeoSeries.from_wkb(cells_to_wkb_polygons(h3_int_indexes), crs=4326)
    
    def interpolate(self, h3_resolution: int = 9, buffer: bool = True):
        self.gdf = self.gdf.explode(index_parts=True).reset_index(drop=True)
        h3_list = list(set(self._shapely_geometry_to_h3(self.gdf['geometry'], h3_resolution)))

        return gpd.GeoDataFrame(
            data={"h3": h3_list},
            geometry=self._h3_to_geoseries(h3_list),
            crs=4326,
        )

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
    def intra_inter_region_transition(poi1, poi2, column=COLUMN_INDEX):
        if poi1[column] == poi2[column]:
            return 1
        else:
            return 0.5

class Preprocess():
    def __init__(self, pois_filename, boroughs_filename, emb_filename, h3=False) -> None:
        self.pois_filename = pois_filename
        self.boroughs_filename = boroughs_filename
        self.embedding_filename = emb_filename
        self.h3 = h3


    def _read_poi_data(self):
        self.pois = pd.read_csv(self.pois_filename)
        self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
        self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
        self.pois["geometry"] = self.pois["geometry"].apply(lambda x: x if x.geom_type == "Point" else x.centroid)

        if self.h3:
          self.regions_unique_index = self.pois.h3.unique().tolist()
        else:
          self.regions_unique_index = self.pois.index_right.unique().tolist()
    
    def _read_boroughs_data(self):
        self.boroughs = pd.read_csv(self.boroughs_filename)
        self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

        if self.h3:
            self.boroughs = H3Interpolation(self.boroughs).interpolate(8)
            self.boroughs = self.boroughs[self.boroughs.h3.isin(self.regions_unique_index)]
            self.boroughs = self.boroughs.drop_duplicates(subset=['h3'], keep='first')
        else:
            self.boroughs = self.boroughs[self.boroughs.index.isin(self.regions_unique_index)]

        self.boroughs = self.boroughs.reset_index(drop=True)

        self.boroughs.to_csv('/content/chicago_boroughs_h3_to_embeddings.csv', index=False)

        self.pois = self.pois[['feature_id', 'category', 'fclass', 'geometry']].sjoin(self.boroughs, how='inner', predicate='intersects')

        self.n_regions = len(self.boroughs)

    def _read_embedding(self):
        emb = torch.load(self.embedding_filename)['in_embed.weight'] ## TODO: change this in_embed.weight to general way of reading the embedding
        self.pois['embedding'] = self.pois['fclass'].apply(lambda x: list(np.array(emb[x])))
        self.embedding_array = self.pois['embedding'].values.tolist()

    def _create_graph(self):
        if os.path.exists('/content/edges.csv'):
            self.edges = pd.read_csv('/content/edges.csv')
            return

        column = 'BoroCT2020'
        if self.h3:
            column = "h3"

        print(self.pois)
        points = np.array(self.pois.geometry.apply(lambda x: [x.x, x.y]).tolist())
        D = Util.diagonal_length_min_box(self.pois.geometry.unary_union.envelope.bounds)

        triangles = scipy.spatial.Delaunay(points, qhull_options="QJ QbB Pp").simplices

        G = nx.Graph()
        G.add_nodes_from(range(len(points)))

        from itertools import combinations

        for simplex in triangles:
            comb = combinations(simplex, 2)
            for x, y in comb:
                if not G.has_edge(x, y):
                    dist = Util.haversine_np(*points[x], *points[y])
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

        self.edges.to_csv('/content/edges.csv', index=False)
    
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

        if os.path.exists('/content/region_coarse_similarity.npy'):
            self.region_coarse_similarity = np.load('/content/region_coarse_similarity.npy')
            return

        onehot = self.pois[['index_right', 'fclass']]
        onehot = pd.concat([onehot[['index_right']],pd.get_dummies(onehot['fclass'], dtype=int)], axis=1)

        arr = onehot.values

        ## Cosine Similarity using arr
        from sklearn.metrics.pairwise import cosine_similarity
        self.region_coarse_similarity = cosine_similarity(arr)

        with open('/content/region_coarse_similarity.npy', 'wb') as f:
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
    pois_filename = "/content/drive/MyDrive/Dados/region-embedding-benchmark/pois.csv"
    boroughs_filename = "/content/drive/MyDrive/Dados/region-embedding-benchmark/data/chicago/cta_chicago.csv"
    # edges_filename = "../../poi-encoder/data/edges.csv"
    emb_filename = "/content/poi-encoder-chicago-h3.tensor"
    pre = Preprocess(pois_filename, boroughs_filename, emb_filename, h3=True)
    data = pre.get_data_torch()
    # print(data)

    with open("/content/ny_hgi_data.pkl", "wb") as f:
        pkl.dump(data, f)

    print("Data saved to /content/ny_hgi_data.pkl")

