import pandas as pd
import geopandas as gpd
from shapely import wkt, wkb
import momepy
import shapely
import numpy as np
import osmnx as ox
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, streets_parquet, buildings_parquet, pois_csv, region_name) -> None:
        self.streets_parquet_filename = streets_parquet
        self.buildings_parquet_filename = buildings_parquet
        self.pois_csv_filename = pois_csv
        self.region_name = region_name


    def _get_boundary(self):
        self.gdf_region = ox.geocode_to_gdf(self.region_name)

        self.name = ("".join(self.gdf_region.name.iloc[0])).lower().replace(" ", "-")
        self.geom = self.gdf_region.geometry.iloc[0]

        self.gdf_region.to_crs(3857).to_file('./data/raw/boundary/boundary.shp', driver='ESRI Shapefile')

    def _get_buildings(self):
        buildings = pd.read_parquet(self.buildings_parquet_filename)
        buildings = gpd.GeoDataFrame(buildings, geometry=buildings["geometry"].apply(wkb.loads), crs=4326)
        buildings = buildings[buildings.geom_type=="Polygon"]
        

        buildings = buildings.sjoin(self.gdf_region[['geometry']].to_crs('4326'))[['feature_id', 'building', 'geometry']].reset_index(drop=True)
        buildings.columns = ['featureid', 'type', 'geometry']
        buildings[['type', 'geometry']].to_crs(3857).to_file('./data/raw/buildings/building.shp')

    def _get_segmentation(self, threshold=2000):
        streets = pd.read_parquet(self.streets_parquet_filename)
        streets = gpd.GeoDataFrame(streets, geometry=streets["geometry"].apply(wkb.loads), crs=4326)
        streets = streets[streets.geom_type=="LineString"]

        linestrings = streets.geometry
        collection = shapely.GeometryCollection(linestrings.array)
        noded = shapely.node(collection) 
        polygonized = shapely.polygonize(noded.geoms)
        polygons = gpd.GeoSeries(polygonized.geoms)
        poly_buff = polygons.set_crs(4326).to_crs(3857).buffer(-10).to_crs(4326).reset_index()

        poly_buff['area'] = poly_buff.set_crs(4326).to_crs(3857).area

        poly_buff = poly_buff.sjoin(self.gdf_region[['geometry']].to_crs('4326'))

        poly_shp = poly_buff[poly_buff['area'] >= threshold][[0, 'area']]
        poly_shp.columns = ['geometry', 'area']
        poly_shp.set_geometry('geometry', inplace=True)

        poly_shp[poly_shp.geom_type=='Polygon'].reset_index(drop=True).to_crs(3857).to_file('./data/raw/segmentation/segmentation.shp')
    
    def _get_pois(self, ):
        pois = pd.read_csv(self.pois_csv_filename, compression='gzip')
        pois = gpd.GeoDataFrame(pois, geometry=pois["geometry"].apply(wkt.loads), crs=4326)
        pois = pois[pois.geom_type=="Point"]

        le = LabelEncoder()
        pois["code"] = le.fit_transform(pois["fclass"])

        pois = pois.sjoin(self.gdf_region, predicate="intersects", how="inner").reset_index(drop=True)

        pois[['fclass', 'code', 'geometry']].to_crs(3857).to_file('./data/raw/pois/pois.shp')

    def run(self):
        self._get_boundary()
        self._get_segmentation()
        self._get_buildings()
        self._get_pois()

if __name__ == "__main__":
    preprocess = Preprocess('../../data/new-york-streets-complete.parquet', '../../data/new-york-buildings.parquet', '../../data/new-york-pois.csv.gz', 'New York City, United States')
    preprocess.run()