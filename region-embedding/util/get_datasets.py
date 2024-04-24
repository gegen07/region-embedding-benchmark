import quackosm as qosm
import pandas as pd
import duckdb

import geoarrow.pyarrow as ga
from geoarrow.pyarrow import io
from quackosm import PbfFileReader

from pyarrow.parquet import write_table
import geopandas
import osmnx as ox
from quackosm._constants import GEOMETRY_COLUMN


from shapely import wkb

import os

class QuackosmData():
    def __init__(self, pbf_path, output_path='../data', region_name='New York City, United States') -> None:
        self.pbf_path = pbf_path
        self.output_path = output_path
        self.region_name = region_name

        self.connection = duckdb.connect()
        self.connection.load_extension("parquet")
        self.connection.load_extension("spatial")

        gdf_region = ox.geocode_to_gdf(self.region_name)
        self.name = ("".join(gdf_region.name.iloc[0])).lower().replace(" ", "-")
        self.geom = gdf_region.geometry.iloc[0]

    def _reader(self, parquet_file, tags_filter):
        reader = PbfFileReader(geometry_filter=self.geom, tags_filter=tags_filter)
        gpq_path = reader.convert_pbf_to_gpq(self.pbf_path)


        parquet_table = io.read_geoparquet_table(gpq_path)
        write_table(parquet_table, parquet_file)

    def get_pois_osm(self, tags_filter=None):
        print('reading pois')

        parquet_file = os.path.join(self.output_path, f"{self.name}-pois.parquet")

        if not os.path.exists(parquet_file):
            self._reader(parquet_file, tags_filter)

        print('filtering pois')
            
        df = pd.read_parquet(parquet_file)
        gdf = geopandas.GeoDataFrame(df, geometry=df["geometry"].apply(wkb.loads))

        tags_first_level_list = list(tags_filter.keys())

        cat = gdf.melt(id_vars=["feature_id"], value_vars=tags_first_level_list).dropna().reset_index(drop=True)
        cat.columns = ["feature_id", "category", "subcategory"]
        gdf = gdf[["feature_id", "geometry"]].merge(cat, on="feature_id", how="inner")

        gdf.columns = ['feature_id', 'geometry', 'category', 'fclass']

        gdf.to_csv(os.path.join(self.output_path, f"{self.name}-pois.csv.gz"), index=False)

        return gdf

    def get_streets_osm(self,):
        highways_filter = {
            "highway": [
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
            ]
        }

        parquet_file = os.path.join(self.output_path, f"{self.name}-streets.parquet")

        if not os.path.exists(parquet_file):
            print('reading streets')
            self._reader(parquet_file, highways_filter)

        print('filtering streets')
        df = pd.read_parquet(parquet_file)
        gdf = geopandas.GeoDataFrame(df, geometry=df["geometry"].apply(wkb.loads))

        gdf.to_csv(os.path.join(self.output_path, f"{self.name}-streets.csv.gz"), index=False)

        return gdf


def main():
    from filters import HEX2VEC_FILTER
    quack = QuackosmData("/media/gegen07/Expansion/data/mestrado/region-embedding/new-york-latest.osm.pbf")
    
    pois = quack.get_pois_osm(HEX2VEC_FILTER)
    print(pois.head())
    
    streets = quack.get_streets_osm()
    print(streets.head())

if __name__ == "__main__":
    main()

