import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from shapely import wkt


def get_landuse_default():
    df = pd.read_csv('./data/catagory-buildings.csv.gz') ## borough, landuse, latitude, longitude
    
    gdf_aux = pd.read_csv('../../data/cta_nyc.csv').to_crs(4326)
    gdf_aux['geometry'] = gdf_aux['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf_aux, geometry=gdf_aux['geometry'], crs=4326)
    gdf = gdf[['BoroCT2020', 'GEOID', 'geometry']]
    
    gdf_buildings = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    gdf_buildings = gdf_buildings.sjoin(gdf, how='inner', op='within')
  
    count = gdf_buildings.groupby(['BoroCT2020', 'landuse']).count().unstack()['borough'].fillna(0)
    count['sum'] = count.sum(axis=1)
    count = count.div(count['sum'], axis=0)

    columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'sum']
    count.columns = columns

    count[columns[:-1]].to_csv('./data/nyc-landuse.csv')

def get_landuse_h3(resolution=9):
    df = pd.read_csv('./data/category-buildings.csv.gz') ## borough, landuse, latitude, longitude

    gdf_buildings = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    gdf_buildings = gdf_buildings.dropna(axis=0, subset=['latitude', 'longitude', 'landuse'])
    gdf_buildings['region_id'] = gdf_buildings.apply(lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, resolution), axis=1)
    
    count = gdf_buildings.groupby(['region_id', 'landuse']).count().unstack()['borough'].fillna(0)
    count['sum'] = count.sum(axis=1)
    count = count.div(count['sum'], axis=0)

    columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'sum']
    count.columns = columns

    count[columns[:-1]].to_csv(f'./data/nyc-landuse-res-{resolution}.csv')

if __name__ == "__main__":
    get_landuse_h3(9)