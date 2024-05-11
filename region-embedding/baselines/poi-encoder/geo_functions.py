import pandas
import geopandas
import h3
import numpy as np
from shapely import wkt
from h3ronpy.arrow import cells_to_string, grid_disk
from h3ronpy.arrow.vector import ContainmentMode, cells_to_wkb_polygons, wkb_to_cells
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from collections.abc import Iterable

__all__ = [
    "haversine_np"
]

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
        if isinstance(geometry, geopandas.GeoSeries):
            wkb = geometry.to_wkb()
        elif isinstance(geometry, geopandas.GeoDataFrame):
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
            return geopandas.GeoSeries.from_wkb(cells_to_wkb_polygons(h3_int_indexes), crs=4326)
    
    def interpolate(self, h3_resolution: int = 9, buffer: bool = True):
        self.gdf = self.gdf.explode(index_parts=True).reset_index(drop=True)
        h3_list = list(set(self._shapely_geometry_to_h3(self.gdf['geometry'], h3_resolution)))

        return geopandas.GeoDataFrame(
            data={"h3": h3_list},
            geometry=self._h3_to_geoseries(h3_list),
            crs=4326,
        )