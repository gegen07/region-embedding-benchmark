from srai.embedders import GeoVexEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from filters import REDUCED_FILTER
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from srai.h3 import ring_buffer_h3_regions_gdf

import warnings
import torch

area_gdf = geocode_to_region_gdf("New York City, United States")
resolution = 9
k_ring_buffer_radius = 2

regionalizer = H3Regionalizer(resolution=resolution)
base_h3_regions = regionalizer.transform(area_gdf)

buffered_h3_regions = ring_buffer_h3_regions_gdf(base_h3_regions, distance=k_ring_buffer_radius)
buffered_h3_geometry = buffered_h3_regions.unary_union

tags = REDUCED_FILTER
loader = OSMPbfLoader()

features_gdf = loader.load(buffered_h3_geometry, tags)
joiner = IntersectionJoiner()
joint_gdf = joiner.transform(buffered_h3_regions, features_gdf)

neighbourhood = H3Neighbourhood(buffered_h3_regions)

embedder = GeoVexEmbedder(
    target_features=REDUCED_FILTER,
    batch_size=16,
    neighbourhood_radius=k_ring_buffer_radius,
    convolutional_layers=2,
    embedding_size=64,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    embeddings = embedder.fit_transform(
        regions_gdf=buffered_h3_regions,
        features_gdf=features_gdf,
        joint_gdf=joint_gdf,
        neighbourhood=neighbourhood,
        trainer_kwargs={
            # "max_epochs": 20, # uncomment for a longer training
            "max_epochs": 5,
            "accelerator": (
                "cpu" if torch.backends.mps.is_available() else "auto"
            ),  # GeoVexEmbedder does not support MPS
        },
        learning_rate=0.001,
    )

embeddings.to_csv("./data/ny_geovex_emb.csv")