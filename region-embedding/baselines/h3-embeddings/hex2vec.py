from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from filters import REDUCED_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

loader = OSMPbfLoader(download_source="geofabrik")
regionalizer = H3Regionalizer(resolution=9)
joiner = IntersectionJoiner()

area = geocode_to_region_gdf("New York City, United States")
features = loader.load(area, REDUCED_FILTER)
regions = regionalizer.transform(area)
joint = joiner.transform(regions, features)

embedder = Hex2VecEmbedder()
neighbourhood = H3Neighbourhood(regions_gdf=regions)

embedder = Hex2VecEmbedder([64, 32, 16, 64])

trainer_kwargs = {"max_epochs": 5, "fast_dev_run": False}
embeddings = embedder.fit_transform(regions, features, joint, neighbourhood, batch_size=128, trainer_kwargs=trainer_kwargs)

embeddings.to_csv("./data/ny_hex2vec_emb.csv")