import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from collections import defaultdict
def main(h3=False):
    if h3:
        column = 'h3'
        name_index = 'region_id'
    else:
        column = 'GEOID'
        name_index = column

    embeddings = torch.load('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/poi-encoder/data/poi-encoder-nyc-h3.tensor', map_location=torch.device('cpu'))['in_embed.weight']
    pois = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/poi-encoder/data/pois-nyc-h3.csv')[['fclass', column]]

    # pois['embedding'] = pois['fclass'].apply(lambda x: np.array(embeddings[x]))
    pois['embedding'] = pois['fclass'].apply(lambda x: embeddings[x])
    cta = pois[column].unique().tolist()

    d = defaultdict(list)
    for x in cta:
        emb = pois[pois[column] == x]['embedding'].apply(np.array).mean(axis=0)
        d[x] = emb
    region = pd.DataFrame.from_dict(d, orient='index', columns=[str(i) for i in range(64)])
    region.index.name = name_index
    region = region.reset_index()
    region.to_csv('./data/region_embedding-nyc-h3.csv', index=False)
    # region.to_parquet('./data/region_embedding.parquet', index=False)
    
if __name__ == "__main__":
    main(h3=True)