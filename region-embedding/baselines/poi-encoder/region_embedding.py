import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from collections import defaultdict
def main():
    embeddings = torch.load('./data/poi-encoder.tensor')['in_embed.weight']
    pois = pd.read_csv('./data/pois.csv')[['fclass', 'BoroCT2020']]

    pois['embedding'] = pois['fclass'].apply(lambda x: np.array(embeddings[x]))
    cta = pois['BoroCT2020'].unique().tolist()

    d = defaultdict(list)
    for x in cta:
        emb = pois[pois['BoroCT2020'] == x]['embedding'].apply(np.array).mean(axis=0)
        d[x] = emb
    region = pd.DataFrame.from_dict(d, orient='index', columns=[str(i) for i in range(64)])
    region.index.name = 'BoroCT2020'
    region = region.reset_index()
    region.to_csv('./data/region_embedding.csv', index=False)
    
if __name__ == "__main__":
    main()