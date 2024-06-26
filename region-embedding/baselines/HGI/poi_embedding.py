import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from collections import defaultdict
def main():
    embeddings = torch.load('./data/ny.torch').numpy()
    pois = pd.read_csv('../../data/new-york-pois.csv.gz', compression='gzip')
    print(pois.columns)
    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [str(i) for i in range(64)]
    emb_df['feature_id'] = pois['feature_id']
    emb_df['fclass'] = pois['fclass']
    emb_df['category'] = pois['category']

    emb_df.to_csv('./data/poi_embedding.csv', index=False)
    
if __name__ == "__main__":
    main()