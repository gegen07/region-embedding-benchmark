import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from collections import defaultdict
def main():
    embeddings = torch.load('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/HGI/data/nyc-h3/ny_h3_emb.torch').numpy()
    cta = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/HGI/data/nyc-h3/nyc_boroughs_h3_to_embeddings.csv')
    
    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [str(i) for i in range(64)]
    # emb_df['BoroCT2020'] = cta['BoroCT2020']
    emb_df['region_id'] = cta['h3']

    emb_df.to_csv('./data/region_embedding-nyc-h3.csv', index=False)
    
if __name__ == "__main__":
    main()