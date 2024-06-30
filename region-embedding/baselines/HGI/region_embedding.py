import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from collections import defaultdict
def main():
    embeddings = torch.load('./data/ny_emb-best.torch').numpy()
    cta = pd.read_csv('../../data/cta_nyc.csv')
    
    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [str(i) for i in range(64)]
    emb_df['BoroCT2020'] = cta['BoroCT2020']

    emb_df.to_csv('./data/region_embedding.csv', index=False)
    
if __name__ == "__main__":
    main()