import pandas as pd
import numpy as np
import pickle as pkl

def main():
    emb = pd.DataFrame.from_dict(pd.read_pickle('./data/embeddings/RegionDCL_10-h3-chicago.pkl'), orient='index')
    # emb = emb.rename(columns={'index': 'TRACTCE'})
    # emb['BoroCT2020'] = emb['BoroCT2020'].astype(str)
    # emb.to_csv('./data/region_embedding.csv', index=False)
    # print(emb.index)
    cta = pd.read_csv('/home/gegen07/dev/projects/master-region-embedding/src/data/cta_chicago-h3.csv')
    print(emb)
    cta[['h3']].merge(emb, left_index=True, right_index=True).sort_values('h3').to_csv('./data/region_embedding-chicago-h3.csv', index=False)


if __name__ == '__main__':
    main()