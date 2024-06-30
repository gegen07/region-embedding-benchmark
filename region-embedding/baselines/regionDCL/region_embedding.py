import pandas as pd
import numpy as np
import pickle as pkl

def main():
    emb = pd.DataFrame.from_dict(pd.read_pickle('./data/embeddings/RegionDCL_100.pkl'), orient='index')
    print(emb.duplicated().sum())
    cta = pd.read_csv('../../data/cta_nyc.csv')

    cta[['BoroCT2020']].merge(emb, left_index=True, right_index=True).sort_values('BoroCT2020').to_csv('./data/region_embedding.csv', index=False)


if __name__ == '__main__':
    main()