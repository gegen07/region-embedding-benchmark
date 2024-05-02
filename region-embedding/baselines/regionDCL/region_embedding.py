import pandas as pd
import numpy as np
import pickle as pkl

def main():
    emb = pd.DataFrame.from_dict(pd.read_pickle('./data/RegionDCL_20.pkl'), orient='index')
    cta = pd.read_csv('../../data/cta_nyc.csv')

    cta[['BoroCT2020']].merge(emb, left_index=True, right_index=True).to_csv('./data/region_embedding.csv', index=False)


if __name__ == '__main__':
    main()