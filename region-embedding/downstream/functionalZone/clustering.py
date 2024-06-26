from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch

def cluster(ground_truth_df, model_df, ground_truth, ground_truth_h3, embeddings, names, boroughs):
    """
    Clustering embeddings using KMeans
    """

    arr = []

    for index, embedding in enumerate(embeddings):
        pca = PCA(n_components=11)
        embedding = pca.fit_transform(embedding)
        embeddings[index] = embedding
        
    for k in [11]:
        kmeans_normal = KMeans(n_clusters=k, random_state=0)
        # kmeans_normal.fit(ground_truth)
        labels_normal = ground_truth
        # ground_truth_df['label'] = labels_normal
        labels_normal = list(zip(boroughs[0], labels_normal))


        kmeans_h3 = KMeans(n_clusters=k, random_state=0)
        # kmeans_h3.fit(ground_truth_h3)
        labels_h3 = ground_truth_h3
        labels_h3 = list(zip(boroughs[1], labels_h3))

        for model in range(len(names)):            
            name = names[model]
            if name in ['geovex', 'hex2vec']:
                labels = labels_h3
                labels_pred = kmeans_h3.fit_predict(embeddings[model])
            else:
                labels = labels_normal
                labels_pred = kmeans_normal.fit_predict(embeddings[model])

            # if name == 'model':
            #     model_df['label'] = labels_pred
            #     ground_truth_df[['BoroCT2020', 'label']].to_csv('ground_truth.csv', index=False)
            #     model_df[['BoroCT2020', 'label']].to_csv('model.csv', index=False)
            #     return

            labels_observed= []
            for i in range(len(labels)):
                if labels[i][0] in boroughs[model+2]:
                    labels_observed.append(labels[i][1])

            arr.append((k, adjusted_rand_score(labels_observed, labels_pred), normalized_mutual_info_score(labels_observed, labels_pred), f1_score(labels_observed, labels_pred, average='macro'), name))

    pd.DataFrame(arr, columns=['k', 'ARI', 'NMI', 'f1-score', 'model']).to_csv('clustering.csv', index=False)


if __name__ == '__main__':
    boroughs = []

    ground_truth = pd.read_csv('./data/nyc-landuse.csv')
    boroughs.append(ground_truth['BoroCT2020'].values)
    ground_truth = ground_truth.iloc[:, 1:].values
    ground_truth = torch.argmax(torch.from_numpy(ground_truth), dim=1).numpy()

    ground_truth_h3 = pd.read_csv('./data/nyc-landuse-res-9.csv')
    boroughs.append(ground_truth_h3['region_id'].values)
    ground_truth_h3 = ground_truth_h3.iloc[:, 1:].values
    ground_truth_h3 = torch.argmax(torch.from_numpy(ground_truth_h3), dim=1).numpy()
    
    geovex = pd.read_csv('../../baselines/h3-embeddings/data/ny_geovex_emb.csv')
    geovex = geovex[geovex['region_id'].isin(boroughs[1])].reset_index(drop=True)
    boroughs.append(geovex['region_id'].values)
    geovex = geovex.iloc[:, 1:].values

    
    hex2vec = pd.read_csv('../../baselines/h3-embeddings/data/ny_hex2vec_emb.csv')
    hex2vec = hex2vec[hex2vec['region_id'].isin(boroughs[1])].reset_index(drop=True)
    boroughs.append(hex2vec['region_id'].values)
    hex2vec = hex2vec.iloc[:, 1:].values

    hgi = pd.read_csv('../../baselines/HGI/data/region_embedding.csv')
    hgi = hgi[hgi['BoroCT2020'].isin(boroughs[0])].reset_index(drop=True)
    boroughs.append(hgi['BoroCT2020'].values)
    hgi = hgi.iloc[:, 0:64].values

    poi_encoder = pd.read_csv('../../baselines/poi-encoder/data/region_embedding.csv')
    poi_encoder = poi_encoder[poi_encoder['BoroCT2020'].isin(boroughs[0])].reset_index(drop=True)
    boroughs.append(poi_encoder['BoroCT2020'].values)
    poi_encoder = poi_encoder.iloc[:, 1:].values

    regionDCL = pd.read_csv('/home/gegen07/dev/projects/region-embedding-exploration/ny/data/emb_20_regionDCL_clustered.csv')
    regionDCL = regionDCL[regionDCL['BoroCT2020'].isin(boroughs[0])].reset_index(drop=True)
    boroughs.append(regionDCL['BoroCT2020'].values)
    regionDCL = regionDCL.iloc[:, 2:66].values

    model = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/data/region_embedding-info-nce.csv')
    model = model[model['BoroCT2020'].isin(boroughs[0])].reset_index(drop=True)
    boroughs.append(model['BoroCT2020'].values)
    model = model.iloc[:, 0:64].values

    embeddings = [geovex, hex2vec, hgi, poi_encoder, regionDCL, model]
    names = ['geovex', 'hex2vec', 'hgi', 'poi-encoder', 'regionDCL', 'model']

    hgi = pd.read_csv('../../baselines/HGI/data/region_embedding.csv')
    hgi = hgi[hgi['BoroCT2020'].isin(boroughs[0])].reset_index(drop=True)

    model = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/data/region_embedding-info-nce.csv')
    model = model[model['BoroCT2020'].isin(boroughs[0])].reset_index(drop=True)

    # pd.read_csv('./data/nyc-landuse.csv'), hgi, 
    cluster(pd.read_csv('./data/nyc-landuse.csv'), model, ground_truth, ground_truth_h3, embeddings, names, boroughs)
    