from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

def train(df):
    y = df['crime_number'].values
    X = df.drop(columns=['crime_number', 'GEOID']).values

    # model = XGBRegressor(n_estimators=100, max_depth=8, eta=0.1, subsample=0.7, colsample_bytree=0.8, random_state=42)
    # model = ElasticNet(alpha=0.2, l1_ratio=0.1, random_state=42)
    model = Lasso(random_state=0, max_iter=10000)
    # model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)


    grid = dict()
    grid['alpha'] = np.arange(0.1, 60, 0.01)
    # grid['max_depth'] = [2, 3, 5, 7, 10]
    # grid['n_estimators'] = [1, 2, 3, 4, 5, 8, 10, 50, 100]

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    model = Lasso(alpha=results.best_params_['alpha'], random_state=0)
    # model = RandomForestRegressor(n_estimators=results.best_params_['n_estimators'], max_depth=results.best_params_['max_depth'], random_state=42)

    rmse_list = []
    mae_list = []
    r2_list = []
    rmse_train_list = []

    for train_index, test_index in cv.split(X):
        # print(train_index, test_index)
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]


        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)


        rmse = root_mean_squared_error(y_test_fold, y_pred)
        mae = mean_absolute_error(y_test_fold, y_pred)

        y_train_pred = model.predict(X_train_fold)
        # y_train_pred = min_max_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).reshape(-1)

        r2 = r2_score(y_train_fold, y_train_pred)
        rmse_train = root_mean_squared_error(y_train_fold, y_train_pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        rmse_train_list.append(rmse_train)
    print(f'Config: {results.best_params_}')

    print(f'RMSE: {sum(rmse_list)/len(rmse_list)} +- {np.std(np.array(rmse_list))}')
    print(f'MAE: {sum(mae_list)/len(mae_list)} +- {np.std(np.array(mae_list))}')
    print(f'RMSE Train: {sum(rmse_train_list)/len(rmse_train_list)} +- {np.std(np.array(rmse_train_list))}')
    print(f'R2: {sum(r2_list)/len(r2_list)} +- {np.std(np.array(r2_list))}')
    return model

if __name__ == "__main__":
    crime = pd.read_csv('./data/crime-number-central-area.csv')
    # crime = pd.read_csv('./data/crime_number-manhattan.csv')
    # embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/HGI/data/region_embedding.csv')
    # embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/h3-embeddings/data/ny_hex2vec_emb.csv')
    embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/region-embedding-chicago.csv').drop(columns=['geometry'])
    # embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/GAE/notebooks/new-york-GAE.csv')

    crime = crime.merge(embedding, on='GEOID', how='inner')

    train(crime)

### Chicago
## Our Model
# Config: {'alpha': 58.61}
# RMSE: 6476.80113187814 +- 1539.3063599114323
# MAE: 4481.519489536409 +- 363.84183354174286
# RMSE Train: 6327.535760738328 +- 192.1720668722736
# R2: 0.3130418042725436 +- 0.009479448993419798

## GAE
# RMSE: 7241.672893828564 +- 1481.7350563494888
# MAE: 5153.213603753288 +- 414.5273191113673
# RMSE Train: 7234.264224258167 +- 200.93499434977093
# R2: 0.10195811893290041 +- 0.00824518964384521

## DGI
# RMSE: 7000.492839596889 +- 1474.9420893917957
# MAE: 4935.264853064052 +- 411.45690390453194
# RMSE Train: 6823.888612795963 +- 185.65022781712116
# R2: 0.200878607460539 +- 0.01157923429612731

## Poi-Encoder
# Config: {'alpha': 17.11}
# RMSE: 6865.583984205524 +- 1422.874620566402
# MAE: 4853.1947189200855 +- 547.6344175078374
# RMSE Train: 6669.333938299504 +- 182.2647831484082
# R2: 0.23646554739470652 +- 0.008050312429644792

## HGI
# Config: {'alpha': 1.43}
# RMSE: 5989.646580009793 +- 1102.9497831445804
# MAE: 4120.603123557344 +- 437.6031655237989
# RMSE Train: 5634.537832424791 +- 127.09814246956061
# R2: 0.4548263884438038 +- 0.00975497415510678

## RegionDCL
# Config: {'alpha': 3.67}
# RMSE: 6827.627994633136 +- 1433.9910842251672
# MAE: 4767.282772351401 +- 364.36075446926105
# RMSE Train: 6566.036159998573 +- 186.33198748166996
# R2: 0.25956994703317904 +- 0.01007744943122576


### New York
## Our Model
# RMSE: 123.73104514038019 +- 31.79306610366947
# MAE: 87.00835083375021 +- 12.235879750514167
# RMSE Train: 121.46215958045556 +- 3.5525799994189846
# R2: 0.2668257055781149 +- 0.012140932285128738

# RMSE: 821.7512896904797 +- 198.6729339175574
# MAE: 590.0446693799563 +- 64.37725678681669
# RMSE Train: 784.3156997232279 +- 23.980497902299483
# R2: 0.35579315247451493 +- 0.014447861967557823

# RMSE: 694.2418243951284 +- 151.16040535903537
# MAE: 486.97684099922736 +- 50.98528467790095
# RMSE Train: 673.0375005813339 +- 16.23911524939741
# R2: 0.36859394029757553 +- 0.01233431215440227

## GAE
# RMSE: 129.8377566303254 +- 33.09795871991274
# MAE: 93.73036995591431 +- 16.184608422243834
# RMSE Train: 128.9665736949699 +- 3.552213602969412
# R2: 0.17336065594483124 +- 0.012402351545501985

# RMSE: 877.7882600867639 +- 248.972179947692
# MAE: 636.3788086745597 +- 99.55134710951887
# RMSE Train: 875.0224120626266 +- 33.37860083842005
# R2: 0.1986499987282178 +- 0.016431384009335068

# RMSE: 819.9551814801005 +- 159.44232900391023
# MAE: 584.3496104166813 +- 50.46273487874771
# RMSE Train: 826.829775559702 +- 18.210113863769326
# R2: 0.04714041916650548 +- 0.005274390974717902

## DGI
# RMSE: 130.8894364646805 +- 29.698774006104397
# MAE: 94.6507313252225 +- 11.591208731018082
# RMSE Train: 125.88762225911611 +- 3.3262295149062533
# R2: 0.21223680750689028 +- 0.016365484053280713

# RMSE: 910.8173601322158 +- 205.4677937812537
# MAE: 662.2412194814834 +- 84.4087412779307
# RMSE Train: 890.0386857102594 +- 27.007488285008893
# R2: 0.1705424478586943 +- 0.011183656108625713

# RMSE: 767.9898211477096 +- 128.17427579185807
# MAE: 532.7021523078254 +- 49.192617462434704
# RMSE Train: 720.9986714523009 +- 15.19213912780522
# R2: 0.2753492634402688 +- 0.011257665735607162

## Poi-Encoder
# RMSE: 118.81846176503666 +- 31.19725517306184
# MAE: 87.58718266708209 +- 13.84709376018578
# RMSE Train: 117.68298224584764 +- 3.4937857433837363
# R2: 0.3106151162047963 +- 0.01617555528301019

# RMSE: 807.4074254175861 +- 251.77790252920013
# MAE: 607.2484936924338 +- 145.09969403280394
# RMSE Train: 793.7492619891052 +- 31.249797487725548
# R2: 0.33901246276128355 +- 0.014489473550034914

# RMSE: 720.728563382582 +- 156.61697482747005
# MAE: 505.18664509324145 +- 46.02519889687244
# RMSE Train: 683.7331510778947 +- 18.951145061764596
# R2: 0.3402488112194182 +- 0.009634385254169731

## HGI
# RMSE: 116.99795420207154 +- 28.16175648433319
# MAE: 84.3893852351394 +- 13.192466721170458
# RMSE Train: 112.86226108013747 +- 3.189983595045776
# R2: 0.3657556684863471 +- 0.019258220742257

# RMSE: 772.6789308365082 +- 248.21446884748923
# MAE: 583.3170623231396 +- 146.95880039833668
# RMSE Train: 770.4402871947065 +- 31.37696421857771
# R2: 0.37746287312264887 +- 0.00553135556209407

# RMSE: 641.1836837273936 +- 137.9292282241742
# MAE: 448.13763590398696 +- 45.21252776841338
# RMSE Train: 588.6574034890194 +- 14.766462684442269
# R2: 0.5108458731734578 +- 0.010186922909923103

## RegionDCL
# RMSE: 119.92514261595029 +- 30.57982541939391
# MAE: 86.85649153695324 +- 13.563942131502163
# RMSE Train: 111.98836003747053 +- 3.5523417081891524
# R2: 0.3758564713869547 +- 0.011732913987097462

# RMSE: 818.6980931364773 +- 236.19818434895708
# MAE: 584.425670943204 +- 116.92859385531192
# RMSE Train: 771.486214138228 +- 27.444741509884125
# R2: 0.37522739818864037 +- 0.018895610101950903

# RMSE: 714.7481893660774 +- 180.19039198637194
# MAE: 494.7120397968289 +- 44.41131763685737
# RMSE Train: 697.9637013599174 +- 25.53319435870106
# R2: 0.3113560298362823 +- 0.01234480349708382
