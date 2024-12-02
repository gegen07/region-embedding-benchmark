from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

##
#  alpha=2 para regionDCL, 
#  alpha=0.6 para HGI,
#  alpha=4 para poi-encoder,
#  alpha=0.6 para our model,
#  alpha=0.6 para hex2vec,
#  alpha=100 para geovex,

def train(df):
    y = df['pop'].values
    X = df.drop(columns=['pop', 'GEOID']).values

    # model = XGBRegressor(n_estimators=15, max_depth=3, eta=0.1, subsample=0.7, colsample_bytree=0.8, random_state=42)
    # model = ElasticNet(alpha=1, l1_ratio=0.1, random_state=42)
    model = Lasso(random_state=0)
    # model = CatBoostRegressor(iterations=10, depth=2, learning_rate=0.1, loss_function='RMSE', random_state=42, verbose=False)
    # model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    grid = dict()
    grid['alpha'] = np.arange(0, 100, 0.1)
    # grid['max_depth'] = [2, 3, 5, 7, 10]
    # grid['n_estimators'] = [1, 2, 3, 4, 5, 8, 10, 50]

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

    print(f'MAE: {results.best_score_}')
    print(f'Config: {results.best_params_}')

    print(f'RMSE: {sum(rmse_list)/len(rmse_list)} +- {np.std(np.array(rmse_list))}')
    print(f'MAE: {sum(mae_list)/len(mae_list)} +- {np.std(np.array(mae_list))}')
    print(f'RMSE Train: {sum(rmse_train_list)/len(rmse_train_list)} +- {np.std(np.array(rmse_train_list))}')
    print(f'R2: {sum(r2_list)/len(r2_list)} +- {np.std(np.array(r2_list))}')
    return model

if __name__ == "__main__":
    pop = pd.read_csv('./data/chicago_population.csv')
    # embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/HGI/data/region_embedding-chicago.csv')
    # embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/h3-embeddings/data/ny_geovex_emb.csv')
    embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/model/region-embedding-chicago.csv').drop(columns=['geometry'])
    # embedding = pd.read_csv('/home/gegen07/dev/projects/region-embedding-benchmark/region-embedding/baselines/GAE/notebooks/new-york-GAE.csv')

    pop = pop.merge(embedding, on='GEOID', how='inner')
    print(len(pop))

    train(pop)

### Chicago
## Our Model
# RMSE: 3552.3391656762515 +- 296.1121410445461
# MAE: 2703.0379027113604 +- 236.81838988560423
# RMSE Train: 3337.375223944364 +- 29.085278363074064
# R2: 0.22527993389345236 +- 0.006601061678020316
# alpha = 25.0

## GAE
# RMSE: 3716.2181237036457 +- 354.2546655559166
# MAE: 2865.8228001410753 +- 229.1243151490726
# RMSE Train: 3671.869323700176 +- 39.16646900932974
# R2: 0.06226654666030844 +- 0.003223545933804075

## DGI
# RMSE: 3709.639574073113 +- 395.1358849269455
# MAE: 2867.4874410924594 +- 269.52833408219914
# RMSE Train: 3607.7182091777786 +- 45.086431614656064
# R2: 0.09474667557264205 +- 0.008932659107843495

## Poi-Encoder
# RMSE: 3694.738637823716 +- 413.29218896334635
# MAE: 2861.791724316604 +- 321.7702596639798
# RMSE Train: 3649.7260554242143 +- 44.37948306658049
# R2: 0.07159487073488137 +- 0.004455683751743405
# alpha = 107.1

## HGI
# RMSE: 3568.5186588008946 +- 360.30830771584345
# MAE: 2682.872700718163 +- 240.5273522079294
# RMSE Train: 3408.44122864963 +- 34.7138382638189
# R2: 0.19019890236688036 +- 0.01034553880915207
# alpha = 3.0

## RegionDCL
# RMSE: 3569.4629802060917 +- 528.6506628157115
# MAE: 2717.465482565786 +- 442.82896292117437
# RMSE Train: 3329.1522382834532 +- 64.91377871011754
# R2: 0.22262419889666804 +- 0.009492827971546278
# alpha = 4.0


### New York
## Our Model
# RMSE: 7873.35946470297 +- 468.342404953046
# MAE: 5971.170067320274 +- 379.91887226385694
# RMSE Train: 7645.507905546144 +- 49.88853090201583
# R2: 0.47069953518856006 +- 0.005543798790795622

## GAE
# RMSE: 9158.426732982105 +- 425.49587240274553
# MAE: 7048.121988426038 +- 218.97025426326417
# RMSE Train: 9107.270251800637 +- 48.540284878869386
# R2: 0.24897195621636686 +- 0.004493533792880525

## DGI
# RMSE: 8114.68218716038 +- 444.2357762698842
# MAE: 6078.807487681877 +- 338.8046440288451
# RMSE Train: 7582.60626093632 +- 45.782152515249685
# R2: 0.479365602156988 +- 0.0058365185982465805

## Poi-Encoder
# RMSE: 8315.722553786443 +- 606.4211873747591
# MAE: 6366.655423895002 +- 398.25371258919694
# RMSE Train: 8086.946300996087 +- 66.81247706994954
# R2: 0.40567304106257784 +- 0.0059230862222866

## HGI
# RMSE: 6863.235365697123 +- 406.5929459798659
# MAE: 5096.610800072669 +- 211.37760006146348
# RMSE Train: 6646.114357481216 +- 48.65303788024172
# R2: 95890403893537 +- 0.003047721195684845

## RegionDCL
# RMSE: 7355.894690761235 +- 513.9798873169091
# MAE: 5549.157890195759 +- 406.41527747350983
# RMSE Train: 7169.896861593346 +- 59.132071441258304
# R2: 0.5316145132765661 +- 0.003746365374436174