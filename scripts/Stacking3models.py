import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor,ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV, RFE
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skopt.space import Real, Categorical, Integer
from lightgbm import LGBMRegressor

from skopt import BayesSearchCV

train_a = pd.read_parquet('A/train_targets.parquet')
train_b = pd.read_parquet('B/train_targets.parquet')
train_c = pd.read_parquet('C/train_targets.parquet')

X_train_estimated_a = pd.read_parquet('A/X_train_estimated.parquet')
X_train_estimated_b = pd.read_parquet('B/X_train_estimated.parquet')
X_train_estimated_c = pd.read_parquet('C/X_train_estimated.parquet')

X_train_observed_a = pd.read_parquet('A/X_train_observed.parquet')
X_train_observed_b = pd.read_parquet('B/X_train_observed.parquet')
X_train_observed_c = pd.read_parquet('C/X_train_observed.parquet')

X_test_estimated_a = pd.read_parquet('A/X_test_estimated.parquet')
X_test_estimated_b = pd.read_parquet('B/X_test_estimated.parquet')
X_test_estimated_c = pd.read_parquet('C/X_test_estimated.parquet')

X_test_estimated_a['time'] = X_test_estimated_a['date_forecast'].dt.floor('H')
X_test_estimated_a = X_test_estimated_a.groupby(['time']).mean().reset_index()

X_test_estimated_b['time'] = X_test_estimated_b['date_forecast'].dt.floor('H')
X_test_estimated_b = X_test_estimated_b.groupby(['time']).mean().reset_index()

X_test_estimated_c['time'] = X_test_estimated_c['date_forecast'].dt.floor('H')
X_test_estimated_c = X_test_estimated_c.groupby(['time']).mean().reset_index()

#X_test['delta_forecast'] = (X_test['time']-X_test['date_calc']).apply(lambda x: x.total_seconds() / 3600)

#Xt2 = pd.DataFrame(X_test['delta_forecast'])

X_test_estimated_a.drop(['date_calc'], axis=1, inplace=True)
X_test_estimated_b.drop(['date_calc'], axis=1, inplace=True)
X_test_estimated_c.drop(['date_calc'], axis=1, inplace=True)

#X_observed = pd.concat([X_train_observed_a, X_train_observed_b, X_train_observed_c])
X_train_observed_a['time'] = X_train_observed_a['date_forecast'].dt.floor('H')
X_train_observed_a = X_train_observed_a.groupby(['time']).mean().reset_index()

X_train_observed_b['time'] = X_train_observed_b['date_forecast'].dt.floor('H')
X_train_observed_b = X_train_observed_b.groupby(['time']).mean().reset_index()

X_train_observed_c['time'] = X_train_observed_c['date_forecast'].dt.floor('H')
X_train_observed_c = X_train_observed_c.groupby(['time']).mean().reset_index()

#y_train = pd.concat([train_a, train_b, train_c])


# combine and remove rows with missing values in y
Xy_obs_a = pd.merge(X_train_observed_a, train_a, on=['time'], how='inner')
Xy_obs_a = Xy_obs_a[Xy_obs_a['pv_measurement'].notna()]

Xy_obs_b = pd.merge(X_train_observed_b, train_b, on=['time'], how='inner')
Xy_obs_b = Xy_obs_b[Xy_obs_b['pv_measurement'].notna()]

Xy_obs_c = pd.merge(X_train_observed_c, train_c, on=['time'], how='inner')
Xy_obs_c = Xy_obs_c[Xy_obs_c['pv_measurement'].notna()]

Xa = Xy_obs_a.drop(['pv_measurement'], axis=1)
ya = Xy_obs_a['pv_measurement']

Xb = Xy_obs_b.drop(['pv_measurement'], axis=1)
yb = Xy_obs_b['pv_measurement']

Xc = Xy_obs_c.drop(['pv_measurement'], axis=1)
yc = Xy_obs_c['pv_measurement']

drop_cols = ['time', 'date_forecast', 'snow_density:kgm3']
Xa = Xa.drop(drop_cols, axis=1)
Xb = Xb.drop(drop_cols, axis=1)
Xc = Xc.drop(drop_cols, axis=1)

# get y mean per building id
#mean_y_per_building = y.groupby(Xy_observed['building_id']).mean()

# divide y by mean per building id
#y = y.groupby(Xy_observed['building_id']).transform(lambda x: x / mean_y_per_building[x.name]) 

# setting types of columns
categorical_features = [
   # 'building_id'
]

impute_features = [
    "cloud_base_agl:m",
    "ceiling_height_agl:m",
]


#XGBOOOST-----------------------------------------------------------------------------------------------

parameters = {
    'estimator__n_estimators': Integer(10,500),
    'estimator__max_depth': Integer(3, 10),
    'estimator__learning_rate': Real(0.01, 0.5),
    'estimator__subsample': Real(0.5, 1.0),
    'estimator__colsample_bytree': Real(0.5, 1.0),
    'estimator__gamma': Real(0, 5),
    'estimator__reg_alpha': Real(0, 5),
    'estimator__reg_lambda': Real(0, 5),
}

# set column transformer
columnTransformer = ColumnTransformer(
    transformers=[
        ('imputer', SimpleImputer(strategy='mean'),impute_features),
        ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough',  # Include other columns
    n_jobs=-1
)

# build the pipeline
pipeline = Pipeline(steps=[
    ('columnTransformer', columnTransformer),
    ('estimator', xgboost.XGBRegressor())
])

# create bayesian search estimator
bayes_search_estimator = BayesSearchCV(
    pipeline, parameters, scoring='neg_mean_absolute_error', cv=3, error_score='raise', n_jobs=-1, verbose=10, n_iter=100, random_state=42)

# fit the estimator on the data
bayes_search_estimator.fit(Xa, ya)

with open('xgbA.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xb, yb)

with open('xgbB.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xc, yc)

with open('xgbC.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")


#LGBoost-----------------------------------------------------------------------------------------------

parameters = {
    'estimator__num_leaves': Integer(20,100),
    'estimator__n_estimators': Integer(100,1000),
    'estimator__max_depth': Integer(3, 20),
    'estimator__learning_rate': Real(0.01, 0.3),
    'estimator__min_child_samples': Integer(20,100),
    'estimator__min_child_weight': Integer(1,10),
    'estimator__subsample': Real(0.5, 1.0),
    'estimator__colsample_bytree': Real(0.5, 1.0),
    'estimator__reg_alpha': Real(0, 1),
    'estimator__reg_lambda': Real(0, 1),
    'estimator_objective' : ['regression']
}

# set column transformer
columnTransformer = ColumnTransformer(
    transformers=[
        ('imputer', SimpleImputer(strategy='mean'),impute_features),
        ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough',  # Include other columns
    n_jobs=-1
)

# build the pipeline
pipeline = Pipeline(steps=[
    ('columnTransformer', columnTransformer),
    ('estimator', LGBMRegressor())
])

# create bayesian search estimator
bayes_search_estimator = BayesSearchCV(
    pipeline, parameters, scoring='neg_mean_absolute_error', cv=3, error_score='raise', n_jobs=-1, verbose=10, n_iter=100, random_state=42)

# fit the estimator on the data
bayes_search_estimator.fit(Xa, ya)

with open('lgbA.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xb, yb)

with open('lgbB.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xc, yc)

with open('lgbC.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

#HGBoost-----------------------------------------------------------------------------------------------
parameters = {
    'estimator__n_estimators': Integer(100,1000),
    'estimator__max_depth': Integer(3, 10),
    'estimator__learning_rate': Real(0.01, 0.3),
    'estimator__min_child_weight': Integer(1,10),
    'estimator__subsample': Real(0.5, 1.0),
    'estimator__colsample_bytree': Real(0.5, 1.0),
    'estimator__alpha': Real(0, 1),
    'estimator__lambda': Real(0, 1),
    'estimator__gamma': Real(0, 1),
}

# set column transformer
columnTransformer = ColumnTransformer(
    transformers=[
        ('imputer', SimpleImputer(strategy='mean'),impute_features),
        ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough',  # Include other columns
    n_jobs=-1
)

# build the pipeline
pipeline = Pipeline(steps=[
    ('columnTransformer', columnTransformer),
    ('estimator', HistGradientBoostingRegressor())
])

# create bayesian search estimator
bayes_search_estimator = BayesSearchCV(
    pipeline, parameters, scoring='neg_mean_absolute_error', cv=3, error_score='raise', n_jobs=-1, verbose=10, n_iter=100, random_state=42)

# fit the estimator on the data
bayes_search_estimator.fit(Xa, ya)

with open('hgbA.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xb, yb)

with open('hgbB.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xc, yc)

with open('hgbC.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

#XT-----------------------------------------------------------------------------------------------
parameters = {
    'estimator__n_estimators': Integer(100,1000),
    'estimator__max_depth': Integer(3, 10),
    'estimator__min_samples_split': Integer(2, 20),
}

# set column transformer
columnTransformer = ColumnTransformer(
    transformers=[
        ('imputer', SimpleImputer(strategy='mean'),impute_features),
        ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough',  # Include other columns
    n_jobs=-1
)

# build the pipeline
pipeline = Pipeline(steps=[
    ('columnTransformer', columnTransformer),
    ('estimator', ExtraTreesRegressor())
])

# create bayesian search estimator
bayes_search_estimator = BayesSearchCV(
    pipeline, parameters, scoring='neg_mean_absolute_error', cv=3, error_score='raise', n_jobs=-1, verbose=10, n_iter=100, random_state=42)

# fit the estimator on the data
bayes_search_estimator.fit(Xa, ya)

with open('xtA.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xb, yb)

with open('xtB.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")

bayes_search_estimator.fit(Xc, yc)

with open('xtC.txt', 'w') as f:
    f.write("best params: "+str(bayes_search_estimator.best_params_)+"\n")
    f.write("best score "+str(bayes_search_estimator.best_score_)+"\n")


