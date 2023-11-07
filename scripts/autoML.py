import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#Importing data
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

# Labeling data
train_a['building_id'] = 'a'
train_b['building_id'] = 'b'
train_c['building_id'] = 'c'

X_train_estimated_a['building_id'] = 'a'
X_train_estimated_b['building_id'] = 'b'
X_train_estimated_c['building_id'] = 'c'

X_train_observed_a['building_id'] = 'a'
X_train_observed_b['building_id'] = 'b'
X_train_observed_c['building_id'] = 'c'

X_test_estimated_a['building_id'] = 'a'
X_test_estimated_b['building_id'] = 'b'
X_test_estimated_c['building_id'] = 'c'

# Combining data
all_observed_X = pd.concat([X_train_observed_a,X_train_observed_b,X_train_observed_c])
all_estimated_X = pd.concat([X_train_estimated_a,X_train_estimated_b,X_train_estimated_c])
all_y = pd.concat([train_a, train_b, train_c])
X_test = pd.concat([X_test_estimated_a,X_test_estimated_b,X_test_estimated_c])

# add type of weather data
all_observed_X['isObserved'] = 1
all_estimated_X['isObserved'] = 0
X_test['isObserved'] = 0

# Combining training data
all_X = pd.concat([all_estimated_X,all_observed_X])

# Aggregating all_X
all_X_aggregated = all_X.copy()
all_X_aggregated['time_hour'] = all_X_aggregated['date_forecast'].dt.floor('H')
all_X_aggregated = all_X_aggregated.groupby(['building_id','time_hour','isObserved']).mean().reset_index()

# Aggregating X_test
X_test_aggregated = X_test.copy()
X_test_aggregated['time_hour'] = X_test_aggregated['date_forecast'].dt.floor('H')
X_test_aggregated = X_test_aggregated.groupby(['building_id','time_hour','isObserved']).mean().reset_index()

# Renaming time column in all_y
all_y = all_y.rename(columns={'time': 'time_hour'})

# Merging all_X_aggregated and all_y
all_train = pd.merge(all_X_aggregated,all_y,on=['building_id','time_hour'],how='right') # right join to keep all y values

#Encoding date
all_X_aggregated['sin_mon'] = np.sin((all_X_aggregated['time_hour'].dt.month - 1)*np.pi/11)
all_X_aggregated['cos_mon'] = np.cos((all_X_aggregated['time_hour'].dt.month - 1)*np.pi/11)

all_X_aggregated['sin_hr']= np.sin(all_X_aggregated['time_hour'].dt.hour*np.pi/23)
all_X_aggregated['cos_hr']= np.sin(all_X_aggregated['time_hour'].dt.hour*np.pi/23)

X_test_aggregated['sin_mon'] = np.sin((X_test_aggregated['time_hour'].dt.month - 1)*np.pi/11)
X_test_aggregated['cos_mon'] = np.cos((X_test_aggregated['time_hour'].dt.month - 1)*np.pi/11)

X_test_aggregated['sin_hr']= np.sin(X_test_aggregated['time_hour'].dt.hour*np.pi/23)
X_test_aggregated['cos_hr']= np.sin(X_test_aggregated['time_hour'].dt.hour*np.pi/23)

# define groups of variables

sun_features_list = ['clear_sky_energy_1h:J', 'clear_sky_rad:W', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'direct_rad:W', 'direct_rad_1h:J', 'is_day:idx', 'is_in_shadow:idx', 'sun_elevation:d']

humidity_features_list = ['absolute_humidity_2m:gm3', 'air_density_2m:kgm3', 'dew_point_2m:K', 't_1000hPa:K']

snow_features_list = ['fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm', 'fresh_snow_3h:cm', 'fresh_snow_6h:cm']

cloud_height_features_list = ['ceiling_height_agl:m', 'cloud_base_agl:m']

feature_groups = [
    {
        'name': 'sun',
        'features': sun_features_list
    },
    {
        'name': 'humidity',
        'features': humidity_features_list
    },
    {
        'name': 'snow',
        'features': snow_features_list
    },
    {
        'name': 'cloud_height',
        'features': cloud_height_features_list
    }
]

all_pca_features = sun_features_list + humidity_features_list + snow_features_list + cloud_height_features_list

all_data = all_X_aggregated.merge(all_y,on=['time_hour','building_id'],how='left')

# drop non useful features
drop_features = [
    'snow_density:kgm3', # always 250 or nan
    'date_calc', # TODO put dates back in, only for testing
    'date_forecast', # TODO
    'time_hour', # TODO
    'elevation:m'
]

all_data = all_data.drop(columns=drop_features)

#fill as mean, (iterative imputer)
all_data['cloud_base_agl:m'].fillna(all_data['cloud_base_agl:m'].mean(), inplace=True)
all_data['ceiling_height_agl:m'].fillna(all_data['ceiling_height_agl:m'].mean(), inplace=True)

#same for test dataset
X_test_aggregated = X_test_aggregated.drop(['date_forecast','date_calc','snow_density:kgm3','time_hour','elevation:m'],axis=1) # TODO remove columns at better cell
X_test_aggregated['cloud_base_agl:m'].fillna(X_test_aggregated['cloud_base_agl:m'].mean(), inplace=True)
X_test_aggregated['ceiling_height_agl:m'].fillna(X_test_aggregated['ceiling_height_agl:m'].mean(), inplace=True)

# remove rows that are NaN in target column
all_data = all_data[~all_data['pv_measurement'].isna()]

# shuffle all_data to have approximately the same distribution of buildings and observed/estimated in each fold of CV
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True) # TODO turn back on?

X_test = X_test_aggregated

# define X, y and X_test
X = all_data.drop(['pv_measurement'], axis=1)
y = all_data['pv_measurement']

sun_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])
humidity_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca' , PCA(n_components=2))
])
snow_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=1))
])
cloud_height_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=1))
])

# set column transformer
columnTransformer = ColumnTransformer(
    transformers = [
        ('pca_sun', sun_pipeline, sun_features_list),
        ('pca_humidity', humidity_pipeline, humidity_features_list),
        ('pca_snow', snow_pipeline, snow_features_list),
        ('pca_cloud_height', cloud_height_pipeline, cloud_height_features_list),
        ('oneHotEncoder', OneHotEncoder(handle_unknown='ignore'), ['building_id']),
    ],
    remainder='passthrough',  # Include other columns
)

# build the pipeline
pipelinePrep = Pipeline([
    ('columnTransformer', columnTransformer),
])

prep_model = pipelinePrep.fit(X)

X = pipelinePrep.fit_transform(X)
X_test = pipelinePrep.fit_transform(X_test)



import h2o
from h2o.automl import H2OAutoML

# Start the H2O cluster (locally)
h2o.init()

train = np.c_[X, y]
train = h2o.H2OFrame(pd.DataFrame(train))

x = train.columns
y = "37"
x.remove(y)

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# model_id                                                  auc    logloss    mean_per_class_error      rmse       mse
# ---------------------------------------------------  --------  ---------  ----------------------  --------  --------
# StackedEnsemble_AllModels_AutoML_20181212_105540     0.789801   0.551109                0.333174  0.43211   0.186719
# StackedEnsemble_BestOfFamily_AutoML_20181212_105540  0.788425   0.552145                0.323192  0.432625  0.187165
# XGBoost_1_AutoML_20181212_105540                     0.784651   0.55753                 0.325471  0.434949  0.189181
# XGBoost_grid_1_AutoML_20181212_105540_model_4        0.783523   0.557854                0.318819  0.435249  0.189441
# XGBoost_grid_1_AutoML_20181212_105540_model_3        0.783004   0.559613                0.325081  0.435708  0.189841
# XGBoost_2_AutoML_20181212_105540                     0.78136    0.55888                 0.347074  0.435907  0.190015
# XGBoost_3_AutoML_20181212_105540                     0.780847   0.559589                0.330739  0.43613   0.190209
# GBM_5_AutoML_20181212_105540                         0.780837   0.559903                0.340848  0.436191  0.190263
# GBM_2_AutoML_20181212_105540                         0.780036   0.559806                0.339926  0.436415  0.190458
# GBM_1_AutoML_20181212_105540                         0.779827   0.560857                0.335096  0.436616  0.190633
# GBM_3_AutoML_20181212_105540                         0.778669   0.56179                 0.325538  0.437189  0.191134
# XGBoost_grid_1_AutoML_20181212_105540_model_2        0.774411   0.575017                0.322811  0.4427    0.195984
# GBM_4_AutoML_20181212_105540                         0.771426   0.569712                0.33742   0.44107   0.194543
# GBM_grid_1_AutoML_20181212_105540_model_1            0.769752   0.572583                0.344331  0.442452  0.195764
# GBM_grid_1_AutoML_20181212_105540_model_2            0.754366   0.918567                0.355855  0.496638  0.246649
# DRF_1_AutoML_20181212_105540                         0.742892   0.595883                0.355403  0.452774  0.205004
# XRT_1_AutoML_20181212_105540                         0.742091   0.599346                0.356583  0.453117  0.205315
# DeepLearning_grid_1_AutoML_20181212_105540_model_2   0.741795   0.601497                0.368291  0.454904  0.206937
# XGBoost_grid_1_AutoML_20181212_105540_model_1        0.693554   0.620702                0.40588   0.465791  0.216961
# DeepLearning_1_AutoML_20181212_105540                0.69137    0.637954                0.409351  0.47178   0.222576
# DeepLearning_grid_1_AutoML_20181212_105540_model_1   0.690084   0.661794                0.418469  0.476635  0.227181
# GLM_grid_1_AutoML_20181212_105540_model_1            0.682648   0.63852                 0.397234  0.472683  0.223429
#
# [22 rows x 6 columns]

# The leader model is stored here
aml.leader

with open('automl.txt', 'w') as f:
    f.write(str(lb))
    f.write("\nBEST: "+str(aml.leader)+"\n")

sample_submission = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv')
test['prediction'] = aml.leader.predict(test)
sample_submission = sample_submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
sample_submission.to_csv('AUTO.csv', index=False)

nativeXGBoostParam = aml.leader.convert_H2OXGBoostParams_2_XGBoostParams()
#nativeXGBoostInput = data.convert_H2OFrame_2_DMatrix(myX, y, h2oModelD)