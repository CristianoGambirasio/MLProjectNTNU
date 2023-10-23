import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Integer
#!pip install xgboost
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

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
all_X = pd.concat([all_observed_X])#,all_estimated_X])

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
      # TODO put dates back in, only for testing #TODO readd date_calc for estimated
    'date_forecast', # TODO
    'time_hour', # TODO
    'elevation:m'
]

all_data = all_data.drop(columns=drop_features)

#fill as mean, (iterative imputer)
all_data['cloud_base_agl:m'].fillna(all_data['cloud_base_agl:m'].mean(), inplace=True)
all_data['ceiling_height_agl:m'].fillna(all_data['ceiling_height_agl:m'].mean(), inplace=True)

#same for test dataset
# add 'data_calc' for estimated
X_test_aggregated = X_test_aggregated.drop(['date_forecast','snow_density:kgm3','date_calc','time_hour','elevation:m'],axis=1) # TODO remove columns at better cell
X_test_aggregated['cloud_base_agl:m'].fillna(X_test_aggregated['cloud_base_agl:m'].mean(), inplace=True)
X_test_aggregated['ceiling_height_agl:m'].fillna(X_test_aggregated['ceiling_height_agl:m'].mean(), inplace=True)

# remove rows that are NaN in target column
all_data = all_data[~all_data['pv_measurement'].isna()]

# shuffle all_data to have approximately the same distribution of buildings and observed/estimated in each fold of CV
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True) # TODO turn back on?

X_test = X_test_aggregated

# remove rows that are NaN in target column
all_data = all_data[~all_data['pv_measurement'].isna()]

# shuffle all_data to have approximately the same distribution of buildings and observed/estimated in each fold of CV
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True) # TODO turn back on?

X_test = X_test_aggregated

# PCA
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

#Random Forest tuning
parameters = {'n_estimators': [200],
            'max_depth': [20],
            'eta': [0.1],
            'min_child_weight': [40],
                }
xgb = xgboost.XGBRegressor()
#cross = GridSearchCV(xgb, parameters, verbose=10, cv=4, n_jobs=-1).fit(X,y)

xgb1 = xgboost.XGBRegressor(n_estimators = 200, max_depth=20, eta=0.1, min_child_weight=40)

model1 = xgb1.fit(X,y)


#Estimated X prep-----------------------------------------------------------------------------------
estimated_X_agg = all_estimated_X.copy()
estimated_X_agg['time_hour'] = estimated_X_agg['date_forecast'].dt.floor('H')
estimated_X_agg = estimated_X_agg.groupby(['building_id','time_hour','isObserved']).mean().reset_index()

all_estimated = pd.merge(estimated_X_agg,all_y,on=['building_id','time_hour'],how='left')

#Encoding date
all_estimated['sin_mon'] = np.sin((all_estimated['time_hour'].dt.month - 1)*np.pi/11)
all_estimated['cos_mon'] = np.cos((all_estimated['time_hour'].dt.month - 1)*np.pi/11)

all_estimated['sin_hr']= np.sin(all_estimated['time_hour'].dt.hour*np.pi/23)
all_estimated['cos_hr']= np.sin(all_estimated['time_hour'].dt.hour*np.pi/23)

all_estimated['sin_mon'] = np.sin((all_estimated['time_hour'].dt.month - 1)*np.pi/11)
all_estimated['cos_mon'] = np.cos((all_estimated['time_hour'].dt.month - 1)*np.pi/11)

all_estimated['sin_hr']= np.sin(all_estimated['time_hour'].dt.hour*np.pi/23)
all_estimated['cos_hr']= np.sin(all_estimated['time_hour'].dt.hour*np.pi/23)

drop_features = [
    'snow_density:kgm3', # always 250 or nan
    'date_calc', # TODO put dates back in, only for testing
    'date_forecast', # TODO
    'time_hour', # TODO
    'elevation:m'
]

all_estimated = all_estimated.drop(columns=drop_features)
all_estimated['cloud_base_agl:m'].fillna(all_estimated['cloud_base_agl:m'].mean(), inplace=True)
all_estimated['ceiling_height_agl:m'].fillna(all_estimated['ceiling_height_agl:m'].mean(), inplace=True)

all_estimated = all_estimated[~all_estimated['pv_measurement'].isna()]

X_estimated = all_estimated.drop(['pv_measurement'], axis=1)
y_estimated = all_estimated['pv_measurement']

X_estimated = pipelinePrep.fit_transform(X_estimated)

#Second model-------------------------------------------------------------------------------------------
y1_estimated = model1.predict(X_estimated)

X_estimated =np.c_[X_estimated, y1_estimated]

parameters = {'n_estimators': [100,200,300],
            'max_depth': [10,20,30],
            'eta': [0.15,0.1,0.2],
            'min_child_weight': [30,40,50],
                }
xgb = xgboost.XGBRegressor()
cross = GridSearchCV(xgb, parameters, verbose=10, cv=4, n_jobs=-1,scoring="neg_mean_absolute_error").fit(X_estimated,y_estimated)

with open('double.txt', 'w') as f:
    f.write("\nMODEL "+str(cross.best_estimator_)+"\n")
    f.write("best params: "+str(cross.best_params_)+"\n")
    f.write("best score "+str(cross.best_score_)+"\n")

print(cross.cv_results_['mean_test_score'])

print("linear")
lr = LinearRegression()

linearScore = cross_val_score(LinearRegression(), X_estimated, y_estimated, cv = 4,scoring="neg_mean_absolute_error")
print(linearScore)
linearScore
