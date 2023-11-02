import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV, RFE
import xgboost

from skopt import BayesSearchCV
from skopt.plots import plot_objective
from skopt.space import Real, Categorical, Integer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import PredefinedSplit


import optuna

import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from helpers import *

from sklearn.model_selection import cross_val_score







from sklearn.base import BaseEstimator, TransformerMixin
class Y_Scaler_MaxAbs_per_building:
    def __init__(self):
        self.max_per_building = {}

    def fit(self, y, cat):
        assert type(y) == pd.Series
        assert type(cat) == pd.Series

        y_bid = pd.concat([y, cat], axis=1)
        y_bid.columns = ['y', 'cat']

        self.max_per_building = y_bid.groupby('cat')['y'].max().to_dict()
        return self

    def transform(self, y, cat):
        assert type(y) == pd.Series
        assert type(cat) == pd.Series

        y_bid = pd.concat([y, cat], axis=1)
        y_bid.columns = ['y', 'cat']

        y = y_bid.apply(
            lambda row: row['y'] / self.max_per_building[row['cat']], axis=1)
        return y

    def fit_transform(self, y, cat):
        self.fit(y, cat)
        return self.transform(y, cat)

    def inverse_transform(self, y, cat):
        assert type(y) == pd.Series
        assert type(cat) == pd.Series

        y_bid = pd.concat([y, cat], axis=1)
        y_bid.columns = ['y', 'cat']

        y = y_bid.apply(
            lambda row: row['y'] * self.max_per_building[row['cat']], axis=1)
        return y


class StatusSaver(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # append "1," to status file
        with open('status.csv', 'a') as f:
            f.write('1\n')

        return self

    def transform(self, X):
        # Your transformation logic here
        # Return the transformed data
        return X


# TODO: how to handle function parameters? have to set them for test and submission

def add_features(Xy):
    Xy = add_features_time(Xy.copy())
    Xy = add_features_general(Xy.copy(), norm_radiation_cap=1)
    Xy = add_features_lag(Xy.copy())
    # Xy = add_features_interaction(Xy)
    Xy = add_features_mean(Xy.copy())
    Xy = add_features_differences(Xy.copy())
    return Xy


def add_features_differences(Xy):
    Xy['GHI_0_minus_-1'] = Xy['GHI'] - Xy['GHI_lag-1']
    Xy['GHI_0_minus_-2'] = Xy['GHI'] - Xy['GHI_lag-2']
    Xy['GHI_0_minus_1'] = Xy['GHI'] - Xy['GHI_lag1']
    Xy['GHI_0_minus_2'] = Xy['GHI'] - Xy['GHI_lag2']

    return Xy


def add_features_mean(Xy):
    Xy.loc[:, 'GHI_mean'] = Xy.groupby(['building_id', 'dayMonthYear'])[
        'GHI'].transform('mean')

    return Xy


def add_features_interaction(Xy):
    # general, somewhat intuitive interactions
    Xy.loc[:,'temp*GHI'] = Xy['GHI'].copy() * Xy['t_1000hPa:K'].copy()
    Xy.loc[:,'wind*humidity'] = Xy['wind_speed_10m:ms'] * Xy['relative_humidity_1000hPa:p']
    Xy.loc[:, 'sun_height*diff_rad'] = Xy['sun_elevation:d'] * Xy['diffuse_rad:W']

    # chat GPT
    Xy.loc[:,'hour*wind_speed_10m'] = Xy['hour'] * Xy['wind_speed_10m:ms']
    Xy.loc[:,'hour*clear_sky_rad'] = Xy['hour'] * Xy['clear_sky_rad:W']
    Xy.loc[:,'month*sun_elevation'] = Xy['month'] * Xy['sun_elevation:d']

    Xy.loc[:,'relative_humidity*air_density'] = Xy['relative_humidity_1000hPa:p'] * Xy['air_density_2m:kgm3']
    Xy.loc[:,'temperature*wind_speed'] = Xy['t_1000hPa:K'] * Xy['wind_speed_10m:ms']

    Xy.loc[:,'GHI*clear_sky_energy'] = Xy['GHI'] * Xy['clear_sky_energy_1h:J']
    Xy.loc[:,'GHI*sun_azimuth'] = Xy['GHI'] * Xy['sun_azimuth:d']

    Xy.loc[:,'snow_depth*temp*GHI'] = Xy['snow_depth:cm'] * Xy['temp*GHI']

    Xy.loc[:,'GHI_lag_interaction'] = Xy['GHI_lag-1'] * Xy['GHI_lag-2'] * Xy['GHI_lag1'] * Xy['GHI_lag2']
    Xy.loc[:,'GHI_lag_interaction_all'] = Xy['GHI_lag-1'] * Xy['GHI_lag-2'] * Xy['GHI_lag1'] * Xy['GHI_lag2'] * Xy['GHI']

    Xy.loc[:,'wind_speed*temp*GHI'] = Xy['wind_speed_10m:ms'] * Xy['t_1000hPa:K'] * Xy['GHI']

    Xy.loc[:,'cloud_base*clear_sky_energy'] = Xy['cloud_base_agl:m'] * Xy['clear_sky_energy_1h:J']

    Xy.loc[:,'precip*sun_elevation'] = Xy['precip_5min:mm'] * Xy['sun_elevation:d']
    Xy.loc[:,'supercooled_water*wind_speed'] = Xy['super_cooled_liquid_water:kgm2'] * Xy['wind_speed_10m:ms']

    return Xy


def add_features_lag(Xy, na_fill_value=None):
    # lagged feature of GHI
    Xy['GHI_lag-2'] = Xy.groupby('building_id')['GHI'].shift(-2)
    Xy['GHI_lag-1'] = Xy.groupby('building_id')['GHI'].shift(-1)
    Xy['GHI_lag1'] = Xy.groupby('building_id')['GHI'].shift(1)
    Xy['GHI_lag2'] = Xy.groupby('building_id')['GHI'].shift(2)

    if na_fill_value is None:
        Xy = Xy.dropna(
            subset=['GHI_lag-2', 'GHI_lag-1', 'GHI_lag1', 'GHI_lag2'])
    else:
        print('WADDEHADDEDUDEDA')
        print(na_fill_value)
        Xy = Xy.fillna(na_fill_value)

    return Xy


def add_features_general(Xy, norm_radiation_cap=None):

    Xy['GHI'] = Xy['diffuse_rad:W'] + Xy['direct_rad:W']
    Xy['wind_angle'] = np.arctan2(
        Xy['wind_speed_u_10m:ms'], Xy['wind_speed_v_10m:ms'])
    Xy['norm_radiation'] = (Xy['GHI'] / Xy['clear_sky_rad:W']).fillna(0)
    if norm_radiation_cap:
        Xy.loc[Xy['norm_radiation'] > norm_radiation_cap,
               'norm_radiation'] = norm_radiation_cap

    return Xy


def add_features_time(Xy):
    # Add monthYear column
    Xy['monthYear'] = Xy['date_forecast'].dt.to_period('M')

    # Add dayMonthYear column
    Xy['dayMonthYear'] = Xy['date_forecast'].dt.to_period('D')

    # Add month column
    Xy['month'] = Xy['date_forecast'].dt.month

    # Add week column
    # Xy['week'] = Xy['date_forecast'].dt.week

    # Add day column
    Xy['day'] = Xy['date_forecast'].dt.day

    # Add hour column
    Xy['hour'] = Xy['date_forecast'].dt.hour

    # Calculate difference between date_forecast and date_calc
    Xy['delta_forecast'] = (
        Xy['date_forecast']-Xy['date_calc']).apply(lambda x: x.total_seconds() / 3600)

    return Xy


def split_Xy_X_y(Xy):
    # Split into X and y
    X = Xy.drop(['pv_measurement'], axis=1)
    y = Xy['pv_measurement']
    return X, y


def get_splitted_data(merge_type='mean', split_strategy='2021_summer'):
    X, y, X_submission = get_data()
    # Merge X and y
    Xy = merge_largeX_smallY(X, y, merge_type=merge_type)

    # Split into train and test
    if split_strategy == '2021_summer':
        val_idx = Xy.date_forecast.between('2021-05-01', '2021-07-31')
    else:
        raise ValueError(f'split_strategy \'{split_strategy}\' not supported')

    Xy_train = Xy[~val_idx]
    Xy_val = Xy[val_idx]

    # # Split into X and y
    # X_train = Xy_train.drop(['pv_measurement'], axis=1)
    # y_train = Xy_train['pv_measurement']

    # X_val = Xy_val.drop(['pv_measurement'], axis=1)
    # y_val = Xy_val['pv_measurement']

    return Xy_train, Xy_val


def get_data():
    # Read data
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

    # Add building_id
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

    # Combine Data
    X_o = pd.concat([X_train_observed_a, X_train_observed_b, X_train_observed_c])
    X_e = pd.concat([X_train_estimated_a, X_train_estimated_b, X_train_estimated_c])
    X_submission = pd.concat([X_test_estimated_a, X_test_estimated_b, X_test_estimated_c])
    y = pd.concat([train_a, train_b, train_c]).dropna()

    # Add isEstimated column
    X_o['isEstimated'] = 0
    X_e['isEstimated'] = 1
    X_submission['isEstimated'] = 1


    # Combine
    X = pd.concat([X_o, X_e])

    return X, y, X_submission


def merge_largeX_smallY(X, y, merge_type='mean'):
    # Add time column that only holds the hour
    X['time'] = X['date_forecast'].dt.floor('H')

    if merge_type == 'mean':
        # Perform Transformation from 4 to 1 values per hour
        X = X.groupby(['building_id', 'time']).mean().reset_index()
        y = y
    else:
        raise ValueError(f'merge_type \'{merge_type}\' not supported')

    # Merge X and y
    Xy = pd.merge(X, y, on=['building_id', 'time']).reset_index(drop=True)
    # Drop the time column
    Xy = Xy.drop(columns=['time'])
    return Xy
