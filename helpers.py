import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.base import BaseEstimator, TransformerMixin


class DataWrapper:
    def __init__(self,
                 impute_strategy = 'fbfill',
                 aggregation_strats = ['mean'],
                 y_scalers = None,
                 transform_pred_strategy = ['clip'],
                 training_filters=False,
                 validation_filters=False,
                 test_filters=False,
                 bagging_filters=False, # these are applied additionally to the other filters on val, test, sub
                 ):
        
        self.impute_strategy = impute_strategy
        self.aggregation_strats = aggregation_strats
        #self.date_columns = date_columns
        self.y_scalers = y_scalers
        self.transform_pred_strategy = transform_pred_strategy
        self.training_filters = training_filters
        self.validation_filters = validation_filters
        self.test_filters = test_filters
        self.bagging_filters = bagging_filters

        self.basic_features = [
            'isEstimated',
            'delta_forecast',
            'date_forecast',
            #'hourDayMonthYear',
            'dayMonthYear',
            'monthYear',
            'month',
            'hour',
        ]

        self.X = None
        self.X_agg = None

        self.X_sub = None
        self.X_sub_agg = None

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self.train_agg_idx = None
        self.val_agg_idx = None
        self.test_agg_idx = None

        self._readData()
        self._set_idxs()

    def _readData(self):
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

        # Adding building ID
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
        X_submission = pd.concat([X_test_estimated_a, X_test_estimated_b, X_test_estimated_c]).reset_index(drop=True)
        y = pd.concat([train_a, train_b, train_c]).dropna(subset=['pv_measurement']).rename(columns={'time': 'hourDayMonthYear'}).reset_index(drop=True)

        # Add isEstimated column
        X_o['isEstimated'] = 0
        X_e['isEstimated'] = 1
        X_submission['isEstimated'] = 1


        # Combine
        X = pd.concat([X_o, X_e]).reset_index(drop=True)

        # Create additional feature for estimated data "delta_forecast"
        X['delta_forecast'] = (X['date_forecast']-X['date_calc']).apply(lambda x: x.total_seconds() / 3600)
        X_submission['delta_forecast'] = (X_submission['date_forecast']-X_submission['date_calc']).apply(lambda x: x.total_seconds() / 3600)

        X = self._cleanData(X) # dont apply to submission data

        y = self._cleanY(y)
        X = self._addTimeFeatures(X)
        X = self._addFeatureInteractions_preAgg(X)
        X = self._addAggFeatures_preAgg(X)
        X = self._addLagFeatures_preAgg(X)
        X = self._cleanX(X)
        X_agg, y_agg = self._aggregateData(X, y)

        # merge x and y to ensure that we have the same indices
        XY_TEMP = X.merge(y, on=['building_id', 'hourDayMonthYear'], how='inner')
        X = XY_TEMP.drop(columns=['pv_measurement'])
        y = XY_TEMP[['building_id', 'hourDayMonthYear', 'pv_measurement']]

        XY_TEMP_AGG = X_agg.merge(y_agg, on=['building_id', 'hourDayMonthYear'], how='inner')
        X_agg = XY_TEMP_AGG.drop(columns=['pv_measurement'])
        y_agg = XY_TEMP_AGG[['building_id', 'hourDayMonthYear', 'pv_measurement']]

        self.y_agg_scaled = self.transform_y(y_agg['pv_measurement'], X_agg)
        self.y_scaled = self.transform_y(y['pv_measurement'], X)

        self.X_agg = X_agg
        self.y_agg = y_agg['pv_measurement']
        self.X = X
        self.y = y['pv_measurement']
        
        # submission data
        X_submission = self._addTimeFeatures(X_submission)
        X_submission = self._addFeatureInteractions_preAgg(X_submission)
        X_submission = self._addAggFeatures_preAgg(X_submission)
        X_submission = self._addLagFeatures_preAgg(X_submission)
        X_submission = self._cleanX(X_submission)
        X_submission_agg, _ = self._aggregateData(X_submission)

        self.X_sub = X_submission
        self.X_sub_agg = X_submission_agg



    def _cleanData(self, X):
        X = X[~((X['building_id'] == 'c') & (X['date_forecast'] < '2019-09-06'))] # C is powered on on this day -> weird measurements
        X = X[~((X['building_id'] == 'c') & (X['date_forecast'].between('2020-05-13','2020-05-14')))] # C is powered on on this day -> weird measurements

        # low value analysis
        remove_idx_1 = (X['date_forecast'].between('2019-06-05 15:59:00', '2019-06-06 17:59:00')) & (X['building_id'] == 'a') # weird 0/low values
        remove_idx_2 = (X['date_forecast'].between('2020-05-13 01:59:00', '2020-05-13 12:59:00')) & (X['building_id'] == 'b') # probably to low values
        remove_idx_3 = (X['date_forecast'].between('2022-05-02 07:59:00', '2022-05-02 09:59:00')) & (X['building_id'] == 'b') # low/0 values
        remove_idx_4 = (X['date_forecast'].between('2022-05-02 11:59:00', '2022-05-02 12:59:00')) & (X['building_id'] == 'b') # low/0 values
        remove_idx_5 = (X['date_forecast'].between('2020-05-11 00:59:00', '2020-05-12 12:59:00')) & (X['building_id'] == 'c') # way too low production
        remove_idx_6 = (X['date_forecast'].between('2023-04-26 00:59:00', '2023-04-26 13:59:00')) & (X['building_id'] == 'c') # way too low production

        # snow analysis
        snow_remove_idx1 =  (X['date_forecast'].between('2020-03-28 00:00:00','2020-04-01 00:00:00')) & (X['building_id'] == 'c') 
        snow_remove_idx2 =  (X['date_forecast'].between('2020-03-29 00:00:00','2020-04-02 00:00:00')) & (X['building_id'] == 'b') 
        snow_remove_idx3 =  (X['date_forecast'].between('2020-04-03 00:00:00','2020-04-06 00:00:00')) & (X['building_id'] == 'c') 
        snow_remove_idx4 =  (X['date_forecast'].between('2021-04-09 00:00:00','2021-04-11 00:00:00')) & (X['building_id'] == 'a') 
        snow_remove_idx5 =  (X['date_forecast'].between('2022-04-01 00:00:00','2022-04-08 00:00:00'))
        snow_remove_idx6 =  (X['date_forecast'].between('2022-04-07 00:00:00','2022-04-13 00:00:00')) & (X['building_id'] == 'b')
        snow_remove_idx7 =  (X['date_forecast'].between('2023-04-01 00:00:00','2023-04-11 00:00:00')) & (X['building_id'] == 'c')
        snow_remove_idx_8 = (X['date_forecast'].between('2021-04-06 00:00:00', '2021-04-15 00:00:00')) & (X['building_id'] == 'c') # way too low production

        # spikes at negative sun elevation
        weird_val_night_idx_1 = X['date_forecast'].between('2022-11-25 15:59:00', '2022-11-25 18:59:00') & (X['building_id'] == 'b')
        weird_val_night_idx_2 = X['date_forecast'].between('2023-03-28 00:59:00', '2023-03-28 03:59:00') & (X['building_id'] == 'b')

        remove_idx = remove_idx_1 | remove_idx_2 | remove_idx_3 | remove_idx_4 | remove_idx_5 | remove_idx_6 | snow_remove_idx1 | snow_remove_idx2 | snow_remove_idx3 | snow_remove_idx4 | snow_remove_idx5 | snow_remove_idx6 | snow_remove_idx7 | snow_remove_idx_8 | weird_val_night_idx_1 | weird_val_night_idx_2

        X = X[~remove_idx].reset_index(drop=True)

        return X
    
    def _cleanY(self, y):
        # Cleaning pv_measurement
        eq_prev_row = (
            ((y.pv_measurement == y.pv_measurement.shift(-2)) & (y.pv_measurement == y.pv_measurement.shift(-1))) |
            ((y.pv_measurement == y.pv_measurement.shift(-1)) & (y.pv_measurement == y.pv_measurement.shift(1))) | 
            (y.pv_measurement == y.pv_measurement.shift(1)) & (y.pv_measurement == y.pv_measurement.shift(2))
            ) & (y.pv_measurement > 0)
        y = y[~eq_prev_row].reset_index(drop=True) # Mostly Solar Panel B has some constant values for up to 1000 measurements


        zeroData_24h = y.groupby(['building_id'])['pv_measurement'].transform(lambda x: x.rolling(24*4, 24*4).sum()) == 0
        for i in range(1,24*4):
            zeroData_24h = zeroData_24h | zeroData_24h.copy(deep=True).shift(-1)
        y = y[~zeroData_24h].reset_index(drop=True) # Solar Panels B and C have some 0 values for longer than 24h, also in summer

        return y

    def _addTimeFeatures(self, X):
        X['hourDayMonthYear'] = X['date_forecast'].dt.floor('H')
        X['dayMonthYear'] = X['date_forecast'].dt.floor('D')
        X['monthYear'] = X['date_forecast'].dt.to_period('M')

        X['month'] = X['date_forecast'].dt.month
        X['hour'] = X['date_forecast'].dt.hour

        return X
    
    def _addFeatureInteractions_preAgg(self, X):
        # sum
        X['GHI'] = X['diffuse_rad:W'] + X['direct_rad:W']

        # arctan
        X['wind_angle'] = np.arctan2(X['wind_speed_u_10m:ms'], X['wind_speed_v_10m:ms'])

        # product
        X['temp*GHI'] = X['GHI'] * X['t_1000hPa:K']
        X['wind*humidity'] = X['wind_speed_10m:ms'] * X['relative_humidity_1000hPa:p']
        X['sun_height*diff_rad'] = X['sun_elevation:d'] * X['diffuse_rad:W']

        # sin/cos
        X['wind_angle_sin'] = np.sin(X['wind_angle'])
        X['wind_angle_cos'] = np.cos(X['wind_angle'])

        X['hour_sin'] = np.sin(X['hour'] * (2. * np.pi / 24))
        X['hour_cos'] = np.cos(X['hour'] * (2. * np.pi / 24))

        return X
    
    def _addAggFeatures_preAgg(self, X):
        # add daily mean and std of GHI
        X['GHI_daily_mean'] = X.groupby(['building_id', 'dayMonthYear'])['GHI'].transform('mean')
        X['GHI_daily_std'] = X.groupby(['building_id', 'dayMonthYear'])['GHI'].transform('std')

        X['effective_cloud_cover_5h_mean'] = X.groupby(['building_id'])['effective_cloud_cover:p'].rolling(window=5*4, center=True).mean().reset_index(drop=True)
        return X

    def _aggregateData(self, X, y=None):
        # basic columns --> should be equal in all aggregation strategies
        X_ASSEMBLE = X[['building_id', 'hourDayMonthYear']+self.basic_features].groupby(['building_id', 'hourDayMonthYear']).nth(0).set_index(['building_id', 'hourDayMonthYear']) # get empty DF with only index
        #X_ASSEMBLE = X[['building_id', 'hourDayMonthYear']].drop_duplicates().reset_index(drop=True)

        for agg_strat in self.aggregation_strats:
            
            # set index is needed as nth(x) seems to reset the index after grouping
            if agg_strat == 'mean':
                X_TEMP = X.drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).mean()
                X_TEMP.columns = ['mean_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == '00':
                X_TEMP = X.sort_values('date_forecast').drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).nth(0).set_index(['building_id', 'hourDayMonthYear'])
                X_TEMP.columns = ['00_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == '15':
                X_TEMP = X.sort_values('date_forecast').drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).nth(1).set_index(['building_id', 'hourDayMonthYear'])
                X_TEMP.columns = ['15_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == '30':
                X_TEMP = X.sort_values('date_forecast').drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).nth(2).set_index(['building_id', 'hourDayMonthYear'])
                X_TEMP.columns = ['30_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == '45':
                X_TEMP = X.sort_values('date_forecast').drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).nth(3).set_index(['building_id', 'hourDayMonthYear'])
                X_TEMP.columns = ['45_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == 'min':
                X_TEMP = X.drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).min()
                X_TEMP.columns = ['min_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == 'max':
                X_TEMP = X.drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).max()
                X_TEMP.columns = ['max_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == 'std':
                X_TEMP = X.drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).std()
                X_TEMP.columns = ['std_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)
            elif agg_strat == 'median':
                X_TEMP = X.drop(columns=self.basic_features).groupby(['building_id', 'hourDayMonthYear']).median()
                X_TEMP.columns = ['median_' + col for col in X_TEMP.columns]
                X_ASSEMBLE = X_ASSEMBLE.merge(X_TEMP, how='inner', left_index=True, right_index=True)

        X_ASSEMBLE = X_ASSEMBLE.reset_index().copy() # get building_id and hourDayMonthYear back as columns
        if y is not None:
            y = y.copy()
        # y is already aggregated
        return X_ASSEMBLE, y

    def _addLagFeatures_preAgg(self, X):
        X['GHI_lag_-1h'] = X.groupby(['building_id'])['GHI'].shift(-1*4)
        X['GHI_lag_1h'] = X.groupby(['building_id'])['GHI'].shift(1*4)

        X['temp*GHI_lag_-1h'] = X.groupby(['building_id'])['temp*GHI'].shift(-1*4)
        X['temp*GHI_lag_1h'] = X.groupby(['building_id'])['temp*GHI'].shift(1*4)

        # new features
        X['effective_cloud_cover:p_-1h'] = X.groupby(['building_id'])['effective_cloud_cover:p'].shift(-1*4)
        X['effective_cloud_cover:p_1h'] = X.groupby(['building_id'])['effective_cloud_cover:p'].shift(1*4)


        X['cloud_base_agl:m_-1h'] = X.groupby(['building_id'])['cloud_base_agl:m'].shift(-1*4)
        X['cloud_base_agl:m_1h'] = X.groupby(['building_id'])['cloud_base_agl:m'].shift(1*4)

        return X
    
    def _cleanX(self, X):
        # drop columns
        # impute and drop columns
        drop_cols = ['elevation:m']
        drop_cols += [col for col in X.columns if ('snow' in col)]

        X = X.drop(columns=drop_cols)
        
        impute_cols = [
            'ceiling_height_agl:m',
            'cloud_base_agl:m',
            'cloud_base_agl:m_-1h',
            'cloud_base_agl:m_1h',
            'GHI_daily_std',
            # effective_cloud_cover_5h_mean,
            'GHI_lag_-1h',
            'GHI_lag_1h',
            'temp*GHI_lag_-1h',
            'temp*GHI_lag_1h',
            'effective_cloud_cover:p_-1h',
            'effective_cloud_cover:p_1h',
            'effective_cloud_cover_5h_mean'
        ]

        X.loc[:,impute_cols] = X.copy().sort_values(by=['date_forecast','building_id']).sort_values(by=[f'date_forecast',f'building_id']).loc[:,impute_cols].bfill().ffill().sort_index()

        X['delta_forecast'] = X['delta_forecast'].fillna(0)


        return X

    def get_train(self,aggregated,drop_val=True,drop_test=True,y_scaled=False):
        if aggregated:
            keep_idx = self.train_agg_idx
            if drop_val:
                keep_idx = keep_idx & ~self.val_agg_idx
            if drop_test:
                keep_idx = keep_idx & ~self.test_agg_idx

            if y_scaled:
                return self.X_agg[keep_idx].reset_index(drop=True), self.y_agg_scaled[keep_idx].reset_index(drop=True)
            else:
                return self.X_agg[keep_idx].reset_index(drop=True), self.y_agg[keep_idx].reset_index(drop=True)

        else:
            keep_idx = self.train_idx
            if drop_val:
                keep_idx = keep_idx & ~self.val_idx
            if drop_test:
                keep_idx = keep_idx & ~self.test_idx
            
            if y_scaled:
                return self.X[keep_idx].reset_index(drop=True), self.y_scaled[keep_idx].reset_index(drop=True)
            else:
                return self.X[keep_idx].reset_index(drop=True), self.y[keep_idx].reset_index(drop=True)


    def get_val(self, aggregated,y_scaled=False):
        if aggregated:
            X_val = self.X_agg[self.val_agg_idx]
            y_val = self.y_agg[self.val_agg_idx]
        else:
            X_val = self.X[self.val_idx]
            y_val = self.y[self.val_idx]
        if y_scaled:
            return X_val.reset_index(drop=True), self.y_agg_scaled[self.val_agg_idx].reset_index(drop=True)
        else:
            return X_val.reset_index(drop=True), y_val.reset_index(drop=True)

    def get_test(self, aggregated,y_scaled=False):
        if aggregated:
            X_test = self.X_agg[self.test_agg_idx]
            y_test = self.y_agg[self.test_agg_idx]
        else:
            X_test = self.X[self.test_idx]
            y_test = self.y[self.test_idx]
        if y_scaled:
            return X_test.reset_index(drop=True), self.y_agg_scaled[self.test_agg_idx].reset_index(drop=True)
        else:
            return X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    
    def get_sub(self, aggregated):
        if aggregated:
            X_sub = self.X_sub_agg
        else:
            X_sub = self.X_sub
        return X_sub.reset_index(drop=True)

    def _filter_to_idx(self, X, filters):
        filter_idxs = []
        for filter in filters:
            if filter['feature'] == 'date_forecast':
                if filter['operator'] == 'before':
                    filter_idxs.append(X['date_forecast'] < filter['value'])
                elif filter['operator'] == 'after':
                    filter_idxs.append(X['date_forecast'] > filter['value'])
                elif filter['operator'] == 'between':
                    filter_idxs.append(X['date_forecast'].between(filter['value'][0], filter['value'][1]))
            elif filter['feature'] == 'hour':
                if filter['operator'] == 'before':
                    filter_idxs.append(X['hour'] < filter['value'])
                elif filter['operator'] == 'after':
                    filter_idxs.append(X['hour'] > filter['value'])
                elif filter['operator'] == 'between':
                    filter_idxs.append(X['hour'].between(filter['value'][0], filter['value'][1]))
            elif filter['feature'] == 'month':
                if filter['operator'] == 'before':
                    filter_idxs.append(X['month'] < filter['value'])
                elif filter['operator'] == 'after':
                    filter_idxs.append(X['month'] > filter['value'])
                elif filter['operator'] == 'between':
                    filter_idxs.append(X['month'].between(filter['value'][0], filter['value'][1]))
            elif filter['feature'] == 'building_id':
                if filter['operator'] == 'equal':
                    filter_idxs.append(X['building_id'] == filter['value'])
                elif filter['operator'] == 'not_equal':
                    filter_idxs.append(X['building_id'] != filter['value'])
                elif filter['operator'] == 'in':
                    filter_idxs.append(X['building_id'].isin(filter['value']))
                elif filter['operator'] == 'not_in':
                    filter_idxs.append(~X['building_id'].isin(filter['value']))
            elif filter['feature'] == 'isEstimated':
                if filter['operator'] == 'equal':
                    filter_idxs.append(X['isEstimated'] == filter['value'])
            elif filter['feature'] == 'sun_elevation:d':
                if filter['operator'] == 'below':
                    filter_idxs.append(X['sun_elevation:d'] < filter['value'])
                elif filter['operator'] == 'above':
                    filter_idxs.append(X['sun_elevation:d'] > filter['value'])
                elif filter['operator'] == 'between':
                    filter_idxs.append(X['sun_elevation:d'].between(filter['value'][0], filter['value'][1]))



        # combine filters
        return(np.array(filter_idxs)).all(axis=0)

    def _set_idxs(self):
        self.train_idx = self._filter_to_idx(self.X, self.training_filters)
        self.val_idx = self._filter_to_idx(self.X, self.validation_filters)
        self.test_idx = self._filter_to_idx(self.X, self.test_filters)

        self.train_agg_idx = self._filter_to_idx(self.X_agg, self.training_filters)
        self.val_agg_idx = self._filter_to_idx(self.X_agg, self.validation_filters)
        self.test_agg_idx = self._filter_to_idx(self.X_agg, self.test_filters)




    def get_mae(self, y_pred, X, y, is_aggregated):
        assert len(y_pred) == len(X)
        assert len(y_pred) == len(y)
        y_pred = pd.Series(y_pred, name='prediction')
        y_pred = self.transform_prediction(y_pred, X)


        # XY_TEMP = pd.concat(
        #         [
        #             X.reset_index(drop=True),
        #             y_pred
        #         ], 
        #         axis=1)
        
        if is_aggregated:
            return np.abs(y_pred - y).mean()
        else:
            XY_TEMP_AGG = XY_TEMP.groupby(['building_id', 'hourDayMonthYear']).mean().reset_index()
            return np.abs(XY_TEMP_AGG['prediction'] - XY_TEMP_AGG['pv_measurement']).mean()

        

    def transform_y(self, y, X):
        y = y.rename('y').reset_index(drop=True)
        if self.y_scalers is not None:
            for scaler in self.y_scalers:
                y = scaler.fit_transform(y,X)
        return y
    
    def transform_prediction(self, y_pred, X, sun_el_thr=-100):
        y_pred = y_pred.rename('y').reset_index(drop=True)
        if self.y_scalers is not None:
            y_scalers_reversed = self.y_scalers.copy()
            y_scalers_reversed.reverse()
            for scaler in y_scalers_reversed:
                y_pred = scaler.inverse_transform(y_pred,X)
        for pred_strat in self.transform_pred_strategy:
            if pred_strat == 'clip':
                y_pred = y_pred.clip(lower=0)
            if pred_strat == 'sun_el_thr':

                y_pred = y_pred * (X['max_sun_elevation:d'] > sun_el_thr)
        return y_pred

    def y_pred_to_csv(self, y_pred, X, is_aggregated,name='submission.csv', sun_el_thr=-100):
        y_pred = self.transform_prediction(y_pred, X, sun_el_thr=sun_el_thr)
        if is_aggregated:
            y_pred.to_csv(name, index=True)
        else:
            XY_TEMP = pd.concat(
                [
                    X.reset_index(drop=True),
                    y_pred
                ], 
                axis=1)
        
            XY_TEMP_AGG = XY_TEMP.groupby(['building_id', 'hourDayMonthYear']).mean().sort_values(['building_id', 'hourDayMonthYear']).reset_index()
            XY_TEMP_AGG['prediction'].to_csv(name, index=True)

        XY_TEMP = pd.concat(
                [
                    X.reset_index(drop=True),
                    y_pred
                ], 
                axis=1)
        
    def plot_Pred_vs_PV(self,y_pred,y,X,start_idx=None,end_idx=None):
        y_pred = self.transform_prediction(y_pred, X)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(y)
        plt.figure(figsize=(20,6))
        # plt.ylim(0,300)
        plt.plot(y_pred[start_idx:end_idx],label='prediction')
        plt.plot(y[start_idx:end_idx],label='pv_measurement')
        plt.legend()
        plt.show()

    def plot_Residuals(self,y_pred,y,X):
        y_pred = self.transform_prediction(y_pred, X)
        Xy = pd.concat([X.reset_index(drop=True),y.reset_index(drop=True).rename('pv_measurement')],axis=1)
        res = y - y_pred
        plt.figure(figsize=(20,6))
        sns.lineplot(data=Xy,x='date_forecast',y=res,hue='building_id',legend='full')
        plt.xticks(rotation=90)
        plt.show()

    def plot_yPredSub_vs_submission(self,ypred,X,sub_file='./submission_files/152_029_xgboost.csv'):
        ypred = self.transform_prediction(ypred, X)
        xgb = pd.read_csv(sub_file)['prediction']
        plt.figure(figsize=(20,6))
        sns.lineplot(ypred, color='red')
        sns.lineplot(xgb, color='black')
        plt.show()
        


# Class to scale y
class Y_Scaler_MaxAbs_per_building:
    def __init__(self):
        self.max_per_building = {}

    def fit(self, y, X):
        assert type(y) == pd.Series
        assert type(X) == pd.DataFrame

        y_bid = pd.concat([y.rename('pv_measurement'), X], axis=1)

        self.max_per_building = y_bid.groupby('building_id')['pv_measurement'].max().to_dict()
        return self

    def transform(self, y, X):
        assert type(y) == pd.Series
        assert type(X) == pd.DataFrame        
        
        y_bid = pd.concat([y.rename('pv_measurement'), X], axis=1)

        y = y_bid.apply(
            lambda row: row['pv_measurement'] / self.max_per_building[row['building_id']], axis=1)
        return y

    def fit_transform(self, y, X):
        self.fit(y, X)
        return self.transform(y, X)

    def inverse_transform(self, y, X):
        assert type(y) == pd.Series
        assert type(X) == pd.DataFrame        
        
        y_bid = pd.concat([y.rename('pv_measurement'), X], axis=1)

        y = y_bid.apply(
            lambda row: row['pv_measurement'] * self.max_per_building[row['building_id']], axis=1)
        return y
    


class StatusSaver(BaseEstimator, TransformerMixin):
    def __init__(self):
        # create text file
        open('status.csv', 'w').close()

    def fit(self, X, y=None):
        # append "1," to status file
        with open('status.csv', 'a') as f:
            f.write('1\n')

        return self

    def transform(self, X):
        # Your transformation logic here
        # Return the transformed data
        return X