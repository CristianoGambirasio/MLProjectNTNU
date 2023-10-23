import pandas as pd
train_a = pd.read_parquet('A/train_targets.parquet')

X_train_estimated_a = pd.read_parquet('A/X_train_estimated.parquet')

X_train_observed_a = pd.read_parquet('A/X_train_observed.parquet')

X_test_estimated_a = pd.read_parquet('A/X_test_estimated.parquet')


# Aggregating all_X
X = X_train_observed_a.copy()
X['time_hour'] = X['date_forecast'].dt.floor('H')
X = X.groupby(['time_hour']).mean().reset_index()

# Renaming time column in all_y
y = train_a.rename(columns={'time': 'time_hour'})

# Merging all_X_aggregated and all_y
data = pd.merge(X,y,on=['time_hour'],how='left') # right join to keep all y values

data.dropna(subset='pv_measurement',inplace=True)

cutoff = int(len(X)*0.8)

data_train = data[:cutoff]
data_test = data[cutoff:]

from statsmodels.tsa.seasonal import MSTL

analysis = data[['time_hour','pv_measurement']]

analysis.set_index('time_hour', inplace=True)

print('a')

model = MSTL(analysis[:17521], periods=(24,24*365))
res = model.fit()
res.plot()

"""
decompose_result_mult = seasonal_decompose(analysis[:8760], model="additive")

decompose_result_mult.plot()"""