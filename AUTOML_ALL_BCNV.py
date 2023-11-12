algos = ['LightGBM', 'Xgboost', 'CatBoost']

from supervised.automl import AutoML

from helpers import *

training_filters = [
    {
        'feature': 'month',
        'operator': 'between',
        'value': [3, 10]
    }
]

validation_filters = [
    {
        'feature': 'date_forecast',
        'operator': 'between',
        'value': ['2020-05-01', '2020-07-01']
    }
]

test_filters = [
    {
        'feature': 'date_forecast',
        'operator': 'between',
        'value': ['2021-05-01', '2021-07-01']
    }
]


agg_strats = ['min','max','mean']

m_wrap = DataWrapper(
    impute_strategy = 'fbfill',
    aggregation_strats = agg_strats,
    y_scalers = [Y_Scaler_MaxAbs_per_building()],
    transform_pred_strategy = [],
    training_filters=training_filters,
    validation_filters=validation_filters,
    test_filters=test_filters,
    bagging_filters=False, # these are applied additionally to the other filters on val, test, sub
    golden_features=False,
    features_selection=False
    )

X_train_agg, y_train_agg = m_wrap.get_train(aggregated=True,drop_val=False,drop_test=False,y_scaled=True)
X_sub_agg = m_wrap.get_sub(aggregated=True)

train_idx = X_train_agg[~X_train_agg['date_forecast'].between('2020-05-01', '2020-07-01')].index
val_idx = X_train_agg[X_train_agg['date_forecast'].between('2020-05-01', '2020-07-01')].index

X_train_agg = X_train_agg.select_dtypes(exclude=['datetime','timedelta','period[M]'])


mod = AutoML(
    results_path='AutoML_Results_ALL_BCNV/',
    mode='Compete',
    total_time_limit=60*60*8,
    eval_metric='mae',
    algorithms=algos,
    train_ensemble=True,
    stack_models=True,
    verbose=2,
    random_state=42,
)

mod.fit(X_train_agg, y_train_agg)