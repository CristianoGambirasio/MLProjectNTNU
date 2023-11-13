
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