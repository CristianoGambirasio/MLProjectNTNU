# Useful links
bayesian optimization:
https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html

# ideas
- look at trends in MAE of model
- elbow method for PCA feature selection
- create feature 'is_estimated_weather' (or 2 models)
- hyperparameter tuning (models,parameters,...)
- bayesian CV with pipelines etc...
- Creating a model only on observed data, then create a model ensembling the first model and another model created with only estimated data
in this way the final model should give a rough prediction thanks to observed data, and then refine the prevision with the second (estimated data) model



- new features
    - GHI
    - lagged (GHI,...)
    - take all 4 values of GHI,...
    - wind direction + new speed
    - normalize radiation based on clear sky radiation

- create Test set 
- use optuna
- compare feature importance of xgboost with lasso feature importance

- plot feature vs pv_measurement
- new train/test split
- use day of year instead of month
- shuffeling data before training

- try different standardizers per building
    



- remove weird lag features
- sort min->max
- add mean per hour


- stacking models by random seed