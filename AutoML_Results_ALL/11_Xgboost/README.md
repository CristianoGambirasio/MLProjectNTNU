# Summary of 11_Xgboost

[<< Go back](../README.md)


## Extreme Gradient Boosting (Xgboost)
- **n_jobs**: -1
- **objective**: reg:squarederror
- **eta**: 0.1
- **max_depth**: 8
- **min_child_weight**: 1
- **subsample**: 0.6
- **colsample_bytree**: 0.5
- **eval_metric**: mae
- **explain_level**: 0

## Validation
 - **validation_type**: custom

## Optimized metric
mae

## Training time

12.1 seconds

### Metric details:
| Metric   |       Score |
|:---------|------------:|
| MAE      | 0.0522231   |
| MSE      | 0.00843954  |
| RMSE     | 0.091867    |
| R2       | 0.886595    |
| MAPE     | 9.09215e+11 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)