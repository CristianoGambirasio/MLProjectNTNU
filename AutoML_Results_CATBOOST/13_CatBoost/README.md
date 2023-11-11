# Summary of 13_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.05
- **depth**: 6
- **rsm**: 1.0
- **loss_function**: MAPE
- **eval_metric**: MAE
- **explain_level**: 0

## Validation
 - **validation_type**: custom

## Optimized metric
mae

## Training time

43.0 seconds

### Metric details:
| Metric   |       Score |
|:---------|------------:|
| MAE      | 0.051427    |
| MSE      | 0.00825953  |
| RMSE     | 0.090882    |
| R2       | 0.889014    |
| MAPE     | 7.17195e+11 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
