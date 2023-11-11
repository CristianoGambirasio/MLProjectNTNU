# Summary of 12_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.05
- **depth**: 9
- **rsm**: 0.9
- **loss_function**: MAPE
- **eval_metric**: MAE
- **explain_level**: 0

## Validation
 - **validation_type**: custom

## Optimized metric
mae

## Training time

152.8 seconds

### Metric details:
| Metric   |      Score |
|:---------|-----------:|
| MAE      | 0.0509137  |
| MSE      | 0.00807282 |
| RMSE     | 0.0898489  |
| R2       | 0.891523   |
| MAPE     | 8.6203e+11 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
