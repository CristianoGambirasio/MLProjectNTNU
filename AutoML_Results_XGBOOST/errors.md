## Error for 15_Xgboost_Stacked

Input X contains NaN.
MiniBatchKMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
Traceback (most recent call last):
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/base_automl.py", line 1195, in _fit
    trained = self.train_model(params)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/base_automl.py", line 401, in train_model
    mf.train(results_path, model_subpath)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/model_framework.py", line 195, in train
    X_train, y_train, sample_weight = self.preprocessings[
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/preprocessing/preprocessing.py", line 203, in fit_and_transform
    self._kmeans.fit(X_train[numeric_cols], y_train)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/preprocessing/kmeans_transformer.py", line 58, in fit
    self._kmeans.fit(X)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/cluster/_kmeans.py", line 2084, in fit
    X = self._validate_data(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/base.py", line 605, in _validate_data
    out = check_array(X, input_name="X", **check_params)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 957, in check_array
    _assert_all_finite(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 122, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
MiniBatchKMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for 4_Xgboost_KMeansFeatures_Stacked

Input X contains NaN.
MiniBatchKMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
Traceback (most recent call last):
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/base_automl.py", line 1195, in _fit
    trained = self.train_model(params)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/base_automl.py", line 401, in train_model
    mf.train(results_path, model_subpath)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/model_framework.py", line 195, in train
    X_train, y_train, sample_weight = self.preprocessings[
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/preprocessing/preprocessing.py", line 203, in fit_and_transform
    self._kmeans.fit(X_train[numeric_cols], y_train)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/preprocessing/kmeans_transformer.py", line 58, in fit
    self._kmeans.fit(X)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/cluster/_kmeans.py", line 2084, in fit
    X = self._validate_data(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/base.py", line 605, in _validate_data
    out = check_array(X, input_name="X", **check_params)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 957, in check_array
    _assert_all_finite(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 122, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
MiniBatchKMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

## Error for 4_Xgboost_categorical_mix_KMeansFeatures_Stacked

Input X contains NaN.
MiniBatchKMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
Traceback (most recent call last):
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/base_automl.py", line 1195, in _fit
    trained = self.train_model(params)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/base_automl.py", line 401, in train_model
    mf.train(results_path, model_subpath)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/model_framework.py", line 195, in train
    X_train, y_train, sample_weight = self.preprocessings[
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/preprocessing/preprocessing.py", line 203, in fit_and_transform
    self._kmeans.fit(X_train[numeric_cols], y_train)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/supervised/preprocessing/kmeans_transformer.py", line 58, in fit
    self._kmeans.fit(X)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/cluster/_kmeans.py", line 2084, in fit
    X = self._validate_data(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/base.py", line 605, in _validate_data
    out = check_array(X, input_name="X", **check_params)
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 957, in check_array
    _assert_all_finite(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 122, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/cristianogambirasio/.local/lib/python3.9/site-packages/sklearn/utils/validation.py", line 171, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
MiniBatchKMeans does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values


Please set a GitHub issue with above error message at: https://github.com/mljar/mljar-supervised/issues/new

