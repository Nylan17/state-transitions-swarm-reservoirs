import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def nrmse(y_true, y_pred):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).
    """
    return mean_absolute_error(y_true, y_pred)

def r_squared(y_true, y_pred):
    """
    Calculate the R^2 (coefficient of determination) regression score.
    """
    return r2_score(y_true, y_pred)
