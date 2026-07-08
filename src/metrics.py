import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1.0):
    # Guard divide-by-zero on near-zero sales weeks
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def wmae(y_true, y_pred, is_holiday, holiday_weight=5):
    """
    Weighted Mean Absolute Error
    WMAE is the Kaggle competition's official metric (holidays weighted 5x).
    """
    weights = np.where(is_holiday, holiday_weight, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)
