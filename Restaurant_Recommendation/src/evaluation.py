import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    precision_at_5 = np.mean([1 if pred >= 4 and true >= 4 else 0 for pred, true in zip(y_pred, y_true)])
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Precision@5: {precision_at_5:.4f}")
    return rmse, mae, precision_at_5