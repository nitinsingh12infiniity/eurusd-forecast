# models.py
# The three models: LSTM, XGBoost and Ridge regression.
# Also one predict() function that works for all three.
 
import json
 
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
 
import common
 
 
# ---------------------------------------------------------------
# Model 1: LSTM (neural network that reads the 60 bars in order)
# ---------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 features go in, the LSTM keeps a memory of size 100
        self.lstm = nn.LSTM(input_size=5, hidden_size=100, batch_first=True)
        # turn the memory of size 100 into 1 number: the predicted close
        self.linear = nn.Linear(100, 1)
 
    def forward(self, x):
        # x has shape (batch, 60, 5)
        lstm_output, _ = self.lstm(x)
        last_step = lstm_output[:, -1, :]      # keep only the output after the 60th bar
        prediction = self.linear(last_step)    # shape (batch, 1)
        return prediction.squeeze(1)           # shape (batch,)
 
 
def load_lstm(path):
    model = LSTMModel()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
 
 
# ---------------------------------------------------------------
# Model 2: XGBoost (many small decision trees)
# ---------------------------------------------------------------
def load_xgboost(path):
    booster = xgb.Booster()
    booster.load_model(path)
    return booster
 
 
# ---------------------------------------------------------------
# Model 3: Ridge regression (a straight-line model, written by hand)
# ---------------------------------------------------------------
def fit_ridge(X, y, alpha):
    """Closed-form ridge regression: w = (Xc'Xc + alpha*I)^-1 Xc'yc
 
    We centre X and y first (subtract their averages) so the intercept
    is not shrunk by alpha.
    """
    X_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    X_centred = X - X_mean
    y_centred = y - y_mean
 
    n_features = X.shape[1]
    A = X_centred.T @ X_centred + alpha * np.eye(n_features)
    b = X_centred.T @ y_centred
    weights = np.linalg.solve(A, b)
    intercept = y_mean - float(X_mean @ weights)
 
    return {"weights": weights.tolist(), "intercept": intercept, "alpha": alpha}
 
 
def predict_ridge(model, X_flat):
    weights = np.array(model["weights"])
    return X_flat @ weights + model["intercept"]
 
 
def save_ridge(model, path):
    with open(path, "w") as f:
        json.dump(model, f)
 
 
def load_ridge(path):
    with open(path) as f:
        return json.load(f)
 
 
# ---------------------------------------------------------------
# One predict function for all three models
# ---------------------------------------------------------------
def predict(model_name, model, X):
    """X is an array of windows with shape (n, 60, 5). Returns n predictions."""
    if model_name == "lstm":
        with torch.no_grad():
            predictions = model(torch.tensor(X, dtype=torch.float32))
        return predictions.numpy()
 
    elif model_name == "xgboost":
        X_flat = common.flatten_windows(X)
        return model.predict(xgb.DMatrix(X_flat))
 
    elif model_name == "ridge":
        X_flat = common.flatten_windows(X)
        return predict_ridge(model, X_flat)
 
    else:
        raise ValueError("Unknown model name: " + model_name)
