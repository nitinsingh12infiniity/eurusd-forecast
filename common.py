# common.py
# Code shared by every other file:
#   1. downloading EUR/USD prices
#   2. turning prices into features
#   3. cutting features into 60-bar windows
#   4. measuring how good the predictions are
 
import json
 
import numpy as np
import pandas as pd
import yfinance as yf
 
SYMBOL = "EURUSD=X"
WINDOW_SIZE = 60        # the models look at the last 60 bars (60 x 15 min = 15 hours)
BACKTEST_STEPS = 100    # the dashboard re-checks the last 100 bars
FEATURE_NAMES = ["time_token", "norm_open", "norm_high", "norm_low", "norm_close"]
CLOSE_COLUMN = 4        # position of "norm_close" in FEATURE_NAMES
 
 
# ---------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------
def download_data(period):
    """Download 15-minute EUR/USD candles. period is a text like "7d" or "60d"."""
    df = yf.download(SYMBOL, period=period, interval="15m", progress=False, auto_adjust=True)
 
    if df is None or len(df) == 0:
        raise RuntimeError("Yahoo Finance returned no data. Check your internet connection.")
 
    # New yfinance versions give two-level column names like ("Close", "EURUSD=X").
    # We keep only the first part: "Close".
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
 
    # Make column names lowercase: "Close" -> "close"
    lowercase_names = []
    for name in df.columns:
        lowercase_names.append(name.lower())
    df.columns = lowercase_names
 
    # Remove the timezone information from the times
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
 
    df = df[["open", "high", "low", "close"]].dropna()
    return df
 
 
# ---------------------------------------------------------------
# 2. Build features
# ---------------------------------------------------------------
def build_features(df):
    """Turn raw prices into 5 features per bar.
 
    Every price is scaled between 0 and 1 using the day's high and low SO FAR.
    The high and low start again at midnight. We only use bars that have
    already happened, so the model never sees future information.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
 
    rows = []
    current_day = None
    day_high = 0.0
    day_low = 0.0
 
    for i in range(len(df)):
        timestamp = df.index[i]
        day = timestamp.date()
 
        # New day: start the running high and low again
        if day != current_day:
            current_day = day
            day_high = highs[i]
            day_low = lows[i]
        else:
            if highs[i] > day_high:
                day_high = highs[i]
            if lows[i] < day_low:
                day_low = lows[i]
 
        day_range = day_high - day_low
        if day_range == 0:
            # Avoid dividing by zero when the price has not moved yet
            norm_open = 0.0
            norm_high = 0.0
            norm_low = 0.0
            norm_close = 0.0
        else:
            norm_open = (opens[i] - day_low) / day_range
            norm_high = (highs[i] - day_low) / day_range
            norm_low = (lows[i] - day_low) / day_range
            norm_close = (closes[i] - day_low) / day_range
 
        # Time of day as a number between 0 (midnight) and 1 (next midnight)
        seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        time_token = seconds / 86400
 
        rows.append([time_token, float(norm_open), float(norm_high), float(norm_low), float(norm_close)])
 
    features = pd.DataFrame(rows, columns=FEATURE_NAMES, index=df.index)
    return features
 
 
# ---------------------------------------------------------------
# 3. Windows and splitting
# ---------------------------------------------------------------
def make_windows(features):
    """Cut the features into windows.
 
    X[i] = 60 bars of features (shape 60 x 5)
    y[i] = the normalised close of the bar right after that window
    """
    values = features.values
    X = []
    y = []
    for i in range(WINDOW_SIZE, len(values)):
        X.append(values[i - WINDOW_SIZE:i])
        y.append(values[i][CLOSE_COLUMN])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
 
 
def split_train_val(X, y, val_fraction=0.15):
    """First 85% for training, last 15% for validation. No shuffling: time order matters."""
    split = int(len(X) * (1 - val_fraction))
    return X[:split], y[:split], X[split:], y[split:]
 
 
def flatten_windows(X):
    """XGBoost and Ridge need flat rows: (n, 60, 5) becomes (n, 300)."""
    return X.reshape(len(X), -1)
 
 
def naive_predictions(X):
    """Baseline: "the next close will be the same as the last close"."""
    predictions = []
    for window in X:
        predictions.append(window[-1][CLOSE_COLUMN])
    return np.array(predictions)
 
 
# ---------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------
def mae(true_values, predicted_values):
    """Mean absolute error: average size of the mistake."""
    total = 0.0
    for i in range(len(true_values)):
        total += abs(predicted_values[i] - true_values[i])
    return float(total / len(true_values))
 
 
def rmse(true_values, predicted_values):
    """Root mean squared error: like MAE, but big mistakes count more."""
    total = 0.0
    for i in range(len(true_values)):
        error = predicted_values[i] - true_values[i]
        total += error * error
    return float((total / len(true_values)) ** 0.5)
 
 
def directional_accuracy(true_values, predicted_values):
    """How often the model guessed the direction (up or down) correctly.
 
    For each bar we compare against the PREVIOUS true close:
      real move      = true[i]      - true[i-1]
      predicted move = predicted[i] - true[i-1]
    Bars where the price did not move at all are skipped.
    """
    correct = 0
    total = 0
    for i in range(1, len(true_values)):
        real_move = true_values[i] - true_values[i - 1]
        predicted_move = predicted_values[i] - true_values[i - 1]
 
        if real_move == 0:
            continue
 
        total += 1
        if real_move > 0 and predicted_move > 0:
            correct += 1
        elif real_move < 0 and predicted_move < 0:
            correct += 1
 
    if total == 0:
        return 0.0
    return correct / total
 
 
def all_metrics(true_values, predicted_values):
    return {
        "mae": mae(true_values, predicted_values),
        "rmse": rmse(true_values, predicted_values),
        "directional_accuracy": directional_accuracy(true_values, predicted_values),
    }
 
 
def print_metrics(name, metrics):
    print(f"{name:<10} MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"Direction={metrics['directional_accuracy'] * 100:.1f}%")
 
 
def save_report(filename, report):
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved report to", filename)
