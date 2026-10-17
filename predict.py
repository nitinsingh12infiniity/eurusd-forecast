# predict.py
# Loads the saved models, downloads the latest data, and makes predictions.
# The dashboard (app.py) calls run_all_models() every 60 seconds.
 
import os
 
import numpy as np
 
import common
import models
 
MODEL_FILES = {
    "lstm": "lstm_model.pth",
    "xgboost": "xgboost_model.json",
    "ridge": "ridge_model.json",
}
 
loaded_models = {}   # models are loaded once and kept here
 
 
def get_model(model_name):
    """Load a model from its file the first time, then reuse it."""
    if model_name in loaded_models:
        return loaded_models[model_name]
 
    path = MODEL_FILES[model_name]
    if not os.path.exists(path):
        return None   # this model has not been trained yet
 
    if model_name == "lstm":
        model = models.load_lstm(path)
    elif model_name == "xgboost":
        model = models.load_xgboost(path)
    else:
        model = models.load_ridge(path)
 
    loaded_models[model_name] = model
    print("Loaded", model_name, "from", path)
    return model
 
 
def run_backtest(model_name, model, features):
    """Predict each of the last 100 bars (backtest), then predict the next bar."""
    values = features.values.astype(np.float32)
    needed = common.WINDOW_SIZE + common.BACKTEST_STEPS
    if len(values) < needed:
        raise ValueError(f"Not enough data: have {len(values)} bars, need {needed}")
 
    windows = []
    true_prices = []
    timestamps = []
    for step in range(common.BACKTEST_STEPS, 0, -1):
        end = len(values) - step                                   # the bar we try to predict
        windows.append(values[end - common.WINDOW_SIZE:end])       # the 60 bars before it
        true_prices.append(float(values[end][common.CLOSE_COLUMN]))
        timestamps.append(str(features.index[end]))
 
    predictions = models.predict(model_name, model, np.array(windows))
    predicted_prices = []
    for p in predictions:
        predicted_prices.append(float(p))
 
    # Forecast for the bar that has not happened yet: use the latest 60 bars
    last_window = values[-common.WINDOW_SIZE:]
    next_prediction = float(models.predict(model_name, model, np.array([last_window]))[0])
    predicted_change = next_prediction - true_prices[-1]
 
    metrics = common.all_metrics(true_prices, predicted_prices)
 
    return {
        "timestamps": timestamps,
        "true_prices": true_prices,
        "predicted_prices": predicted_prices,
        "next_prediction": next_prediction,
        "predicted_change": predicted_change,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "directional_accuracy": metrics["directional_accuracy"],
    }
 
 
def run_all_models():
    """Download the last 7 days and run every trained model."""
    df = common.download_data("7d")
    features = common.build_features(df)
 
    results = {}
    for model_name in ["lstm", "xgboost", "ridge"]:
        model = get_model(model_name)
        if model is None:
            print("Skipping", model_name, "- not trained yet")
            continue
        results[model_name] = run_backtest(model_name, model, features)
    return results
 
 
# Run "python predict.py" to test predictions without the dashboard
if __name__ == "__main__":
    all_results = run_all_models()
    for name in all_results:
        r = all_results[name]
        if r["predicted_change"] > 0:
            direction = "UP"
        else:
            direction = "DOWN"
        print(f"{name:<8} next bar: {direction:<5} backtest direction accuracy: "
              f"{r['directional_accuracy'] * 100:.1f}%")
