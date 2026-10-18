# tests/test_project.py
# Run with:  pytest tests -v
# These tests use made-up prices, so they need no internet and no trained models.
 
import os
os.environ["DISABLE_SCHEDULER"] = "1"   # stop app.py from downloading real data
 
import numpy as np
import pandas as pd
 
import common
import models
import predict
 
 
def make_fake_prices(n=400):
    """Random-walk prices that look a bit like EUR/USD."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-01-01", periods=n, freq="15min")
    price = 1.10 + np.cumsum(rng.standard_normal(n) * 0.0002)
    spread = np.abs(rng.standard_normal(n)) * 0.0001 + 0.000001
    return pd.DataFrame({
        "open": price + rng.uniform(-0.9, 0.9, n) * spread,
        "high": price + spread,
        "low": price - spread,
        "close": price + rng.uniform(-0.9, 0.9, n) * spread,
    }, index=times)
 
 
def test_features_have_5_columns():
    features = common.build_features(make_fake_prices())
    assert list(features.columns) == common.FEATURE_NAMES
 
 
def test_features_are_between_0_and_1():
    features = common.build_features(make_fake_prices())
    assert features.min().min() >= 0
    assert features.max().max() <= 1
 
 
def test_flat_prices_do_not_crash():
    times = pd.date_range("2026-01-01", periods=10, freq="15min")
    flat = pd.DataFrame({"open": [1.1] * 10, "high": [1.1] * 10,
                         "low": [1.1] * 10, "close": [1.1] * 10}, index=times)
    features = common.build_features(flat)
    assert not features.isnull().values.any()
 
 
def test_window_shapes():
    features = common.build_features(make_fake_prices(200))
    X, y = common.make_windows(features)
    assert X.shape == (140, 60, 5)
    assert y.shape == (140,)
 
 
def test_directional_accuracy_perfect():
    true_values = [1.0, 1.1, 1.05, 1.2]
    predicted = [1.0, 1.2, 0.9, 1.3]     # every predicted direction is right
    assert common.directional_accuracy(true_values, predicted) == 1.0
 
 
def test_mae_and_rmse():
    assert common.mae([0, 0, 0], [1, 1, 1]) == 1.0
    assert common.rmse([0, 0], [3, 3]) == 3.0
 
 
def test_ridge_matches_formula():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)
    model = models.fit_ridge(X, y, alpha=2.0)
 
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    expected = np.linalg.inv(Xc.T @ Xc + 2.0 * np.eye(5)) @ Xc.T @ yc
    assert np.allclose(model["weights"], expected)
 
 
def test_lstm_output_shape():
    model = models.LSTMModel()
    X = np.zeros((3, 60, 5), dtype=np.float32)
    assert models.predict("lstm", model, X).shape == (3,)
 
 
def test_backtest_gives_100_predictions():
    features = common.build_features(make_fake_prices())
    X, y = common.make_windows(features)
    ridge = models.fit_ridge(common.flatten_windows(X), y, alpha=10.0)
    result = predict.run_backtest("ridge", ridge, features)
    assert len(result["predicted_prices"]) == 100
    assert len(result["true_prices"]) == 100
 
 
def test_health_page():
    import app
    client = app.app.test_client()
    assert client.get("/health").get_json() == {"status": "ok"}
 
 
def test_refresh_is_rate_limited(monkeypatch):
    import app
    monkeypatch.setattr(app, "refresh_predictions", lambda: None)   # no real download
    app.limiter.reset()
    client = app.app.test_client()
    client.post("/refresh")
    second = client.post("/refresh", follow_redirects=True)
    assert "limited to once every" in second.get_data(as_text=True)
