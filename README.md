# EUR/USD Forecasting Dashboard: LSTM vs XGBoost vs Ridge
 
A web dashboard that predicts the direction of the next 15-minute EUR/USD candle using three different machine learning models, trained and tested on the same data so they can be compared fairly.
 
**Live demo:** _add your Render link here after deployment_
 
![Dashboard screenshot](docs/screenshot.png)
 
## What it does
 
- Downloads real EUR/USD 15-minute prices from Yahoo Finance (no API key needed)
- Looks at the last 60 bars (15 hours) and predicts the next bar's close
- Compares three models:
  - **LSTM** (PyTorch), a neural network that reads the bars in order
  - **XGBoost**, gradient-boosted decision trees
  - **Ridge regression**, a linear model written by hand in NumPy
- Compares every model against a naive baseline ("next close = last close")
- Shows live predictions on a Flask dashboard that updates every 60 seconds
 
## Results
 
Validation results on the last 15% of the data (chronological split, no shuffling):
 
| Model | Direction accuracy | MAE | RMSE |
|---|---|---|---|
| LSTM | _fill in_ | _fill in_ | _fill in_ |
| XGBoost | _fill in_ | _fill in_ | _fill in_ |
| Ridge | _fill in_ | _fill in_ | _fill in_ |
| Naive baseline | n/a | _fill in_ | _fill in_ |
 
Numbers come from `report_lstm.json`, `report_xgboost.json` and `report_ridge.json`. Yahoo only serves the last 60 days of 15-minute data, so results change a little each time the models are retrained. Judge direction accuracy against 50% (random guessing).
 
## How it works
 
1. **Features:** each bar's open, high, low and close are scaled to 0-1 using the day's running high and low (reset at midnight, using only bars seen so far, so no future data leaks in). A time-of-day value is added, giving 5 features per bar.
2. **Windows:** 60 bars of features form one input. The target is the next bar's scaled close.
3. **Split:** first 85% for training, last 15% for validation, in time order.
4. **Models:** the LSTM reads the 60×5 window directly. XGBoost and Ridge get it flattened into 300 numbers.
5. **Dashboard:** every 60 seconds the app downloads the last 7 days, re-checks each model on the last 100 bars, and forecasts the next bar.
 
## Project structure
 
```
common.py          download data, build features, windows, metrics
models.py          LSTM, XGBoost, Ridge and one predict() function
train_ridge.py     trains Ridge   -> ridge_model.json
train_xgboost.py   trains XGBoost -> xgboost_model.json
train_lstm.py      trains LSTM    -> lstm_model.pth
predict.py         runs all trained models on the latest data
app.py             Flask dashboard, background scheduler, rate limit
templates/         dashboard HTML page
tests/             unit tests (pytest)
Dockerfile         container for deployment
render.yaml        Render deployment settings
```
 
## Run it on your computer
 
Requires Python 3.14.
 
```bash
git clone https://github.com/YOUR-USERNAME/eurusd-forecast.git
cd eurusd-forecast
py -3.14 -m venv venv           # Windows
source venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
 
python train_ridge.py
python train_xgboost.py
python train_lstm.py
 
python app.py
```
 
Open http://127.0.0.1:5001
 
Run the tests with `pytest tests -v`.
 
## Security notes
 
- `/refresh` is rate-limited to once per 60 seconds per visitor (Flask-Limiter). `ProxyFix` makes sure the real visitor IP is used behind Render's proxy.
- Model files are loaded without integrity checks. Adding SHA-256 verification is a planned improvement.
- Market data comes from Yahoo Finance, a third party the app cannot verify.
 
## Limitations
 
- Not a trading system: no spreads, fees, position sizing or risk management.
- Predictions are on scaled prices (0-1 within each day), not real pips.
- One train/validation split on one 60-day window, not a full statistical benchmark.
 
## Credits
 
Based on the idea of [EURUSD-lstm-dashboard](https://github.com/DebajyotiRay/EURUSD-lstm-dashboard) by Debajyoti Ray, rebuilt from scratch.
 
## Disclaimer
 
Educational project only. Not financial advice.
