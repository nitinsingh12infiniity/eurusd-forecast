# app.py
# The web dashboard. Run with:  python app.py   then open http://127.0.0.1:5001
 
import os
import secrets
from datetime import datetime
 
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
 
import predict
 
REFRESH_SECONDS = int(os.environ.get("REFRESH_SECONDS", "60"))
MODEL_NAMES = ["lstm", "xgboost", "ridge"]
 
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(16))
 
# On Render the app sits behind a proxy. ProxyFix makes Flask see the
# visitor's real IP address, so the rate limit below works per visitor.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
 
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")
 
# The latest predictions are kept here
latest = {"results": {}, "updated_at": None, "error": None}
 
 
def refresh_predictions():
    """Run all models and store the results. Called every REFRESH_SECONDS."""
    try:
        latest["results"] = predict.run_all_models()
        latest["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        latest["error"] = None
        print("Predictions updated at", latest["updated_at"])
    except Exception as error:
        latest["error"] = str(error)
        print("Prediction failed:", error)
 
 
def make_charts(result):
    """Build the two Plotly charts and return them as HTML text."""
    times = result["timestamps"]
    actual = result["true_prices"]
    predicted = result["predicted_prices"]
 
    # Chart 1: actual vs predicted price
    price_chart = go.Figure()
    price_chart.add_trace(go.Scatter(x=times, y=actual, mode="lines", name="Actual"))
    price_chart.add_trace(go.Scatter(x=times, y=predicted, mode="lines", name="Predicted",
                                     line=dict(dash="dot")))
    price_chart.update_layout(title="Actual vs predicted close (normalised 0-1)",
                              template="plotly_dark", height=350,
                              margin=dict(l=40, r=20, t=50, b=40))
 
    # Chart 2: change from one bar to the next
    actual_change = []
    predicted_change = []
    for i in range(1, len(actual)):
        actual_change.append(actual[i] - actual[i - 1])
        predicted_change.append(predicted[i] - predicted[i - 1])
 
    change_chart = go.Figure()
    change_chart.add_trace(go.Bar(x=times[1:], y=actual_change, name="Actual"))
    change_chart.add_trace(go.Bar(x=times[1:], y=predicted_change, name="Predicted"))
    change_chart.update_layout(title="Bar-by-bar change", barmode="group",
                               template="plotly_dark", height=350,
                               margin=dict(l=40, r=20, t=50, b=40))
 
    price_html = price_chart.to_html(full_html=False, include_plotlyjs="cdn")
    change_html = change_chart.to_html(full_html=False, include_plotlyjs=False)
    return price_html, change_html
 
 
@app.route("/")
def index():
    results = latest["results"]
    selected = request.args.get("model", "lstm")
 
    # If the chosen model has no results, pick the first one that does
    if selected not in results:
        selected = None
        for name in MODEL_NAMES:
            if name in results:
                selected = name
                break
 
    page = {
        "model_names": MODEL_NAMES,
        "available": results.keys(),
        "selected": selected,
        "updated_at": latest["updated_at"],
        "error": latest["error"],
        "refresh_seconds": REFRESH_SECONDS,
    }
 
    if selected is not None:
        result = results[selected]
        price_html, change_html = make_charts(result)
        page["going_up"] = result["predicted_change"] > 0
        page["change"] = f"{result['predicted_change']:+.5f}"
        page["mae"] = f"{result['mae']:.4f}"
        page["rmse"] = f"{result['rmse']:.4f}"
        page["direction_acc"] = f"{result['directional_accuracy'] * 100:.1f}%"
        page["price_chart"] = price_html
        page["change_chart"] = change_html
 
    return render_template("index.html", **page)
 
 
@app.route("/health")
def health():
    return jsonify(status="ok")
 
 
@app.route("/refresh", methods=["POST"])
@limiter.limit(f"1 per {REFRESH_SECONDS} seconds")
def refresh():
    refresh_predictions()
    model = request.form.get("model", "lstm")
    return redirect(url_for("index", model=model))
 
 
@app.errorhandler(429)
def too_many_requests(error):
    flash(f"Refresh is limited to once every {REFRESH_SECONDS} seconds.")
    return redirect(url_for("index"))
 
 
def start_scheduler():
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(refresh_predictions, "interval", seconds=REFRESH_SECONDS,
                      next_run_time=datetime.now())
    scheduler.start()
 
 
# Tests set DISABLE_SCHEDULER=1 so they don't download real data
if os.environ.get("DISABLE_SCHEDULER") != "1":
    start_scheduler()
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"Open http://127.0.0.1:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=False)
