# train_xgboost.py
# Trains the XGBoost model and saves it to xgboost_model.json
 
import xgboost as xgb
 
import common
import models
 
 
def main():
    print("Downloading 60 days of EUR/USD 15-minute data...")
    df = common.download_data("60d")
    print("Downloaded", len(df), "bars")
 
    features = common.build_features(df)
    X, y = common.make_windows(features)
    X_train, y_train, X_val, y_val = common.split_train_val(X, y)
    print("Training windows:", len(X_train), " Validation windows:", len(X_val))
 
    train_data = xgb.DMatrix(common.flatten_windows(X_train), label=y_train)
    val_data = xgb.DMatrix(common.flatten_windows(X_val), label=y_val)
 
    settings = {
        "max_depth": 4,              # shallow trees to avoid overfitting
        "eta": 0.05,                 # learning rate
        "subsample": 0.8,            # each tree sees 80% of the rows
        "colsample_bytree": 0.8,     # each tree sees 80% of the features
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }
 
    booster = xgb.train(
        settings,
        train_data,
        num_boost_round=200,
        evals=[(val_data, "validation")],
        early_stopping_rounds=15,    # stop if validation MAE does not improve for 15 rounds
        verbose_eval=False,
    )
 
    # Keep only the trees up to the best round
    print("Best round:", booster.best_iteration)
    booster = booster[: booster.best_iteration + 1]
 
    train_metrics = common.all_metrics(y_train, models.predict("xgboost", booster, X_train))
    val_metrics = common.all_metrics(y_val, models.predict("xgboost", booster, X_val))
    baseline_metrics = common.all_metrics(y_val, common.naive_predictions(X_val))
 
    print("\n--- Results ---")
    common.print_metrics("Train", train_metrics)
    common.print_metrics("Validation", val_metrics)
    print(f"Baseline   MAE={baseline_metrics['mae']:.4f}  RMSE={baseline_metrics['rmse']:.4f}")
 
    gap = val_metrics["mae"] - train_metrics["mae"]
    print(f"Train/validation MAE gap: {gap:.4f}")
    if gap > 0.05:
        print("WARNING: big gap, the model may be overfitting.")
 
    booster.save_model("xgboost_model.json")
    print("Saved model to xgboost_model.json")
 
    common.save_report("report_xgboost.json", {
        "train": train_metrics,
        "validation": val_metrics,
        "baseline": baseline_metrics,
    })
 
 
if __name__ == "__main__":
    main()
