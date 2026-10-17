# train_ridge.py
# Trains the Ridge regression model and saves it to ridge_model.json
 
import common
import models
 
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]   # regularisation strengths to try
 
 
def main():
    print("Downloading 60 days of EUR/USD 15-minute data...")
    df = common.download_data("60d")
    print("Downloaded", len(df), "bars")
 
    features = common.build_features(df)
    X, y = common.make_windows(features)
    X_train, y_train, X_val, y_val = common.split_train_val(X, y)
    print("Training windows:", len(X_train), " Validation windows:", len(X_val))
 
    X_train_flat = common.flatten_windows(X_train)
    X_val_flat = common.flatten_windows(X_val)
 
    # Try every alpha and keep the one with the lowest validation MAE
    best_model = None
    best_val_mae = None
    for alpha in ALPHAS:
        model = models.fit_ridge(X_train_flat, y_train, alpha)
        val_predictions = models.predict_ridge(model, X_val_flat)
        val_mae = common.mae(y_val, val_predictions)
        print(f"alpha={alpha:<8} validation MAE={val_mae:.4f}")
 
        if best_val_mae is None or val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
 
    print("Best alpha:", best_model["alpha"])
 
    train_metrics = common.all_metrics(y_train, models.predict_ridge(best_model, X_train_flat))
    val_metrics = common.all_metrics(y_val, models.predict_ridge(best_model, X_val_flat))
    baseline_metrics = common.all_metrics(y_val, common.naive_predictions(X_val))
 
    print("\n--- Results ---")
    common.print_metrics("Train", train_metrics)
    common.print_metrics("Validation", val_metrics)
    print(f"Baseline   MAE={baseline_metrics['mae']:.4f}  RMSE={baseline_metrics['rmse']:.4f}  "
          f"(direction not meaningful for a no-change guess)")
 
    models.save_ridge(best_model, "ridge_model.json")
    print("Saved model to ridge_model.json")
 
    common.save_report("report_ridge.json", {
        "best_alpha": best_model["alpha"],
        "train": train_metrics,
        "validation": val_metrics,
        "baseline": baseline_metrics,
    })
 
 
if __name__ == "__main__":
    main()
