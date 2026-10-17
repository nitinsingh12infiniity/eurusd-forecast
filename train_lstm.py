# train_lstm.py
# Trains the LSTM model and saves it to lstm_model.pth
 
import copy
import random
 
import torch
import torch.nn as nn
 
import common
import models
 
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 5          # stop if validation loss does not improve for 5 epochs
 
 
def main():
    print("Downloading 60 days of EUR/USD 15-minute data...")
    df = common.download_data("60d")
    print("Downloaded", len(df), "bars")
 
    features = common.build_features(df)
    X, y = common.make_windows(features)
    X_train, y_train, X_val, y_val = common.split_train_val(X, y)
    print("Training windows:", len(X_train), " Validation windows:", len(X_val))
 
    # Same random numbers every run, so results can be repeated
    random.seed(42)
    torch.manual_seed(42)
 
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
 
    model = models.LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
 
    best_val_loss = None
    best_weights = None
    epochs_without_improvement = 0
 
    for epoch in range(1, EPOCHS + 1):
        # ---- training: go through the training windows in small batches ----
        model.train()
        order = list(range(len(X_train)))
        random.shuffle(order)
 
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            x_batch = X_train_t[batch]
            y_batch = y_train_t[batch]
 
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stops very large updates
            optimizer.step()
 
            total_loss += loss.item() * len(batch)
 
        train_loss = total_loss / len(order)
 
        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(X_val_t), y_val_t).item()
 
        print(f"Epoch {epoch:2d}  train loss={train_loss:.5f}  validation loss={val_loss:.5f}")
 
        # ---- early stopping ----
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("No improvement for", PATIENCE, "epochs. Stopping early.")
                break
 
    # Go back to the best weights we saw
    model.load_state_dict(best_weights)
    model.eval()
 
    train_metrics = common.all_metrics(y_train, models.predict("lstm", model, X_train))
    val_metrics = common.all_metrics(y_val, models.predict("lstm", model, X_val))
    baseline_metrics = common.all_metrics(y_val, common.naive_predictions(X_val))
 
    print("\n--- Results ---")
    common.print_metrics("Train", train_metrics)
    common.print_metrics("Validation", val_metrics)
    print(f"Baseline   MAE={baseline_metrics['mae']:.4f}  RMSE={baseline_metrics['rmse']:.4f}")
 
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Saved model to lstm_model.pth")
 
    common.save_report("report_lstm.json", {
        "train": train_metrics,
        "validation": val_metrics,
        "baseline": baseline_metrics,
    })
 
 
if __name__ == "__main__":
    main()
