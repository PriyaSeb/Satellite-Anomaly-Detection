# model

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from joblib import dump, load

CSV_PATH = Path("../satellite_data/orbital_elements/Haiyang-2A.csv")
TARGET_COL = "inclination"
WINDOW_SIZE = 10  # number of lag observations to use
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp_unix"] = df["timestamp"].astype("int64") // 1_000_000_000
    return df[["timestamp", "timestamp_unix", TARGET_COL]]


def series_to_supervised(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # Create lag features
    for lag in range(1, window + 1):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)
    supervised = df.dropna().reset_index(drop=True)
    return supervised


# Helper functions

def evaluate_model(model, X_scaled, y_true, y_scaler):

    pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae, y_pred


def sliding_window_forecast(model, history, steps, x_scaler, y_scaler):
    history = list(history)
    preds = []
    for _ in range(steps):
        window_feats = np.array(history[-WINDOW_SIZE:])  
        X_feat = window_feats.reshape(1, -1)
        X_scaled = x_scaler.transform(X_feat)
        y_scaled = model.predict(X_scaled)
        y_pred = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
        preds.append(y_pred)
        history.append(y_pred)  # slide window
    return np.array(preds)

# Main routine



def main():
    df = load_data(CSV_PATH)
    supervised_df = series_to_supervised(df, WINDOW_SIZE)


    supervised_df = supervised_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    feature_cols = [f"lag_{lag}" for lag in range(WINDOW_SIZE, 0, -1)]
    X = supervised_df[feature_cols].values
    y = supervised_df[TARGET_COL].values

    split_idx = int(len(supervised_df) * TRAIN_FRACTION)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]


    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    dump(x_scaler, "x_scaler_inclination.joblib")
    dump(y_scaler, "y_scaler_inclination.joblib")



    # Model training 
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        reg_lambda=5,
        # reg_alpha=0.1,
    )
    model.fit(X_train_scaled, y_train_scaled)

    model.save_model("test_regr_h2aman_inclination.json")

  
    # Prediction and evaluation
    rmse, mae, y_pred = evaluate_model(model, X_test_scaled, y_test, y_scaler)
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test  MAE: {mae:.6f}")



    num_steps = 40
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    train_last_idx = int(len(df_sorted) * TRAIN_FRACTION) - 20
    last_window = df_sorted[TARGET_COL].iloc[train_last_idx - WINDOW_SIZE:train_last_idx].values
    future_preds = sliding_window_forecast(
        model,
        last_window,
        steps=num_steps,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    # print("\nSliding-window forecast (20 steps):")
    # print(future_preds)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df_sorted["timestamp"].iloc[train_last_idx:train_last_idx + num_steps], future_preds, label="Predicted", alpha=0.7)
    plt.plot(df_sorted["timestamp"].iloc[train_last_idx:train_last_idx + num_steps], df_sorted[TARGET_COL].iloc[train_last_idx:train_last_idx + num_steps], label="Actual", alpha=0.7)
    plt.title(f"XGBoost forecast with {WINDOW_SIZE}-step window (predict next {num_steps} values)")
    plt.xlabel("Timestamp")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main() 