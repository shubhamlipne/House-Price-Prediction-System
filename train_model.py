"""
train_model.py
----------------
Trains a Linear Regression model to predict house prices from
area (sq ft) and number of rooms, then saves the trained model
to model/model.pkl for the Flask backend to serve.

Dataset: data/house_price.csv (columns: area, rooms, price)
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/house_price.csv"
MODEL_PATH = "model/model.pkl"
METRICS_PATH = "model/metrics.json"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"area", "rooms", "price"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}, found {list(df.columns)}")
    return df


def train():
    df = load_data(DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows and columns {list(df.columns)}")

    X = df[["area", "rooms"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel evaluation on hold-out test set:")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R^2  : {r2:.4f}")
    print(f"\nLearned coefficients: area={model.coef_[0]:.2f}, rooms={model.coef_[1]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2_score": round(float(r2), 4),
        "coef_area": round(float(model.coef_[0]), 4),
        "coef_rooms": round(float(model.coef_[1]), 4),
        "intercept": round(float(model.intercept_), 4),
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    train()
