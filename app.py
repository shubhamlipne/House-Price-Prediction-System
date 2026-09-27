"""
backend/app.py
----------------
Flask REST API that serves house-price predictions from the
trained Linear Regression model (model/model.pkl).

Endpoints:
    GET  /health           -> service status
    GET  /model-info        -> model coefficients / metrics
    POST /predict            -> {"area": <float>, "rooms": <int>} -> {"predicted_price": <float>}
"""

import json
import os

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

app = Flask(__name__)
CORS(app)

model = None
metrics = {}


def load_model():
    global model, metrics
    model = joblib.load(MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify(
        {
            "model_type": "LinearRegression",
            "features": ["area", "rooms"],
            "metrics": metrics,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    if "area" not in data or "rooms" not in data:
        return jsonify({"error": "Request body must include 'area' and 'rooms'"}), 400

    try:
        area = float(data["area"])
        rooms = float(data["rooms"])
    except (TypeError, ValueError):
        return jsonify({"error": "'area' and 'rooms' must be numeric"}), 400

    if area <= 0 or rooms <= 0:
        return jsonify({"error": "'area' and 'rooms' must be positive numbers"}), 400

    predicted_price = model.predict([[area, rooms]])[0]

    return jsonify(
        {
            "area": area,
            "rooms": rooms,
            "predicted_price": round(float(predicted_price), 2),
        }
    )


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    load_model()
