from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request


app = Flask(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Milestone 3 model
MODEL_PATH = PROJECT_ROOT / "milestone_3" / "models" / "best_model.pkl"

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    MODEL_ERROR = str(e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "UrbanClap ML Prediction API",
        "status": "running"
    })


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({
            "status": "error",
            "message": MODEL_ERROR
        }), 500

    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model could not be loaded",
            "details": MODEL_ERROR
        }), 500

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body is empty"
        }), 400

    # Accept either:
    # {"features": [....]}
    # or
    # {"features": [[....]]}
    features = data.get("features")

    if features is None:
        return jsonify({
            "error": "Missing 'features' field"
        }), 400

    try:
        features = np.asarray(features, dtype=float)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction = model.predict(features)

        response = {
            "prediction": prediction.tolist()
        }

        # Include probability if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            response["probability"] = probabilities.tolist()

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False,
        use_reloader=False
    )