"""
DRDO IVHM — Dashboard Backend (Flask)
Loads the report JSON from Day 13 and serves it to the frontend.
Also exposes a /predict endpoint for live single-sample inference.

Run:  python dashboard/app.py
Open: http://localhost:5000
"""

import json
import random
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

REPORT_PATH = Path(__file__).parent.parent / "reports" / "ivhm_report.json"


def load_report() -> dict:
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    # Fallback demo data if report not generated yet
    return _demo_data()


def _demo_data() -> dict:
    """Fallback demo payload so dashboard works before Day 13 is run."""
    subsystems = {
        "Airframe":      {"avg_health": 92, "min_health": 78, "rul_hours": 420, "status": "Healthy",  "n_anomalies": 12},
        "Hydraulics":    {"avg_health": 76, "min_health": 62, "rul_hours": 280, "status": "Healthy",  "n_anomalies": 28},
        "Fuel System":   {"avg_health": 88, "min_health": 74, "rul_hours": 380, "status": "Healthy",  "n_anomalies": 15},
        "Li-Ion Battery":{"avg_health": 65, "min_health": 44, "rul_hours": 120, "status": "Degraded", "n_anomalies": 42},
        "Electronics":   {"avg_health": 91, "min_health": 82, "rul_hours": 410, "status": "Healthy",  "n_anomalies": 9},
        "Actuators":     {"avg_health": 83, "min_health": 68, "rul_hours": 310, "status": "Healthy",  "n_anomalies": 22},
        "ECS":           {"avg_health": 28, "min_health": 12, "rul_hours": 38,  "status": "Faulty",   "n_anomalies": 94},
        "Electrical":    {"avg_health": 87, "min_health": 75, "rul_hours": 350, "status": "Healthy",  "n_anomalies": 18},
    }
    return {
        "overall_health":   round(np.mean([v["avg_health"] for v in subsystems.values()]), 1),
        "total_anomalies":  sum(v["n_anomalies"] for v in subsystems.values()),
        "n_critical":       1,
        "critical_systems": ["ECS"],
        "subsystems":       subsystems,
        "generated_at":     "demo",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/report")
def report():
    """Full report JSON — fetched on page load."""
    return jsonify(load_report())


@app.route("/api/live")
def live():
    """
    Simulates live sensor data by adding small random perturbations.
    In production this would read from real UAV telemetry.
    """
    data = load_report()
    for name, sub in data["subsystems"].items():
        # Flicker health score ±2
        sub["avg_health"] = round(
            max(0, min(100, sub["avg_health"] + random.uniform(-2, 2))), 1)
    data["overall_health"] = round(
        np.mean([s["avg_health"] for s in data["subsystems"].values()]), 1)
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Single-sample inference endpoint.
    Body: {"subsystem": "ECS", "sensors": {"cabin_temp_c": 45, ...}}
    Returns: {"health_label": 2, "health_status": "Faulty", "confidence": 0.92}
    """
    try:
        body      = request.json
        subsystem = body.get("subsystem", "")
        sensors   = body.get("sensors", {})

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import joblib
        model_path = Path(__file__).parent.parent / "models" / f"{subsystem.replace(' ','_')}.pkl"

        if not model_path.exists():
            return jsonify({"error": "Model not found. Run day3 first."}), 404

        pkg   = joblib.load(model_path)
        feats = pkg["features"]
        X_in  = np.array([[sensors.get(f, 0.0) for f in feats]])
        X_sc  = pkg["scaler"].transform(X_in)
        pred  = int(pkg["model"].predict(X_sc)[0])
        conf  = float(pkg["model"].predict_proba(X_sc).max())

        return jsonify({
            "health_label":  pred,
            "health_status": ["Healthy","Degraded","Faulty"][pred],
            "confidence":    round(conf, 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀  DRDO IVHM Dashboard running at http://localhost:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
