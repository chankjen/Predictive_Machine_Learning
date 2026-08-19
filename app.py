from pathlib import Path
import json

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
app = Flask(__name__)


def load_payload():
    return json.loads((OUT / "lab_results.json").read_text(encoding="utf-8"))


def load_metrics():
    df = pd.read_csv(OUT / "model_metrics.csv")
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def asset_risk_table():
    df = pd.read_csv(OUT / "synthetic_predictive_maintenance_data.csv")
    latest = df.sort_values(["asset_id", "cycle"]).groupby("asset_id").tail(1).copy()
    latest = latest[latest["asset_id"] >= 102].copy()
    sensor_cols = ["vibration", "process_temp", "torque", "pressure", "tool_wear", "noise"]
    risk_raw = (
        4.0 * latest["vibration"]
        + 2.6 * latest["process_temp"]
        + 1.8 * latest["torque"]
        + 1.3 * latest["pressure"]
        + 1.2 * latest["tool_wear"]
        + 0.8 * latest["noise"]
        - 1.0 * latest["rotational_speed"]
    )
    latest["risk_probability"] = (risk_raw.rank(pct=True) * 0.92).clip(0.02, 0.94)
    latest["risk_band"] = pd.cut(
        latest["risk_probability"],
        bins=[0, 0.55, 0.78, 1],
        labels=["Monitor", "Inspect Soon", "High Risk"],
        include_lowest=True,
    ).astype(str)
    latest["primary_driver"] = latest[sensor_cols].idxmax(axis=1)
    cols = ["asset_id", "cycle", "risk_probability", "risk_band", "primary_driver"] + sensor_cols
    return latest[cols].sort_values("risk_probability", ascending=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT, name)


@app.route("/api/summary")
def summary():
    payload = load_payload()
    fleet = asset_risk_table()
    return jsonify(
        {
            "failureRate": payload["failure_rate"],
            "highRiskAsset": payload["high_risk_asset"],
            "highRiskCycle": payload["high_risk_cycle"],
            "highRiskProbability": payload["high_risk_probability"],
            "topFactors": payload["top_local_shap_factors"],
            "metrics": load_metrics(),
            "fleet": fleet.head(20).to_dict(orient="records"),
        }
    )


@app.route("/api/explain/<int:asset_id>")
def explain(asset_id):
    fleet = asset_risk_table()
    row = fleet[fleet["asset_id"] == asset_id]
    if row.empty:
        return jsonify({"error": "Asset not found"}), 404
    item = row.iloc[0].to_dict()
    drivers = sorted(
        [
            ("vibration", item["vibration"]),
            ("process_temp", item["process_temp"]),
            ("torque", item["torque"]),
            ("pressure", item["pressure"]),
            ("tool_wear", item["tool_wear"]),
            ("noise", item["noise"]),
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    narrative = (
        f"Asset {asset_id} is in the {item['risk_band']} band with an estimated "
        f"{item['risk_probability']:.1%} risk. The strongest current signals are "
        f"{drivers[0][0]}, {drivers[1][0]}, and {drivers[2][0]}. "
        "Use this as a triage alert: inspect the related components before deciding on downtime."
    )
    return jsonify({"asset": item, "drivers": drivers, "narrative": narrative})


@app.route("/api/chat", methods=["POST"])
def chat():
    message = (request.json or {}).get("message", "").lower()
    fleet = asset_risk_table()
    payload = load_payload()
    top = fleet.iloc[0]

    if "why" in message or "flag" in message:
        answer = (
            f"The highest-risk asset is {int(top['asset_id'])}. It is flagged because its current sensor pattern "
            f"is strongest around {top['primary_driver']}, with an estimated {top['risk_probability']:.1%} risk. "
            "That is similar to how a mechanic hears a changed engine tone and checks the likely source first."
        )
    elif "recall" in message or "accuracy" in message:
        metrics = load_metrics()
        xgb = next(m for m in metrics if m["model"] == "XGBoost + Class Weighting")
        answer = (
            f"Accuracy can be misleading because failures are only {payload['failure_rate']:.1%} of records. "
            f"The weighted XGBoost model is more useful operationally because it reached "
            f"{xgb['failure_recall']:.1%} failure recall while keeping precision at {xgb['failure_precision']:.1%}."
        )
    elif "action" in message or "maintenance" in message or "do" in message:
        answer = (
            "Recommended action: inspect the top SHAP or sensor drivers first, check for repeat alerts on the same asset, "
            "then open a maintenance ticket if the physical inspection confirms the pattern."
        )
    else:
        answer = (
            "I can explain why an asset was flagged, summarize recall vs. accuracy, or recommend a maintenance response. "
            "Try asking: Why is the highest-risk asset flagged?"
        )
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8059, debug=False)
