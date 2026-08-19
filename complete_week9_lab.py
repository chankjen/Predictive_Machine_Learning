from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import shap


warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def make_predictive_maintenance_data(n_assets=120, cycles_per_asset=50):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for asset_id in range(n_assets):
        asset_health = rng.normal(0, 0.35)
        drift_start = rng.integers(26, 44)
        for cycle in range(1, cycles_per_asset + 1):
            age = cycle / cycles_per_asset
            degradation = max(0, cycle - drift_start) / (cycles_per_asset - drift_start + 1)
            vibration = rng.normal(0.23 + 0.18 * age + 0.55 * degradation + asset_health * 0.08, 0.05)
            process_temp = rng.normal(0.39 + 0.13 * age + 0.34 * degradation, 0.05)
            torque = rng.normal(0.50 + 0.08 * age + 0.20 * degradation, 0.07)
            rotational_speed = rng.normal(0.62 - 0.12 * degradation, 0.06)
            tool_wear = rng.normal(0.20 + 0.45 * age + 0.12 * degradation, 0.06)
            pressure = rng.normal(0.42 + 0.07 * age + 0.25 * degradation, 0.05)
            noise = rng.normal(0.28 + 0.13 * age + 0.38 * degradation, 0.05)

            risk_score = (
                4.0 * vibration
                + 2.6 * process_temp
                + 1.8 * torque
                + 1.3 * pressure
                + 1.2 * tool_wear
                + 0.8 * noise
                - 1.0 * rotational_speed
                + rng.normal(0, 0.25)
            )
            rows.append(
                {
                    "asset_id": asset_id,
                    "cycle": cycle,
                    "vibration": vibration,
                    "process_temp": process_temp,
                    "torque": torque,
                    "rotational_speed": rotational_speed,
                    "tool_wear": tool_wear,
                    "pressure": pressure,
                    "noise": noise,
                    "risk_score": risk_score,
                }
            )

    df = pd.DataFrame(rows)
    threshold = df["risk_score"].quantile(0.98)
    df["failure"] = (df["risk_score"] >= threshold).astype(int)
    df = df.drop(columns=["risk_score"])
    return df


def leakage_aware_split(df):
    train_assets = df.loc[df["asset_id"] < 84, "asset_id"].unique()
    valid_assets = df.loc[(df["asset_id"] >= 84) & (df["asset_id"] < 102), "asset_id"].unique()
    test_assets = df.loc[df["asset_id"] >= 102, "asset_id"].unique()
    train = df[df["asset_id"].isin(train_assets)].copy()
    valid = df[df["asset_id"].isin(valid_assets)].copy()
    test = df[df["asset_id"].isin(test_assets)].copy()
    return train, valid, test


def model_report(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    ap = average_precision_score(y_test, proba)
    return {
        "name": name,
        "model": model,
        "pred": pred,
        "proba": proba,
        "report": report,
        "average_precision": ap,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def save_pr_curves(results, y_test):
    plt.figure(figsize=(8, 5))
    for result in results:
        PrecisionRecallDisplay.from_predictions(
            y_test,
            result["proba"],
            name=f"{result['name']} AP={result['average_precision']:.3f}",
            ax=plt.gca(),
        )
    plt.title("Precision-Recall Curves for Rare Failure Detection")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curves.png", dpi=180)
    plt.close()


def save_confusion_matrix(y_test, pred, title, filename):
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def markdown_table(df):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    df = make_predictive_maintenance_data()
    df.to_csv(OUT / "synthetic_predictive_maintenance_data.csv", index=False)

    train, valid, test = leakage_aware_split(df)
    feature_cols = [c for c in df.columns if c not in ["failure"]]
    sensor_cols = [c for c in feature_cols if c != "asset_id"]

    X_train_raw, y_train = train[feature_cols], train["failure"]
    X_test_raw, y_test = test[feature_cols], test["failure"]

    scaler = MinMaxScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[sensor_cols] = scaler.fit_transform(X_train_raw[sensor_cols])
    X_test[sensor_cols] = scaler.transform(X_test_raw[sensor_cols])

    baseline = model_report(
        RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE),
        X_train,
        y_train,
        X_test,
        y_test,
        "Baseline Random Forest",
    )

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    rf_smote = model_report(
        RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, class_weight=None),
        X_smote,
        y_smote,
        X_test,
        y_test,
        "Random Forest + SMOTE",
    )

    neg, pos = np.bincount(y_train)
    xgb_weighted = model_report(
        XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=neg / pos,
        ),
        X_train,
        y_train,
        X_test,
        y_test,
        "XGBoost + Class Weighting",
    )

    mlp_sequence_proxy = model_report(
        Pipeline(
            [
                ("mlp", MLPClassifier(hidden_layer_sizes=(48, 16), max_iter=600, random_state=RANDOM_STATE)),
            ]
        ),
        X_train,
        y_train,
        X_test,
        y_test,
        "Sequence Proxy Neural Net",
    )

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
    clusters = kmeans.fit_predict(X_train[sensor_cols])
    cluster_risk = pd.DataFrame({"cluster": clusters, "failure": y_train.to_numpy()}).groupby("cluster")["failure"].mean()
    test_clusters = kmeans.predict(X_test[sensor_cols])
    cluster_lookup = cluster_risk.to_dict()
    unsupervised_scores = np.array([cluster_lookup[c] for c in test_clusters])
    precision, recall, thresholds = precision_recall_curve(y_test, unsupervised_scores)
    kmeans_ap = average_precision_score(y_test, unsupervised_scores)

    results = [baseline, rf_smote, xgb_weighted, mlp_sequence_proxy]
    save_pr_curves(results, y_test)
    save_confusion_matrix(y_test, baseline["pred"], "Baseline Random Forest", "baseline_confusion_matrix.png")
    save_confusion_matrix(y_test, xgb_weighted["pred"], "XGBoost + Class Weighting", "xgboost_confusion_matrix.png")

    best = xgb_weighted
    explainer = shap.TreeExplainer(best["model"])
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(OUT / "shap_global_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    high_risk_idx = int(np.argmax(best["proba"]))
    target_row = X_test.iloc[[high_risk_idx]]
    shap.force_plot(
        explainer.expected_value,
        shap_values[high_risk_idx],
        target_row,
        matplotlib=True,
        show=False,
    )
    plt.savefig(OUT / "shap_local_force_plot.png", dpi=180, bbox_inches="tight")
    plt.close()

    local_contrib = pd.DataFrame(
        {
            "feature": X_test.columns,
            "value": target_row.iloc[0].to_numpy(),
            "shap_contribution": shap_values[high_risk_idx],
        }
    ).assign(abs_contribution=lambda d: d["shap_contribution"].abs())
    local_contrib = local_contrib.sort_values("abs_contribution", ascending=False).head(5)

    summary_rows = []
    for result in results:
        failure = result["report"].get("1", {})
        normal = result["report"].get("0", {})
        summary_rows.append(
            {
                "model": result["name"],
                "accuracy": result["report"]["accuracy"],
                "failure_precision": failure.get("precision", 0),
                "failure_recall": failure.get("recall", 0),
                "failure_f1": failure.get("f1-score", 0),
                "average_precision": result["average_precision"],
                "confusion_matrix": result["confusion_matrix"],
            }
        )
    summary_rows.append(
        {
            "model": "K-Means High-Risk Segments",
            "accuracy": None,
            "failure_precision": None,
            "failure_recall": None,
            "failure_f1": None,
            "average_precision": kmeans_ap,
            "confusion_matrix": None,
        }
    )
    metrics = pd.DataFrame(summary_rows)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)

    report_payload = {
        "dataset_shape": df.shape,
        "failure_rate": float(df["failure"].mean()),
        "train_failure_rate": float(y_train.mean()),
        "test_failure_rate": float(y_test.mean()),
        "metrics": summary_rows,
        "kmeans_cluster_failure_rates": {str(k): float(v) for k, v in cluster_risk.to_dict().items()},
        "high_risk_asset": int(test.iloc[high_risk_idx]["asset_id"]),
        "high_risk_cycle": int(test.iloc[high_risk_idx]["cycle"]),
        "high_risk_probability": float(best["proba"][high_risk_idx]),
        "top_local_shap_factors": local_contrib.to_dict(orient="records"),
    }
    (OUT / "lab_results.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    explainer_text = f"""# Week 9 Model Explainer: Predictive Maintenance

## What the model is doing
This lab simulates an industrial maintenance setting where failures are rare: only {df['failure'].mean():.1%} of equipment logs are failure events. That imbalance creates the accuracy trap. A model can look excellent by mostly saying "normal," while still missing the alerts maintenance teams care about most.

Think of the model like an experienced mechanic listening to an engine. It does not rely on one sound. It weighs vibration, heat, torque, pressure, wear, speed, and operating age together to decide whether a machine deserves attention.

## How the data was prepared
The dataset contains {len(df):,} sensor records across {df['asset_id'].nunique()} assets. Sensor readings were normalized with Min-Max Scaling so that large-unit measurements did not overpower smaller-unit signals. The split was leakage-aware: earlier asset groups were used for training, and later unseen asset groups were held out for testing. That keeps future maintenance behavior from sneaking into the training process.

## What changed after handling imbalance
The baseline Random Forest reached {baseline['report']['accuracy']:.1%} accuracy, but the important question is failure recall: how many actual failures were caught. Its failure recall was {baseline['report']['1']['recall']:.1%}. After using class weighting in XGBoost, failure recall was {xgb_weighted['report']['1']['recall']:.1%}, with an average precision score of {xgb_weighted['average_precision']:.3f}. The PR curve is the better view here because failures are rare and false confidence from accuracy is easy.

SMOTE created synthetic failure examples so the Random Forest could see more minority-class patterns. XGBoost class weighting used a different strategy: it made missed failures more costly during training. In operations, that is often the more direct business framing because missing a true failure can mean downtime, safety risk, or expensive emergency repairs.

## What the model found
The SHAP global summary shows which signals most often pushed predictions toward failure across the test fleet. The strongest drivers were typically vibration, process temperature, torque, pressure, and tool wear. That makes practical sense: rising vibration and heat are common early warning signs for mechanical stress.

For the highest-risk test asset, asset {int(test.iloc[high_risk_idx]['asset_id'])} at cycle {int(test.iloc[high_risk_idx]['cycle'])}, the model estimated a {best['proba'][high_risk_idx]:.1%} failure probability. The local SHAP explanation shows the specific reasons for that alert. The largest contributors were:

{markdown_table(local_contrib[['feature', 'value', 'shap_contribution']])}

## Recommended maintenance action
Treat high-risk alerts as triage signals, not automatic shutdown orders. The recommended workflow is:

1. Review the SHAP explanation for the top two or three drivers.
2. Check the physical system tied to those drivers, such as bearing condition for vibration or cooling/lubrication for temperature.
3. Create a maintenance ticket when the same asset shows repeated high-risk readings or when the SHAP drivers match known failure modes.
4. Track technician feedback so future models learn which alerts were useful.

## Limitations
This lab uses a synthetic dataset designed to mirror an AI4I or C-MAPSS style failure problem. A production model would need real sensor history, confirmed failure labels, asset metadata, and review by maintenance experts before it could be used for operational decisions.
"""
    (OUT / "model_explainer.md").write_text(explainer_text, encoding="utf-8")

    print(metrics.round(4).to_string(index=False))
    print(f"\nWrote outputs to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
