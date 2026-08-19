from pathlib import Path
import json
import textwrap

try:
    import nbformat as nbf
except ModuleNotFoundError:
    class _V4:
        @staticmethod
        def new_code_cell(source):
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }

        @staticmethod
        def new_markdown_cell(source):
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": source.splitlines(keepends=True),
            }

        @staticmethod
        def new_notebook():
            return {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }

    class _NBFormatFallback:
        v4 = _V4()

        @staticmethod
        def write(nb, path):
            Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")

    nbf = _NBFormatFallback()


NOTEBOOK = Path("Week_9_Predictive_Maintenance_Lab.ipynb")


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nb["cells"] = [
    md(
        """
        # Week 9 Lab: Predictive Intelligence & Industrial Trust

        This notebook completes the predictive maintenance lab end to end:

        - frames the accuracy trap for rare failures
        - builds a leakage-aware preprocessing pipeline
        - compares baseline, SMOTE, class weighting, neural, and clustering approaches
        - evaluates with precision-recall metrics
        - uses SHAP for global and local explanations
        - produces operational recommendations for maintenance teams
        """
    ),
    md(
        """
        ## 1. Setup

        The lab uses a synthetic industrial dataset that mirrors the structure of AI4I/C-MAPSS-style maintenance data: many normal logs and a small number of failures.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys
        import json
        import warnings

        # Notebook environment guard:
        # Put workspace-local packages first. This fixes kernels that can see
        # matplotlib but lost numpy in the user-site package directory.
        NOTEBOOK_DEPS = Path.cwd() / "notebook_deps"
        if NOTEBOOK_DEPS.exists():
            sys.path.insert(0, str(NOTEBOOK_DEPS))

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
        """
    ),
    md("## 2. Create a Rare-Failure Industrial Dataset"),
    code(
        """
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
                        4.0 * vibration + 2.6 * process_temp + 1.8 * torque + 1.3 * pressure
                        + 1.2 * tool_wear + 0.8 * noise - 1.0 * rotational_speed + rng.normal(0, 0.25)
                    )
                    rows.append({
                        "asset_id": asset_id, "cycle": cycle, "vibration": vibration,
                        "process_temp": process_temp, "torque": torque,
                        "rotational_speed": rotational_speed, "tool_wear": tool_wear,
                        "pressure": pressure, "noise": noise, "risk_score": risk_score,
                    })
            df = pd.DataFrame(rows)
            df["failure"] = (df["risk_score"] >= df["risk_score"].quantile(0.98)).astype(int)
            return df.drop(columns=["risk_score"])

        df = make_predictive_maintenance_data()
        df.to_csv(OUT / "synthetic_predictive_maintenance_data.csv", index=False)
        df.head()
        """
    ),
    code(
        """
        print(f"Rows: {len(df):,}")
        print(f"Assets: {df['asset_id'].nunique()}")
        print(f"Failure rate: {df['failure'].mean():.2%}")
        df["failure"].value_counts(normalize=True).rename("share")
        """
    ),
    md("## 3. Leakage-Aware Split and Min-Max Scaling"),
    code(
        """
        train = df[df["asset_id"] < 84].copy()
        valid = df[(df["asset_id"] >= 84) & (df["asset_id"] < 102)].copy()
        test = df[df["asset_id"] >= 102].copy()

        feature_cols = [c for c in df.columns if c != "failure"]
        sensor_cols = [c for c in feature_cols if c != "asset_id"]

        X_train_raw, y_train = train[feature_cols], train["failure"]
        X_test_raw, y_test = test[feature_cols], test["failure"]

        scaler = MinMaxScaler()
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()
        X_train[sensor_cols] = scaler.fit_transform(X_train_raw[sensor_cols])
        X_test[sensor_cols] = scaler.transform(X_test_raw[sensor_cols])

        pd.DataFrame({
            "split": ["train", "validation", "test"],
            "rows": [len(train), len(valid), len(test)],
            "failure_rate": [train["failure"].mean(), valid["failure"].mean(), test["failure"].mean()],
        })
        """
    ),
    md("## 4. Baseline Random Forest and the Accuracy Trap"),
    code(
        """
        def model_report(model, X_train, y_train, X_test, y_test, name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            return {
                "name": name,
                "model": model,
                "pred": pred,
                "proba": proba,
                "report": report,
                "average_precision": average_precision_score(y_test, proba),
                "confusion_matrix": confusion_matrix(y_test, pred),
            }

        baseline = model_report(
            RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE),
            X_train, y_train, X_test, y_test, "Baseline Random Forest"
        )
        print(classification_report(y_test, baseline["pred"], zero_division=0))
        """
    ),
    md("## 5. Imbalanced Data Strategies: SMOTE and Class Weighting"),
    code(
        """
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

        rf_smote = model_report(
            RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE),
            X_smote, y_smote, X_test, y_test, "Random Forest + SMOTE"
        )

        neg, pos = np.bincount(y_train)
        xgb_weighted = model_report(
            XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=RANDOM_STATE, scale_pos_weight=neg / pos,
            ),
            X_train, y_train, X_test, y_test, "XGBoost + Class Weighting"
        )
        """
    ),
    md("## 6. Advanced Benchmarking: Neural Proxy and K-Means Segments"),
    code(
        """
        neural_proxy = model_report(
            Pipeline([("mlp", MLPClassifier(hidden_layer_sizes=(48, 16), max_iter=600, random_state=RANDOM_STATE))]),
            X_train, y_train, X_test, y_test, "Sequence Proxy Neural Net"
        )

        kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
        train_clusters = kmeans.fit_predict(X_train[sensor_cols])
        cluster_risk = pd.DataFrame({"cluster": train_clusters, "failure": y_train.to_numpy()}).groupby("cluster")["failure"].mean()
        test_clusters = kmeans.predict(X_test[sensor_cols])
        unsupervised_scores = np.array([cluster_risk.to_dict()[c] for c in test_clusters])
        kmeans_ap = average_precision_score(y_test, unsupervised_scores)
        cluster_risk
        """
    ),
    md("## 7. Precision-Recall Curves"),
    code(
        """
        results = [baseline, rf_smote, xgb_weighted, neural_proxy]
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
        plt.show()
        """
    ),
    code(
        """
        metric_rows = []
        for result in results:
            failure = result["report"].get("1", {})
            metric_rows.append({
                "model": result["name"],
                "accuracy": result["report"]["accuracy"],
                "failure_precision": failure.get("precision", 0),
                "failure_recall": failure.get("recall", 0),
                "failure_f1": failure.get("f1-score", 0),
                "average_precision": result["average_precision"],
            })
        metric_rows.append({
            "model": "K-Means High-Risk Segments",
            "accuracy": np.nan,
            "failure_precision": np.nan,
            "failure_recall": np.nan,
            "failure_f1": np.nan,
            "average_precision": kmeans_ap,
        })
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(OUT / "model_metrics.csv", index=False)
        metrics
        """
    ),
    md("## 8. SHAP Trust Builder"),
    code(
        """
        explainer = shap.TreeExplainer(xgb_weighted["model"])
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap.summary_plot(shap_values, X_test, show=False, max_display=8)
        plt.tight_layout()
        plt.savefig(OUT / "shap_global_summary.png", dpi=180, bbox_inches="tight")
        plt.show()
        """
    ),
    code(
        """
        high_risk_idx = int(np.argmax(xgb_weighted["proba"]))
        target_row = X_test.iloc[[high_risk_idx]]
        shap.force_plot(
            explainer.expected_value,
            shap_values[high_risk_idx],
            target_row,
            matplotlib=True,
            show=False,
        )
        plt.savefig(OUT / "shap_local_force_plot.png", dpi=180, bbox_inches="tight")
        plt.show()

        local_contrib = pd.DataFrame({
            "feature": X_test.columns,
            "value": target_row.iloc[0].to_numpy(),
            "shap_contribution": shap_values[high_risk_idx],
        }).assign(abs_contribution=lambda d: d["shap_contribution"].abs())
        local_contrib.sort_values("abs_contribution", ascending=False).head(8)
        """
    ),
    md("## 9. Operational Translation"),
    code(
        """
        print(
            f"Plain-language alert: Asset {int(test.iloc[high_risk_idx]['asset_id'])} at cycle "
            f"{int(test.iloc[high_risk_idx]['cycle'])} was flagged because the model saw a pattern "
            f"similar to prior failure conditions. The top SHAP factors show which sensor readings "
            f"pushed the alert upward, giving the technician a starting point for inspection."
        )
        """
    ),
    md(
        """
        ## 10. Recommended Response

        Treat high-risk predictions as triage alerts, not automatic shutdown orders. Review the SHAP explanation, inspect the physical system tied to the top drivers, open a maintenance ticket when repeated alerts appear, and feed technician outcomes back into the model improvement loop.
        """
    ),
]

nbf.write(nb, NOTEBOOK)
print(NOTEBOOK.resolve())
