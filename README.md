# Week 9 Lab: Predictive Intelligence & Industrial Trust

This folder contains a completed Week 9 predictive maintenance lab.

## Files

- `complete_week9_lab.py` generates the dataset, trains the models, evaluates imbalance strategies, runs SHAP, and writes the explainer.
- `outputs/synthetic_predictive_maintenance_data.csv` is the generated industrial sensor dataset.
- `outputs/model_metrics.csv` contains the benchmark table.
- `outputs/precision_recall_curves.png` compares models using PR curves.
- `outputs/baseline_confusion_matrix.png` and `outputs/xgboost_confusion_matrix.png` show baseline vs. weighted model behavior.
- `outputs/shap_global_summary.png` shows global feature importance.
- `outputs/shap_local_force_plot.png` explains one high-risk prediction.
- `outputs/model_explainer.md` is the plain-language stakeholder deliverable.
- `Week_9_Predictive_Maintenance_Lab.ipynb` is a notebook version of the full lab.
- `app.py` serves the proposed operational dashboard and chat-style explanation interface.

## Run

The notebook includes a small environment guard that prepends `notebook_deps` to `sys.path`. This folder contains a local NumPy install to fix kernels that can import `matplotlib` from user packages but report `ModuleNotFoundError: No module named 'numpy'`.

Use the bundled Codex Python runtime:

```powershell
& 'C:\Users\Zeni_Bets\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' complete_week9_lab.py
```

Create the notebook:

```powershell
& 'C:\Users\Zeni_Bets\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' create_week9_notebook.py
```

Run the web app:

```powershell
& 'C:\Users\Zeni_Bets\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' app.py
```

Then open `http://127.0.0.1:8059`.
