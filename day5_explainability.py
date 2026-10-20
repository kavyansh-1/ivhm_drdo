"""
DRDO IVHM — Day 5: Explainability with SHAP
Loads saved models, computes SHAP values per subsystem.
Critical for DRDO submission — defense systems need interpretable AI.
Outputs: beeswarm plots, waterfall plots, feature importance rankings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shap
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")

from config import (SUBSYSTEMS, DATA_PROCESSED, MODELS_DIR,
                    FIGURES_DIR, LABELS, PLOT_STYLE, GOLD, RANDOM_SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
matplotlib.rcParams.update(PLOT_STYLE)
np.random.seed(RANDOM_SEED)


def get_feature_cols(df):
    skip = {"health_label","health_status","uav_id","timestep",
            "subsystem","rul","capacity_ah","life_fraction","fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in skip]


def load_model_and_data(name: str):
    model_path = MODELS_DIR / f"{name.replace(' ','_')}.pkl"
    data_path  = DATA_PROCESSED / f"{name.replace(' ','_')}.csv"
    if not model_path.exists() or not data_path.exists():
        log.warning("Missing files for %s — run days 1-3 first", name)
        return None, None, None

    pkg       = joblib.load(model_path)
    model     = pkg["model"]
    scaler    = pkg["scaler"]
    feat_cols = pkg["features"]

    df     = pd.read_csv(data_path)
    avail  = [c for c in feat_cols if c in df.columns]
    X_raw  = df[avail].fillna(0).values
    X      = scaler.transform(X_raw)
    y      = df["health_label"].values
    return model, X, y, avail


def plot_shap_beeswarm(shap_values, X, feat_cols, name: str):
    """
    Beeswarm — shows feature impact direction + magnitude across all samples.
    Red = high feature value, Blue = low.
    """
    plt.figure(figsize=(10, 6))
    # For multi-class, use class 2 (Faulty) — most important for maintenance
    sv = shap_values[2] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(sv, X, feature_names=feat_cols,
                      show=False, plot_size=(10, 6),
                      color_bar_label="Feature value")
    plt.title(f"{name} — SHAP Beeswarm (Faulty Class)",
              color="white", fontsize=11, pad=10)
    out = FIGURES_DIR / f"day5_shap_beeswarm_{name.replace(' ','_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#060E1A")
    plt.close()
    log.info("  Saved beeswarm → %s", out.name)


def plot_shap_bar(shap_values, feat_cols, name: str):
    """Bar chart of mean |SHAP| across all classes."""
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    idx = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_cols[i] for i in idx], mean_abs[idx], color=GOLD, alpha=0.85)
    ax.set_title(f"{name} — Mean |SHAP| Feature Importance",
                 color="white", fontsize=11)
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(True, axis="x")
    plt.tight_layout()
    out = FIGURES_DIR / f"day5_shap_bar_{name.replace(' ','_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("  Saved bar chart → %s", out.name)


def run():
    log.info("="*62)
    log.info("  DRDO IVHM — Day 5: SHAP Explainability")
    log.info("="*62)

    global_importance = {}

    for name in SUBSYSTEMS:
        result = load_model_and_data(name)
        if result[0] is None:
            continue
        model, X, y, feat_cols = result

        log.info("\n── %s ──", name)

        # Subsample for speed (SHAP is O(n))
        n_samples = min(500, len(X))
        idx       = np.random.choice(len(X), n_samples, replace=False)
        X_sample  = X[idx]

        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            plot_shap_beeswarm(shap_values, X_sample, feat_cols, name)
            plot_shap_bar(shap_values, feat_cols, name)

            # Store top features for global summary
            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(0) for sv in shap_values], 0)
            else:
                mean_abs = np.abs(shap_values).mean(0)

            top = {feat_cols[i]: round(float(mean_abs[i]),5)
                   for i in np.argsort(mean_abs)[-5:][::-1]}
            global_importance[name] = top
            log.info("  Top 5 features: %s", list(top.keys()))

        except Exception as e:
            log.warning("  SHAP failed for %s: %s", name, e)

    # Global summary CSV
    rows = []
    for sub, feats in global_importance.items():
        for feat, val in feats.items():
            rows.append({"subsystem": sub, "feature": feat, "mean_abs_shap": val})

    if rows:
        out = FIGURES_DIR / "day5_global_shap_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info("\n✅ Day 5 complete! Global SHAP summary → %s", out)


if __name__ == "__main__":
    run()
