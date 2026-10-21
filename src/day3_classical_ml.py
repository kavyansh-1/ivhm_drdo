"""
DRDO IVHM — Day 3: Classical ML Models
Trains RandomForest + XGBoost + GradientBoosting per subsystem.
UAV-aware train/test split (no data leakage across UAVs).
5-fold stratified CV + confusion matrix + feature importance + ROC curves.
Saves best model per subsystem as .pkl for Day 4 + dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import joblib, logging, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

from config import (
    SUBSYSTEMS, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR,
    LABELS, PALETTE, COLORS, PLOT_STYLE, GOLD,
    RANDOM_SEED, CV_FOLDS, TEST_SIZE
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)
matplotlib.rcParams.update(PLOT_STYLE)

# ── Model definitions ─────────────────────────────────────────
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_SEED, verbosity=0
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=RANDOM_SEED
    ),
}


# ════════════════════════════════════════════════════════════
# DATA LOADING + UAV-AWARE SPLIT
# ════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"health_label", "health_status", "uav_id", "timestep",
            "subsystem", "rul", "capacity_ah", "life_fraction", "fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns
            if c not in skip]


def load_and_split(name: str):
    """
    UAV-aware train/test split — entire UAVs go to either train or test.
    This prevents data leakage: adjacent timesteps from the same UAV
    are highly correlated, so mixing them inflates scores artificially.
    """
    path = DATA_PROCESSED / f"{name.replace(' ', '_')}.csv"
    if not path.exists():
        log.warning("Missing processed file for %s — run day2_eda.py", name)
        return None

    df        = pd.read_csv(path)
    feat_cols = get_feature_cols(df)
    X         = df[feat_cols].fillna(0).values
    y         = df["health_label"].values
    uav_ids   = df["uav_id"].values if "uav_id" in df.columns else np.zeros(len(df))

    unique_uavs = np.unique(uav_ids)
    n_test      = max(1, int(len(unique_uavs) * TEST_SIZE))
    test_uavs   = np.random.choice(unique_uavs, n_test, replace=False)

    train_mask  = ~np.isin(uav_ids, test_uavs)
    test_mask   =  np.isin(uav_ids, test_uavs)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test  = scaler.transform(X[test_mask])
    y_train = y[train_mask]
    y_test  = y[test_mask]

    return X_train, X_test, y_train, y_test, scaler, feat_cols


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════

def plot_confusion(y_true, y_pred, model_name: str, subsys: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS.values(),
                yticklabels=LABELS.values(), ax=ax,
                linewidths=0.5, linecolor="#1E3A5F")
    ax.set_title(f"{subsys} | {model_name}", color="white", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    out = FIGURES_DIR / f"day3_cm_{subsys.replace(' ','_')}_{model_name}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_prob, model_name: str, subsys: str):
    """One-vs-Rest ROC curve for all 3 health classes."""
    y_bin  = label_binarize(y_true, classes=[0, 1, 2])
    colors = PALETTE

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (cls_name, color) in enumerate(zip(LABELS.values(), colors)):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc          = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_title(f"{subsys} | {model_name} — ROC Curves", color="white", fontsize=10)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True)
    plt.tight_layout()
    out = FIGURES_DIR / f"day3_roc_{subsys.replace(' ','_')}_{model_name}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feat_cols: list, subsys: str, top_n: int = 15):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_cols[i] for i in idx], imp[idx],
            color=GOLD, alpha=0.85)
    ax.set_title(f"{subsys} — Top {top_n} Feature Importances",
                 color="white", fontsize=11)
    ax.grid(True, axis="x")
    plt.tight_layout()
    out = FIGURES_DIR / f"day3_importance_{subsys.replace(' ','_')}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


def plot_model_comparison(all_results: dict):
    """Heatmap — F1 score per subsystem × model."""
    subsystems = list(all_results.keys())
    model_names = list(MODELS.keys())

    matrix = np.array([
        [all_results[s][m]["test_f1"] for m in model_names]
        for s in subsystems
    ])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=model_names,
                yticklabels=subsystems,
                ax=ax, vmin=0.7, vmax=1.0,
                linewidths=0.5, linecolor="#060E1A",
                annot_kws={"size": 9})
    ax.set_title("Model F1 Score — All Subsystems",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model"); ax.set_ylabel("Subsystem")
    plt.tight_layout()
    out = FIGURES_DIR / "day3_model_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


# ════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════

def train_subsystem(name: str) -> dict | None:
    result = load_and_split(name)
    if result is None:
        return None

    X_train, X_test, y_train, y_test, scaler, feat_cols = result
    log.info("\n── %s  (train: %d, test: %d) ──", name, len(y_train), len(y_test))

    subsys_results = {}
    best_f1, best_name, best_model = 0.0, "", None

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for model_name, model in MODELS.items():
        # Cross-validation
        cv_f1 = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_weighted", n_jobs=-1)
        # Final fit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)
                  if hasattr(model, "predict_proba") else None)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        icon = "🥇" if model_name == "XGBoost" else "  "
        log.info("  %s %-20s  CV-F1: %.3f±%.3f  Acc: %.3f  F1: %.3f",
                 icon, model_name, cv_f1.mean(), cv_f1.std(), acc, f1)

        plot_confusion(y_test, y_pred, model_name, name)
        if y_prob is not None:
            plot_roc(y_test, y_prob, model_name, name)

        subsys_results[model_name] = {
            "cv_f1_mean": round(cv_f1.mean(), 4),
            "cv_f1_std":  round(cv_f1.std(),  4),
            "test_acc":   round(acc,  4),
            "test_f1":    round(f1,   4),
            "test_prec":  round(prec, 4),
            "test_rec":   round(rec,  4),
        }

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, model_name, model

    # Save best model for this subsystem
    model_path = MODELS_DIR / f"{name.replace(' ', '_')}.pkl"
    joblib.dump({"model": best_model, "scaler": scaler,
                 "features": feat_cols, "best_model_name": best_name},
                model_path)
    log.info("  ✅ Best: %-20s F1=%.4f → saved %s",
             best_name, best_f1, model_path.name)

    # Feature importance from best model (if available)
    plot_feature_importance(best_model, feat_cols, name)

    # Print full classification report for best model
    best_model.fit(X_train, y_train)
    log.info("\n%s", classification_report(y_test, best_model.predict(X_test),
                                            target_names=list(LABELS.values())))

    return subsys_results


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run() -> dict:
    log.info("=" * 62)
    log.info("  DRDO IVHM — Day 3: Classical ML Training")
    log.info("  Models: RandomForest | XGBoost | GradientBoosting")
    log.info("  Eval:   5-Fold CV + Hold-out test + ROC + CM")
    log.info("=" * 62)

    all_results = {}
    for name in SUBSYSTEMS:
        res = train_subsystem(name)
        if res:
            all_results[name] = res

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Subsystem':<20} {'Model':<22} {'CV-F1':>8} {'±':>6} {'Test-F1':>9}")
    print(f"  {'-'*65}")
    for sname, sres in all_results.items():
        best = max(sres, key=lambda m: sres[m]["test_f1"])
        r    = sres[best]
        print(f"  {sname:<20} {best:<22} {r['cv_f1_mean']:>8.3f}"
              f" {r['cv_f1_std']:>6.3f} {r['test_f1']:>9.3f}")

    # ── Comparison heatmap ────────────────────────────────────
    if len(all_results) >= 2:
        plot_model_comparison(all_results)

    log.info("\n✅ Day 3 complete! Models → models/  |  Plots → reports/figures/")
    return all_results


if __name__ == "__main__":
    run()
