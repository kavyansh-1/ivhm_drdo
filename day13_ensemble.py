"""
DRDO IVHM — Day 13: Final Ensemble + Report Generation
Combines all models: Classical ML (Day 3) + LSTM (Day 8) + Autoencoder (Day 10).
Weighted voting ensemble → final unified health score.
Generates the summary report JSON consumed by the dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib, json, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch

from config import (SUBSYSTEMS, DATA_PROCESSED, MODELS_DIR,
                    FIGURES_DIR, LABELS, PALETTE, PLOT_STYLE, GOLD, RANDOM_SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
matplotlib.rcParams.update(PLOT_STYLE)
np.random.seed(RANDOM_SEED)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LSTM_DIR = MODELS_DIR / "lstm"
AE_DIR   = MODELS_DIR / "autoencoder"
REPORT   = Path("reports")
REPORT.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# LOAD HELPERS
# ════════════════════════════════════════════════════════════

def get_feature_cols(df):
    skip = {"health_label","health_status","uav_id","timestep",
            "subsystem","rul","capacity_ah","life_fraction","fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in skip]


def load_classical(name: str):
    p = MODELS_DIR / f"{name.replace(' ','_')}.pkl"
    return joblib.load(p) if p.exists() else None


def load_lstm_meta(name: str):
    p = LSTM_DIR / f"{name.replace(' ','_')}_meta.pkl"
    return joblib.load(p) if p.exists() else None


def load_ae_meta(name: str):
    p = AE_DIR / f"{name.replace(' ','_')}_meta.pkl"
    return joblib.load(p) if p.exists() else None


# ════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR
# ════════════════════════════════════════════════════════════

def predict_classical(pkg: dict, df: pd.DataFrame) -> np.ndarray | None:
    """Returns class probabilities (N, 3)."""
    feat_cols = [c for c in pkg["features"] if c in df.columns]
    if not feat_cols:
        return None
    X = pkg["scaler"].transform(df[feat_cols].fillna(0))
    if hasattr(pkg["model"], "predict_proba"):
        return pkg["model"].predict_proba(X)
    return None


def predict_lstm(name: str, df: pd.DataFrame,
                 meta: dict) -> np.ndarray | None:
    """Returns class probabilities (N-SEQ_LEN, 3) from LSTM."""
    try:
        from day8_lstm import IVHMLSTMNet
        model_path = LSTM_DIR / f"{name.replace(' ','_')}_best.pt"
        if not model_path.exists():
            return None

        feat_cols  = [c for c in meta["features"] if c in df.columns]
        seq_len    = meta["seq_len"]
        n_features = meta["n_features"]
        X = meta["scaler"].transform(df[feat_cols].fillna(0))

        model = IVHMLSTMNet(n_features).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        windows = np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])
        if len(windows) == 0:
            return None

        with torch.no_grad():
            Xt = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
            logits, _ = model(Xt)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    except Exception as e:
        log.warning("  LSTM predict failed: %s", e)
        return None


def anomaly_to_probs(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert AE reconstruction error → pseudo-probability vector (3 classes).
    Low error  → P(Healthy)=0.9
    Above thresh → P(Faulty)=0.8
    """
    norm = np.clip(errors / (threshold * 1.5), 0, 1)
    probs = np.zeros((len(norm), 3))
    probs[:, 0] = 1 - norm          # Healthy
    probs[:, 1] = norm * 0.3        # Degraded
    probs[:, 2] = norm * 0.7        # Faulty
    row_sum = probs.sum(axis=1, keepdims=True)
    return probs / np.maximum(row_sum, 1e-9)


def ensemble_subsystem(name: str, df: pd.DataFrame) -> dict | None:
    """
    Ensemble: Classical (w=0.5) + LSTM (w=0.3) + AE (w=0.2).
    Falls back gracefully if any model is missing.
    """
    cls_pkg  = load_classical(name)
    lstm_meta = load_lstm_meta(name)
    ae_meta   = load_ae_meta(name)

    contributions = {}
    weights_used  = {}

    # ── Classical ────────────────────────────────────────────
    cls_probs = predict_classical(cls_pkg, df) if cls_pkg else None
    if cls_probs is not None:
        contributions["classical"] = cls_probs
        weights_used["classical"]  = 0.5

    # ── LSTM ──────────────────────────────────────────────────
    lstm_probs = predict_lstm(name, df, lstm_meta) if lstm_meta else None
    if lstm_probs is not None:
        # Pad to match length of classical
        pad = len(df) - len(lstm_probs)
        if pad > 0:
            lstm_probs = np.vstack([
                np.full((pad, 3), [0.6, 0.3, 0.1]),
                lstm_probs
            ])
        elif pad < 0:
            lstm_probs = lstm_probs[-len(df):]
        contributions["lstm"]  = lstm_probs
        weights_used["lstm"]   = 0.3

    # ── Autoencoder ───────────────────────────────────────────
    if ae_meta:
        try:
            from day10_autoencoder import LSTMAutoencoder, make_windows
            ae_path = AE_DIR / f"{name.replace(' ','_')}.pt"
            if ae_path.exists():
                feat_cols  = [c for c in ae_meta["features"] if c in df.columns]
                n_features = ae_meta["n_features"]
                seq_len    = ae_meta["seq_len"]
                threshold  = ae_meta["threshold"]

                X = ae_meta["scaler"].transform(df[feat_cols].fillna(0))
                windows = make_windows(X, seq_len)
                if len(windows) > 0:
                    ae_model = LSTMAutoencoder(n_features).to(DEVICE)
                    ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
                    ae_model.eval()
                    Xt = torch.tensor(windows, dtype=torch.float32)
                    with torch.no_grad():
                        recon, _ = ae_model(Xt.to(DEVICE))
                        errors   = ((recon - Xt.to(DEVICE))**2).mean(dim=(1,2)).cpu().numpy()

                    ae_probs = anomaly_to_probs(errors, threshold)
                    pad      = len(df) - len(ae_probs)
                    if pad > 0:
                        ae_probs = np.vstack([np.full((pad,3),[0.7,0.2,0.1]), ae_probs])
                    contributions["autoencoder"] = ae_probs
                    weights_used["autoencoder"]  = 0.2
        except Exception as e:
            log.warning("  AE ensemble failed: %s", e)

    if not contributions:
        return None

    # Normalise weights
    total_w = sum(weights_used.values())
    ensemble_probs = sum(
        contributions[m] * (weights_used[m] / total_w)
        for m in contributions
    )

    pred_labels = ensemble_probs.argmax(axis=1)
    confidence  = ensemble_probs.max(axis=1)

    # Health score: confidence × (1 - faulty_prob) × 100
    health_score = (1 - ensemble_probs[:, 2]) * confidence * 100
    health_score = np.clip(health_score, 0, 100)

    # Per-UAV summary
    uav_col = df["uav_id"].values if "uav_id" in df.columns else np.zeros(len(df))
    per_uav = {}
    for uid in np.unique(uav_col):
        mask = uav_col == uid
        per_uav[int(uid)] = {
            "avg_health":    round(float(health_score[mask].mean()), 1),
            "min_health":    round(float(health_score[mask].min()),  1),
            "status":        LABELS[int(pred_labels[mask][-1])],   # last timestep
            "confidence":    round(float(confidence[mask].mean()),  3),
            "n_anomalies":   int((pred_labels[mask] >= 1).sum()),
        }

    # Estimate RUL: simple linear from health trajectory
    min_h  = float(health_score.min())
    avg_h  = float(health_score.mean())
    # RUL heuristic: if avg health 90% → 500h, 50% → 100h, 20% → 30h
    rul_h  = max(0, int(5.5 * avg_h - 50))

    return {
        "subsystem":     name,
        "avg_health":    round(avg_h, 1),
        "min_health":    round(min_h, 1),
        "rul_hours":     rul_h,
        "status":        LABELS[int(np.bincount(pred_labels).argmax())],
        "n_anomalies":   int((pred_labels >= 1).sum()),
        "models_used":   list(contributions.keys()),
        "per_uav":       per_uav,
        "health_series": health_score.tolist(),
    }


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════

def plot_final_health_bars(summary: dict):
    names   = list(summary.keys())
    healths = [summary[n]["avg_health"] for n in names]
    colors  = ["#22C55E" if h >= 70 else ("#F59E0B" if h >= 40 else "#EF4444")
               for h in healths]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars    = ax.barh(names, healths, color=colors, alpha=0.85)
    ax.bar_label(bars, labels=[f"{h:.1f}%" for h in healths],
                 padding=4, color="white", fontsize=9)
    ax.axvline(70, color="#F59E0B", ls="--", lw=1, alpha=0.7, label="Degraded")
    ax.axvline(40, color="#EF4444", ls="--", lw=1, alpha=0.7, label="Faulty")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Average Health Score (0–100)")
    ax.set_title("DRDO IVHM — Final Ensemble Health Scores (All Subsystems)",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(labelcolor="white", facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "day13_final_health.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_model_contribution(summary: dict):
    """Which models contributed to each subsystem's prediction."""
    names    = list(summary.keys())
    all_mods = ["classical", "lstm", "autoencoder"]
    matrix   = np.zeros((len(names), len(all_mods)))
    for i, n in enumerate(names):
        used = summary[n].get("models_used", [])
        for j, m in enumerate(all_mods):
            matrix[i, j] = 1 if m in used else 0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=all_mods, yticklabels=names, ax=ax,
                linewidths=0.4, linecolor="#060E1A",
                cbar_kws={"shrink": 0.7})
    ax.set_title("Model Contribution Matrix", color="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "day13_model_matrix.png", dpi=110, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run():
    log.info("="*62)
    log.info("  DRDO IVHM — Day 13: Ensemble + Report Generation")
    log.info("="*62)

    summary = {}

    for name in SUBSYSTEMS:
        path = DATA_PROCESSED / f"{name.replace(' ','_')}.csv"
        if not path.exists():
            continue

        log.info("\n── %s ──", name)
        df     = pd.read_csv(path).sort_values(["uav_id","timestep"])
        result = ensemble_subsystem(name, df)

        if result:
            summary[name] = result
            icon = ("✅" if result["avg_health"] >= 70 else
                    "⚠️ " if result["avg_health"] >= 40 else "🔴")
            log.info("  %s Health: %5.1f%%  RUL: %4dh  Status: %-10s  Models: %s",
                     icon, result["avg_health"], result["rul_hours"],
                     result["status"], ", ".join(result["models_used"]))
        else:
            log.warning("  No ensemble result for %s", name)

    if not summary:
        log.error("No results! Run days 1-3 first (LSTM+AE optional)")
        return {}

    # ── Build dashboard-ready JSON ────────────────────────────
    overall_health = np.mean([summary[n]["avg_health"] for n in summary])
    total_anomalies= sum(summary[n]["n_anomalies"] for n in summary)
    critical       = [n for n in summary if summary[n]["status"] == "Faulty"]

    report = {
        "overall_health":   round(float(overall_health), 1),
        "total_anomalies":  total_anomalies,
        "n_critical":       len(critical),
        "critical_systems": critical,
        "subsystems":       summary,
        "generated_at":     pd.Timestamp.now().isoformat(),
    }

    report_path = REPORT / "ivhm_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("\nReport saved → %s", report_path)

    # ── CSV summary ───────────────────────────────────────────
    rows = [{
        "subsystem":    n,
        "avg_health":   summary[n]["avg_health"],
        "min_health":   summary[n]["min_health"],
        "rul_hours":    summary[n]["rul_hours"],
        "status":       summary[n]["status"],
        "n_anomalies":  summary[n]["n_anomalies"],
        "models_used":  "|".join(summary[n]["models_used"]),
    } for n in summary]

    pd.DataFrame(rows).to_csv(REPORT / "ivhm_summary.csv", index=False)

    # ── Final printout ────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  DRDO IVHM — FINAL ENSEMBLE REPORT")
    print(f"{'='*62}")
    print(f"  Overall System Health : {overall_health:.1f}%")
    print(f"  Total Anomalies       : {total_anomalies}")
    print(f"  Critical Subsystems   : {critical if critical else 'None ✅'}")
    print(f"{'='*62}")
    print(f"  {'Subsystem':<20} {'Health':>8} {'RUL':>7} {'Status':<10} {'Models'}")
    print(f"  {'-'*58}")
    for n, r in summary.items():
        icon = "✅" if r["avg_health"] >= 70 else ("⚠️ " if r["avg_health"] >= 40 else "🔴")
        print(f"  {icon} {n:<18} {r['avg_health']:>8.1f}% {r['rul_hours']:>6}h "
              f"  {r['status']:<10} {', '.join(r['models_used'])}")

    plot_final_health_bars(summary)
    plot_model_contribution(summary)

    log.info("\n✅ Day 13 complete!")
    log.info("   JSON report → reports/ivhm_report.json")
    log.info("   CSV summary → reports/ivhm_summary.csv")
    return report


if __name__ == "__main__":
    run()
