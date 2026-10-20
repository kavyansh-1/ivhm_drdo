"""
DRDO IVHM — Day 4: Anomaly Detection + Health Scoring
Unsupervised layer ON TOP of Day 3 supervised models.
Isolation Forest + Local Outlier Factor → ensemble anomaly score.
Maps score to a 0-100 health score per UAV per timestep.
Outputs the fleet-wide health heatmap — the key visual for the dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import joblib, logging, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from config import (
    SUBSYSTEMS, DATA_PROCESSED, MODELS_DIR, ANOMALY_DIR,
    FIGURES_DIR, LABELS, PALETTE, COLORS, PLOT_STYLE, GOLD, RANDOM_SEED
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)
matplotlib.rcParams.update(PLOT_STYLE)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"health_label", "health_status", "uav_id", "timestep",
            "subsystem", "rul", "capacity_ah", "life_fraction", "fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns
            if c not in skip]


def norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)


# ════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ════════════════════════════════════════════════════════════

def ensemble_anomaly(X_train: np.ndarray,
                     X_all:   np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains IF + LOF on healthy-only samples, scores all data.

    Why ensemble?
    • Isolation Forest: tree-based, good at global outliers
    • LOF: density-based, catches local anomalies IF misses
    • Combined score is more robust than either alone.

    Returns:
        combined_score : [0, 1] — higher = more anomalous
        if_score       : raw IF anomaly score
        lof_score      : raw LOF anomaly score
    """
    # Isolation Forest
    iso = IsolationForest(
        n_estimators=200, contamination=0.15,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    iso.fit(X_train)
    if_raw = -iso.score_samples(X_all)           # negate: higher = worse

    # Local Outlier Factor (transductive)
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=0.15,
        novelty=True, n_jobs=-1
    )
    lof.fit(X_train)
    lof_raw = -lof.score_samples(X_all)          # negate: higher = worse

    combined = (norm01(if_raw) + norm01(lof_raw)) / 2.0
    return combined, if_raw, lof_raw


def score_to_health(anomaly_score: np.ndarray) -> np.ndarray:
    """
    Non-linear mapping: anomaly score [0,1] → health [0,100].
    Power of 1.5 makes the scale more sensitive in the degraded zone
    (50-80%) which is where early warning matters most.
    """
    return np.clip(100.0 * (1.0 - anomaly_score) ** 1.5, 0.0, 100.0)


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════

def plot_health_timeline(df: pd.DataFrame, health: np.ndarray,
                          anomaly: np.ndarray, subsys: str, uav_id: int = 0):
    """
    Health score + anomaly signal + ground truth for one UAV.
    Key plot for the DRDO report — shows early detection capability.
    """
    mask    = df["uav_id"] == uav_id
    steps   = df.loc[mask, "timestep"].values
    truth   = df.loc[mask, "health_label"].values
    h       = health[mask]
    a       = anomaly[mask]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"{subsys} | UAV #{uav_id} — Health Timeline",
                 color="white", fontsize=13, fontweight="bold")

    # Panel 1: Health score
    ax = axes[0]
    ax.plot(steps, h, color=GOLD, lw=1.8)
    ax.fill_between(steps, h, alpha=0.15, color=GOLD)
    ax.axhline(60, color=PALETTE[1], ls="--", lw=0.9, alpha=0.7, label="Degraded (<60)")
    ax.axhline(30, color=PALETTE[2], ls="--", lw=0.9, alpha=0.7, label="Faulty (<30)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Health Score")
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True)

    # Panel 2: Anomaly score
    ax = axes[1]
    ax.plot(steps, a, color="#38BDF8", lw=1.2, alpha=0.8)
    ax.fill_between(steps, a, alpha=0.2, color="#38BDF8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Anomaly Score")
    ax.axhline(0.6, color=PALETTE[2], ls="--", lw=0.9, alpha=0.7)
    ax.grid(True)

    # Panel 3: Ground truth
    ax = axes[2]
    label_colors = [PALETTE[l] for l in truth]
    ax.scatter(steps, truth, c=label_colors, s=5, alpha=0.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Degraded", "Faulty"], fontsize=8)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Timestep / Cycle")
    ax.grid(True)

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    out = FIGURES_DIR / f"day4_health_{subsys.replace(' ','_')}_uav{uav_id}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_anomaly_dist(anomaly: np.ndarray, health: np.ndarray,
                       labels: np.ndarray, subsys: str):
    """Distribution of anomaly score and health score by true label."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{subsys} — Anomaly & Health Score Distributions",
                 color="white", fontsize=11)

    for label_id, color in enumerate(PALETTE):
        mask = labels == label_id
        if mask.sum() < 5:
            continue
        pd.Series(anomaly[mask]).plot.kde(
            ax=ax1, color=color, label=LABELS[label_id], lw=1.8)
        pd.Series(health[mask]).plot.kde(
            ax=ax2, color=color, label=LABELS[label_id], lw=1.8)

    ax1.set_title("Anomaly Score", color=GOLD)
    ax1.set_xlabel("Score [0–1]")
    ax2.set_title("Health Score", color=GOLD)
    ax2.set_xlabel("Score [0–100]")

    for ax in (ax1, ax2):
        ax.legend(fontsize=8, labelcolor="white",
                  facecolor="#060E1A", edgecolor="#1E3A5F")
        ax.grid(True)

    plt.tight_layout()
    out = FIGURES_DIR / f"day4_dist_{subsys.replace(' ','_')}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


def plot_fleet_heatmap(fleet: dict):
    """
    The key deliverable of Day 4.
    Heatmap: subsystem × UAV → average health score (0–100).
    Red = needs maintenance, green = healthy.
    """
    subsystems = list(fleet.keys())
    uav_ids    = sorted({uid for v in fleet.values() for uid in v})

    matrix = np.full((len(subsystems), len(uav_ids)), np.nan)
    for i, sub in enumerate(subsystems):
        for j, uid in enumerate(uav_ids):
            matrix[i, j] = fleet[sub].get(uid, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = sns.heatmap(
        matrix, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
        xticklabels=[f"UAV-{i}" for i in uav_ids],
        yticklabels=subsystems,
        annot=True, fmt=".0f", annot_kws={"size": 9},
        linewidths=0.4, linecolor="#060E1A",
        cbar_kws={"label": "Health Score (0–100)", "shrink": 0.8}
    )
    ax.set_title("DRDO IVHM — Fleet Health Overview",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("UAV Instance", labelpad=8)
    ax.set_ylabel("Subsystem",    labelpad=8)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "day4_fleet_heatmap.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_rul_vs_health(df: pd.DataFrame, health: np.ndarray,
                        subsys: str):
    """Scatter: actual RUL vs predicted health score (only for C-MAPSS / real data)."""
    if "rul" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["rul"].values, health,
               c=health, cmap="RdYlGn", s=5, alpha=0.4,
               vmin=0, vmax=100)
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Predicted Health Score")
    ax.set_title(f"{subsys} — RUL vs Health Score", color="white", fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    out = FIGURES_DIR / f"day4_rul_vs_health_{subsys.replace(' ','_')}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run() -> dict:
    log.info("=" * 62)
    log.info("  DRDO IVHM — Day 4: Anomaly Detection + Health Scoring")
    log.info("  Models: IsolationForest + LOF (ensemble)")
    log.info("=" * 62)

    fleet_summary = {}   # {subsys: {uav_id: avg_health_score}}
    health_data   = {}   # {subsys: df with health/anomaly cols}

    for name in SUBSYSTEMS:
        proc_path = DATA_PROCESSED / f"{name.replace(' ', '_')}.csv"
        if not proc_path.exists():
            log.warning("Missing: %s — run day2_eda.py first", proc_path)
            continue

        log.info("\n── %s ──", name)
        df        = pd.read_csv(proc_path)
        feat_cols = get_feature_cols(df)

        scaler = StandardScaler()
        X      = scaler.fit_transform(df[feat_cols].fillna(0))

        # Train ONLY on healthy samples — key principle of unsupervised PHM
        healthy_mask = df["health_label"] == 0
        if healthy_mask.sum() < 20:
            log.warning("  Too few healthy samples (%d), skipping", healthy_mask.sum())
            continue

        X_healthy = X[healthy_mask]

        anomaly_scores, if_scores, lof_scores = ensemble_anomaly(X_healthy, X)
        health_scores = score_to_health(anomaly_scores)

        # Attach scores to dataframe
        df["anomaly_score"]  = anomaly_scores
        df["if_score"]       = if_scores
        df["lof_score"]      = lof_scores
        df["health_score"]   = health_scores
        df["is_anomaly"]     = (anomaly_scores > 0.6).astype(int)

        # Per-UAV average health
        uav_health = df.groupby("uav_id")["health_score"].mean().to_dict()
        fleet_summary[name] = uav_health

        # Save enriched data
        save_cols = ["uav_id", "timestep", "health_label",
                     "anomaly_score", "health_score", "is_anomaly"]
        out_path  = DATA_PROCESSED / f"{name.replace(' ', '_')}_health.csv"
        df[save_cols].to_csv(out_path, index=False)

        health_data[name] = df

        # Stats
        n_anomalies  = df["is_anomaly"].sum()
        avg_h        = health_scores.mean()
        min_h        = health_scores.min()

        status = "🔴 CRITICAL" if avg_h < 40 else ("⚠️ DEGRADED" if avg_h < 70 else "✅ HEALTHY")
        log.info("  %s | Avg health: %5.1f | Min: %5.1f | Anomalies: %d/%d",
                 status, avg_h, min_h, n_anomalies, len(health_scores))

        # Plots
        if "uav_id" in df.columns and df["uav_id"].nunique() > 0:
            plot_health_timeline(df, health_scores, anomaly_scores, name, uav_id=0)

        plot_anomaly_dist(anomaly_scores, health_scores,
                          df["health_label"].values, name)
        plot_rul_vs_health(df, health_scores, name)

        # Save anomaly model
        model_out = ANOMALY_DIR / f"{name.replace(' ','_')}_anomaly.pkl"
        joblib.dump({"scaler": scaler, "features": feat_cols,
                     "threshold": 0.6}, model_out)

    # ── Fleet heatmap ─────────────────────────────────────────
    if fleet_summary:
        plot_fleet_heatmap(fleet_summary)

    # ── Per-subsystem health summary ──────────────────────────
    print(f"\n{'='*62}")
    print("  FLEET HEALTH SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Subsystem':<20} {'Avg Health':>12} {'Min':>8} {'Critical UAVs':>15}")
    print(f"  {'-'*58}")

    for name, uav_health in fleet_summary.items():
        vals     = list(uav_health.values())
        avg      = np.mean(vals)
        mn       = np.min(vals)
        n_crit   = sum(1 for v in vals if v < 40)
        status   = "🔴" if avg < 40 else ("⚠️ " if avg < 70 else "✅")
        print(f"  {status} {name:<18} {avg:>12.1f} {mn:>8.1f} {n_crit:>15d}")

    log.info("\n✅ Day 4 complete!")
    log.info("   Health CSVs → data/processed/*_health.csv")
    log.info("   Anomaly models → models/anomaly/")
    log.info("   Key plot → reports/figures/day4_fleet_heatmap.png")

    return {"fleet": fleet_summary, "data": health_data}


if __name__ == "__main__":
    run()
