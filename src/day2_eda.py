"""
DRDO IVHM — Day 2: EDA + Feature Engineering
Loads raw CSVs, runs statistical analysis, then engineers
time-series features: rolling stats, rate-of-change, z-score, lags.
These features are what lets classical ML beat naive baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
import warnings
warnings.filterwarnings("ignore")

from config import (SUBSYSTEMS, DATA_RAW, DATA_PROCESSED,
                    FIGURES_DIR, LABELS, PALETTE, COLORS,
                    PLOT_STYLE, GOLD, RANDOM_SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)
matplotlib.rcParams.update(PLOT_STYLE)


# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════

def load_all_raw() -> dict:
    data = {}
    for name in SUBSYSTEMS:
        path = DATA_RAW / f"{name.replace(' ', '_')}.csv"
        if not path.exists():
            log.warning("Missing: %s  — run day1_data_gen.py first", path)
            continue
        data[name] = pd.read_csv(path)
        log.info("Loaded %-18s  shape: %s", name, data[name].shape)
    return data


# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

def get_sensor_cols(df: pd.DataFrame) -> list[str]:
    """Returns only numeric sensor columns, excluding metadata."""
    skip = {"health_label", "health_status", "uav_id", "timestep",
            "subsystem", "rul", "capacity_ah", "life_fraction", "fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns
            if c not in skip]


def engineer_features(df: pd.DataFrame, sensors: list[str],
                       window: int = 10) -> pd.DataFrame:
    """
    Adds per-sensor time-series features per UAV.

    Features added for each sensor S:
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ S_roll_mean         │ Rolling mean (window=10) → smoothed signal   │
    │ S_roll_std          │ Rolling std  → captures variance spikes      │
    │ S_diff1             │ First derivative → rate of change            │
    │ S_diff2             │ Second derivative → acceleration of change   │
    │ S_zscore            │ Z-score from rolling mean → anomaly signal   │
    │ S_lag5 / S_lag10    │ Lagged values → context from past readings   │
    └─────────────────────┴──────────────────────────────────────────────┘
    These features are critical — they let models understand TRENDS,
    not just instantaneous values. A healthy sensor at value=50 looks
    identical to a degrading sensor at value=50 without trend features.
    """
    df = df.copy().sort_values(["uav_id", "timestep"])

    for s in sensors:
        if s not in df.columns:
            continue
        g = df.groupby("uav_id")[s]

        rm = g.transform(lambda x: x.rolling(window, min_periods=1).mean())
        rs = g.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))

        df[f"{s}_roll_mean"] = rm
        df[f"{s}_roll_std"]  = rs
        df[f"{s}_diff1"]     = g.transform(lambda x: x.diff().fillna(0))
        df[f"{s}_diff2"]     = g.transform(lambda x: x.diff().diff().fillna(0))
        df[f"{s}_zscore"]    = ((df[s] - rm) / rs.replace(0, 1)).clip(-5, 5)
        df[f"{s}_lag5"]      = g.transform(lambda x: x.shift(5).bfill())
        df[f"{s}_lag10"]     = g.transform(lambda x: x.shift(10).bfill())

    return df


# ════════════════════════════════════════════════════════════
# EDA PLOTS
# ════════════════════════════════════════════════════════════

def plot_distributions(df: pd.DataFrame, sensors: list[str],
                        name: str, max_sensors: int = 5):
    """KDE distributions per sensor coloured by health state."""
    sensors_to_plot = sensors[:max_sensors]
    fig, axes = plt.subplots(1, len(sensors_to_plot),
                              figsize=(4 * len(sensors_to_plot), 4))
    if len(sensors_to_plot) == 1:
        axes = [axes]

    fig.suptitle(f"{name} — Sensor Distributions by Health State",
                 color="white", fontsize=12, fontweight="bold")

    for ax, s in zip(axes, sensors_to_plot):
        for status, color in COLORS.items():
            vals = df[df["health_status"] == status][s].dropna()
            if len(vals) > 10:
                vals.plot.kde(ax=ax, color=color, label=status, lw=1.8)
        ax.set_title(s, color=GOLD, fontsize=8)
        ax.set_xlabel("", color="#6B8CAE")
        ax.grid(True)

    axes[0].legend(fontsize=8, labelcolor="white",
                   facecolor="#060E1A", edgecolor="#1E3A5F")
    plt.tight_layout()
    out = FIGURES_DIR / f"day2_dist_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_correlation(df: pd.DataFrame, sensors: list[str], name: str):
    """Sensor–sensor correlation heatmap."""
    avail = [s for s in sensors if s in df.columns]
    if len(avail) < 2:
        return

    corr = df[avail].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdYlGn",
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.5, linecolor="#1E3A5F",
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"{name} — Sensor Correlation", color="white", fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / f"day2_corr_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_pca(df: pd.DataFrame, sensors: list[str], name: str):
    """
    PCA scatter — do the 3 health classes cluster in 2D sensor space?
    If they cluster well, classical ML will work great.
    If they overlap, we need deep learning.
    """
    avail = [s for s in sensors if s in df.columns]
    if len(avail) < 2:
        return

    sample = df.sample(min(3000, len(df)), random_state=RANDOM_SEED)
    X      = StandardScaler().fit_transform(sample[avail].fillna(0))
    pc     = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label_id, color in enumerate(PALETTE):
        mask = sample["health_label"].values == label_id
        ax.scatter(pc[mask, 0], pc[mask, 1],
                   c=color, label=LABELS[label_id],
                   s=6, alpha=0.45)

    ax.set_title(f"{name} — PCA (2D Sensor Space)", color="white", fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(labelcolor="white", facecolor="#060E1A",
              edgecolor="#1E3A5F", markerscale=4)
    ax.grid(True)
    plt.tight_layout()
    out = FIGURES_DIR / f"day2_pca_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_feature_count(data_raw: dict, data_feat: dict):
    """Bar chart showing how many features we added per subsystem."""
    names   = list(data_raw.keys())
    orig    = [len(get_sensor_cols(data_raw[n]))  for n in names]
    engineered = [len(get_sensor_cols(data_feat[n])) for n in names]

    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - w/2, orig,       w, label="Original sensors", color="#38BDF8", alpha=0.85)
    ax.bar(x + w/2, engineered, w, label="After engineering",color=GOLD,     alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature Count: Before vs After Engineering", color="white", fontsize=12)
    ax.legend(labelcolor="white", facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True, axis="y")
    plt.tight_layout()
    out = FIGURES_DIR / "day2_feature_count.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def print_stats(df: pd.DataFrame, sensors: list[str], name: str):
    avail = [s for s in sensors if s in df.columns]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(df[avail].describe().round(3).to_string())
    print(f"\n  Class balance:")
    print(df["health_status"].value_counts().to_string())
    print(f"\n  Shape: {df.shape}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run() -> dict:
    log.info("=" * 62)
    log.info("  DRDO IVHM — Day 2: EDA + Feature Engineering")
    log.info("=" * 62)

    data_raw  = load_all_raw()
    data_feat = {}

    for name, df in data_raw.items():
        log.info("\n── %s ──", name)
        sensors = get_sensor_cols(df)

        print_stats(df, sensors, name)

        # Feature engineering
        df_feat = engineer_features(df, sensors)
        out     = DATA_PROCESSED / f"{name.replace(' ', '_')}.csv"
        df_feat.to_csv(out, index=False)
        data_feat[name] = df_feat

        n_orig  = len(sensors)
        n_feat  = len(get_sensor_cols(df_feat))
        log.info("  Features: %d → %d  |  Saved → %s", n_orig, n_feat, out)

    # ── Visualise for 3 key subsystems ───────────────────────
    for subsys in ["Li-Ion Battery", "ECS", "Airframe"]:
        if subsys not in data_raw:
            continue
        sensors = get_sensor_cols(data_raw[subsys])
        log.info("\n── Plotting: %s ──", subsys)
        plot_distributions(data_raw[subsys], sensors, subsys)
        plot_correlation(data_raw[subsys], sensors, subsys)
        plot_pca(data_raw[subsys], sensors, subsys)

    plot_feature_count(data_raw, data_feat)

    log.info("\n✅ Day 2 complete! Processed CSVs → data/processed/")
    return data_feat


if __name__ == "__main__":
    run()
