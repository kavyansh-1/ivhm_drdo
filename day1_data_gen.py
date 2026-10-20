"""
DRDO IVHM — Day 1: Data Generation
Loads real datasets (C-MAPSS, NASA Battery, FEMTO) for 3 subsystems.
Generates physics-based synthetic data with smooth degradation curves for the rest.
Auto-detects extracted folder names so long paths like
"6. Turbofan Engine Degradation Simulation Data Set" work out of the box.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import logging
from tqdm import tqdm

from config import (
    SUBSYSTEMS, CMAPSS_DIR, BATTERY_DIR, FEMTO_DIR,
    DATA_RAW, FIGURES_DIR, RANDOM_SEED, N_UAVS_SYNTHETIC,
    N_TIMESTEPS, FAULT_START_RANGE, LABELS, PALETTE, COLORS,
    PLOT_STYLE, GOLD, RUL_FAULTY, RUL_DEGRADED
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

def find_file(base_dir: Path, pattern: str) -> Path | None:
    """
    Recursively searches base_dir for a file matching pattern.
    Handles cases where NASA datasets extract into subfolders with long names.
    """
    matches = list(base_dir.rglob(pattern))
    return matches[0] if matches else None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def degradation_weight(t: int, fault_start: int, total: int) -> float:
    """
    Smooth sigmoid transition from 0.0 (healthy) → 1.0 (faulty).
    Mimics real component degradation — gradual, not sudden.
    """
    if t < fault_start:
        return 0.0
    norm = (t - fault_start) / max(total - fault_start, 1)
    return float(sigmoid(10 * norm - 5))


def weight_to_label(w: float) -> int:
    if w < 0.35: return 0
    if w < 0.70: return 1
    return 2


# ════════════════════════════════════════════════════════════
# REAL DATASET LOADERS
# ════════════════════════════════════════════════════════════

def load_cmapss() -> pd.DataFrame | None:
    """
    NASA C-MAPSS Turbofan Engine Dataset → AIRFRAME subsystem.

    21 sensors per engine, run-to-failure trajectories.
    We use FD001 (single fault, single operating condition — cleanest subset).
    RUL binned: >100 = Healthy, 30-100 = Degraded, <30 = Faulty.
    """
    train_file = find_file(CMAPSS_DIR, "train_FD001.txt")
    if train_file is None:
        log.warning("C-MAPSS train_FD001.txt not found in %s", CMAPSS_DIR)
        return None

    col_names = (["engine_id", "cycle", "op1", "op2", "op3"] +
                 [f"s{i}" for i in range(1, 22)])

    df = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)

    # Compute RUL per engine
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]

    # Health label from RUL
    df["health_label"] = np.where(
        df["rul"] <= RUL_FAULTY, 2,
        np.where(df["rul"] <= RUL_DEGRADED, 1, 0)
    )
    df["health_status"] = df["health_label"].map(LABELS)
    df["subsystem"]     = "Airframe"
    df["uav_id"]        = df["engine_id"]
    df["timestep"]      = df["cycle"]

    # Map most physically meaningful sensors to airframe health proxies
    sensor_map = {
        "s2":  "lpc_outlet_temp",   # Low-Pressure Compressor outlet temp
        "s3":  "hpc_outlet_temp",   # High-Pressure Compressor outlet temp
        "s4":  "lpt_outlet_temp",   # Low-Pressure Turbine outlet temp
        "s7":  "fan_speed_rpm",     # Physical fan speed
        "s8":  "core_speed_rpm",    # Core speed
        "s11": "bypass_ratio",      # Bypass ratio
        "s12": "fuel_flow_ratio",   # Burner fuel-air ratio
        "s15": "nrf_speed",         # Fan speed ratio (corrected)
    }
    df = df.rename(columns=sensor_map)

    keep = list(sensor_map.values()) + [
        "uav_id", "timestep", "rul", "health_label", "health_status", "subsystem"
    ]
    df = df[keep].reset_index(drop=True)

    log.info("C-MAPSS loaded  → Airframe       | rows: %6d | H:%d D:%d F:%d",
             len(df),
             (df.health_label==0).sum(),
             (df.health_label==1).sum(),
             (df.health_label==2).sum())
    return df


def load_nasa_battery() -> pd.DataFrame | None:
    """
    NASA Battery Dataset → LI-ION BATTERY subsystem.

    Li-Ion batteries charged/discharged until end-of-life (~70% capacity).
    We load .mat files for batteries B0005, B0006, B0007, B0018.
    Health label from capacity fade ratio vs nominal 2Ah.
    """
    try:
        import scipy.io as sio
    except ImportError:
        log.error("scipy not installed. Run: pip install scipy")
        return None

    nominal_capacity = 2.0   # Ah rated
    battery_ids      = ["B0005", "B0006", "B0007", "B0018"]
    all_rows         = []

    for bat_idx, bat_name in enumerate(battery_ids):
        mat_file = find_file(BATTERY_DIR, f"{bat_name}.mat")
        if mat_file is None:
            log.warning("  %s.mat not found, skipping", bat_name)
            continue

        try:
            mat  = sio.loadmat(str(mat_file), simplify_cells=True)
            data = mat[bat_name]["cycle"]
        except Exception as e:
            log.warning("  Failed to load %s: %s", bat_name, e)
            continue

        cycle_num = 0
        for cycle in data:
            if cycle.get("type") != "discharge":
                continue

            vals = cycle.get("data", {})

            def safe_mean(key):
                v = vals.get(key, None)
                if v is None or (hasattr(v, "__len__") and len(v) == 0):
                    return np.nan
                return float(np.mean(v))

            voltage   = safe_mean("Voltage_measured")
            current   = abs(safe_mean("Current_measured"))
            temp      = safe_mean("Temperature_measured")
            capacity  = vals.get("Capacity", np.nan)
            if hasattr(capacity, "__len__"):
                capacity = float(np.mean(capacity)) if len(capacity) > 0 else np.nan
            else:
                capacity = float(capacity)

            cap_ratio = capacity / nominal_capacity if not np.isnan(capacity) else np.nan

            all_rows.append({
                "voltage_v":       voltage,
                "current_a":       current,
                "temp_c":          temp,
                "soc_pct":         cap_ratio * 100 if not np.isnan(cap_ratio) else np.nan,
                "resistance_mohm": 10 + (1 - cap_ratio) * 80 if not np.isnan(cap_ratio) else np.nan,
                "capacity_ah":     capacity,
                "uav_id":          bat_idx,
                "timestep":        cycle_num,
                "subsystem":       "Li-Ion Battery",
            })
            cycle_num += 1

    if not all_rows:
        log.warning("No battery discharge cycles loaded from %s", BATTERY_DIR)
        return None

    df = pd.DataFrame(all_rows).dropna(subset=["voltage_v", "soc_pct"])

    # Label from SOC (capacity ratio)
    df["health_label"] = np.where(
        df["soc_pct"] < 70, 2,
        np.where(df["soc_pct"] < 90, 1, 0)
    )
    df["health_status"] = df["health_label"].map(LABELS)

    log.info("NASA Battery loaded → Li-Ion Batt | rows: %6d | H:%d D:%d F:%d",
             len(df),
             (df.health_label==0).sum(),
             (df.health_label==1).sum(),
             (df.health_label==2).sum())
    return df


def load_femto_bearing() -> pd.DataFrame | None:
    """
    FEMTO-PRONOSTIA Bearing Dataset → HYDRAULICS subsystem.

    Bearing vibration (horizontal + vertical accelerometers) run to failure.
    Used as a proxy for hydraulic pump mechanical degradation.
    Health labeled by % of total lifetime elapsed.
    """
    # FEMTO has Learning_set and Test_set subdirs
    bearing_pattern = "Bearing*_*"
    bearing_dirs    = sorted(FEMTO_DIR.rglob(bearing_pattern))
    bearing_dirs    = [d for d in bearing_dirs if d.is_dir()]

    if not bearing_dirs:
        log.warning("FEMTO bearing folders not found in %s", FEMTO_DIR)
        return None

    all_dfs = []
    for bid, bdir in enumerate(bearing_dirs[:8]):   # use up to 8 bearings
        csv_files = sorted(bdir.glob("acc_*.csv"))
        if not csv_files:
            continue

        n_total = len(csv_files)
        rows    = []

        for i, f in enumerate(csv_files):
            try:
                temp = pd.read_csv(
                    f, header=None,
                    names=["hour", "min", "sec", "microsec", "h_acc", "v_acc"]
                )
                h_rms = float(np.sqrt(np.mean(temp["h_acc"] ** 2)))
                v_rms = float(np.sqrt(np.mean(temp["v_acc"] ** 2)))
                life_frac = i / n_total

                rows.append({
                    "pressure_bar":   200 - life_frac * 110 + np.random.normal(0, 2),
                    "flow_lpm":       10  - life_frac * 8.5 + np.random.normal(0, 0.3),
                    "temp_c":         45  + life_frac * 75  + np.random.normal(0, 1),
                    "fluid_level_pct":100 - life_frac * 80  + np.random.normal(0, 2),
                    "pump_rpm":       3000- life_frac * 2100+ np.random.normal(0, 30),
                    "vibration_h_rms": h_rms,
                    "vibration_v_rms": v_rms,
                    "uav_id":          bid,
                    "timestep":        i,
                    "life_fraction":   life_frac,
                    "subsystem":       "Hydraulics",
                })
            except Exception:
                continue

        if not rows:
            continue

        df_b = pd.DataFrame(rows)
        n    = len(df_b)
        df_b["health_label"] = np.where(
            df_b["life_fraction"] < 0.60, 0,
            np.where(df_b["life_fraction"] < 0.85, 1, 2)
        )
        df_b["health_status"] = df_b["health_label"].map(LABELS)
        all_dfs.append(df_b)

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    log.info("FEMTO loaded    → Hydraulics     | rows: %6d | H:%d D:%d F:%d",
             len(df),
             (df.health_label==0).sum(),
             (df.health_label==1).sum(),
             (df.health_label==2).sum())
    return df


# ════════════════════════════════════════════════════════════
# SYNTHETIC GENERATOR
# ════════════════════════════════════════════════════════════

def _generate_sensor_series(cfg: dict, sensor_idx: int,
                             n_steps: int, fault_start: int) -> np.ndarray:
    """
    Momentum-based random walk that follows a sigmoid degradation curve.
    momentum=0.82 gives realistic sensor inertia (slow response to changes).
    """
    n_lo, n_hi = cfg["normal"][sensor_idx]
    d_lo, d_hi = cfg["degraded"][sensor_idx]
    f_lo, f_hi = cfg["faulty"][sensor_idx]
    noise_scale = cfg["noise"][sensor_idx] * (n_hi - n_lo)

    current = np.random.uniform(n_lo, n_hi)
    series  = np.zeros(n_steps)

    for t in range(n_steps):
        w = degradation_weight(t, fault_start, n_steps)
        if w < 0.5:
            b  = w / 0.5
            lo = n_lo + b * (d_lo - n_lo)
            hi = n_hi + b * (d_hi - n_hi)
        else:
            b  = (w - 0.5) / 0.5
            lo = d_lo + b * (f_lo - d_lo)
            hi = d_hi + b * (f_hi - d_hi)

        target  = np.random.uniform(lo, hi)
        current = current * 0.82 + target * 0.18
        current += np.random.normal(0, noise_scale)
        series[t] = current

    return series


def generate_synthetic(name: str,
                        n_uavs: int  = N_UAVS_SYNTHETIC,
                        n_steps: int = N_TIMESTEPS) -> pd.DataFrame:
    cfg = SUBSYSTEMS[name]
    all_rows = []

    for uav_id in range(n_uavs):
        fault_start = int(np.random.uniform(*FAULT_START_RANGE) * n_steps)
        sensor_data = {
            sensor: _generate_sensor_series(cfg, i, n_steps, fault_start)
            for i, sensor in enumerate(cfg["sensors"])
        }
        df_uav = pd.DataFrame(sensor_data)
        weights = [degradation_weight(t, fault_start, n_steps) for t in range(n_steps)]
        df_uav["health_label"]  = [weight_to_label(w) for w in weights]
        df_uav["health_status"] = df_uav["health_label"].map(LABELS)
        df_uav["uav_id"]        = uav_id
        df_uav["timestep"]      = range(n_steps)
        df_uav["subsystem"]     = name
        all_rows.append(df_uav)

    return pd.concat(all_rows, ignore_index=True)


# ════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════

def plot_degradation(data: dict, subsys: str = "Li-Ion Battery"):
    """Sensor readings over time for UAV #0, colour-coded by health state."""
    df      = data[subsys]
    uav_df  = df[df["uav_id"] == 0].copy()

    sensor_cols = [c for c in uav_df.columns
                   if c not in ["health_label", "health_status",
                                "uav_id", "timestep", "subsystem",
                                "rul", "capacity_ah", "life_fraction"]][:6]

    fig, axes = plt.subplots(len(sensor_cols), 1,
                              figsize=(14, 2.8 * len(sensor_cols)), sharex=True)
    if len(sensor_cols) == 1:
        axes = [axes]

    fig.suptitle(f"{subsys} — Sensor Degradation (UAV #0)",
                 color="white", fontsize=13, fontweight="bold")

    for ax, sensor in zip(axes, sensor_cols):
        for status, color in COLORS.items():
            mask = uav_df["health_status"] == status
            ax.scatter(uav_df.loc[mask, "timestep"],
                       uav_df.loc[mask, sensor],
                       c=color, s=5, alpha=0.55, label=status)
        ax.set_ylabel(sensor, color=GOLD, fontsize=8)
        ax.grid(True)

    axes[0].legend(fontsize=8, labelcolor="white",
                   facecolor="#060E1A", edgecolor="#1E3A5F", markerscale=3)
    axes[-1].set_xlabel("Timestep / Cycle", color="#6B8CAE")
    plt.tight_layout()

    out = FIGURES_DIR / f"day1_degradation_{subsys.replace(' ', '_')}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_class_balance(data: dict):
    """Grouped bar chart — class balance across all subsystems."""
    names    = list(data.keys())
    healthy  = [(data[n]["health_label"] == 0).mean() * 100 for n in names]
    degraded = [(data[n]["health_label"] == 1).mean() * 100 for n in names]
    faulty   = [(data[n]["health_label"] == 2).mean() * 100 for n in names]

    x, w = np.arange(len(names)), 0.26
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(x - w, healthy,  w, label="Healthy",  color=PALETTE[0], alpha=0.85)
    ax.bar(x,     degraded, w, label="Degraded", color=PALETTE[1], alpha=0.85)
    ax.bar(x + w, faulty,   w, label="Faulty",   color=PALETTE[2], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of samples")
    ax.set_title("Class Distribution Across Subsystems", color="white", fontsize=12)
    ax.legend(labelcolor="white", facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True, axis="y")

    out = FIGURES_DIR / "day1_class_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


def plot_source_summary(data: dict):
    """Pie chart — how many rows come from real vs synthetic sources."""
    real_total  = sum(len(data[n]) for n in data
                      if SUBSYSTEMS[n]["source"] == "real" and n in data)
    synth_total = sum(len(data[n]) for n in data
                      if SUBSYSTEMS[n]["source"] == "synthetic" and n in data)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([real_total, synth_total],
           labels=["Real Datasets\n(NASA/FEMTO)", "Synthetic\n(Physics-based)"],
           colors=[GOLD, "#38BDF8"],
           autopct="%1.0f%%", startangle=90,
           textprops={"color": "white", "fontsize": 11},
           wedgeprops={"edgecolor": "#060E1A", "linewidth": 2})
    ax.set_title("Data Source Mix", color="white", fontsize=13, fontweight="bold")

    out = FIGURES_DIR / "day1_source_mix.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    log.info("Saved → %s", out)


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def run() -> dict:
    log.info("=" * 62)
    log.info("  DRDO IVHM — Day 1: Data Generation")
    log.info("  Real datasets: C-MAPSS | NASA Battery | FEMTO")
    log.info("  Synthetic: Fuel, Electronics, Actuators, ECS, Electrical")
    log.info("=" * 62)

    real_loaders = {
        "Airframe":      load_cmapss,
        "Li-Ion Battery": load_nasa_battery,
        "Hydraulics":    load_femto_bearing,
    }

    data = {}

    # ── Load real datasets ────────────────────────────────────
    log.info("\n── Real Datasets ──────────────────────────────────────")
    for name, loader in real_loaders.items():
        df = loader()
        if df is not None:
            data[name] = df
        else:
            log.warning("  Falling back to synthetic for: %s", name)
            data[name] = generate_synthetic(name)

    # ── Generate synthetic ────────────────────────────────────
    log.info("\n── Synthetic Data ─────────────────────────────────────")
    synthetic_subsystems = [n for n in SUBSYSTEMS
                            if SUBSYSTEMS[n]["source"] == "synthetic"]
    for name in tqdm(synthetic_subsystems, desc="Generating"):
        data[name] = generate_synthetic(name)
        log.info("  ✅ %-18s | rows: %5d", name, len(data[name]))

    # ── Save all to CSV ───────────────────────────────────────
    log.info("\n── Saving raw CSVs ────────────────────────────────────")
    for name, df in data.items():
        out = DATA_RAW / f"{name.replace(' ', '_')}.csv"
        df.to_csv(out, index=False)

    # ── Print summary ─────────────────────────────────────────
    log.info("\n%s", "=" * 62)
    log.info("  SUMMARY")
    log.info("%s", "=" * 62)
    total_rows = 0
    for name, df in data.items():
        h = (df.health_label == 0).sum()
        d = (df.health_label == 1).sum()
        f = (df.health_label == 2).sum()
        src = "🔬 REAL " if SUBSYSTEMS[name]["source"] == "real" else "🔧 SYNTH"
        log.info("  %s | %-18s | rows: %5d | H:%4d D:%4d F:%4d",
                 src, name, len(df), h, d, f)
        total_rows += len(df)
    log.info("  Total rows: %d", total_rows)

    # ── Visualise ─────────────────────────────────────────────
    log.info("\n── Generating plots ───────────────────────────────────")
    plot_class_balance(data)
    plot_source_summary(data)

    # Pick best available subsystem for degradation curve
    for subsys in ["Li-Ion Battery", "Airframe", "Hydraulics", "ECS"]:
        if subsys in data:
            plot_degradation(data, subsys)
            break

    log.info("\n✅ Day 1 complete! CSVs → data/raw/  |  Plots → reports/figures/")
    return data


if __name__ == "__main__":
    run()
