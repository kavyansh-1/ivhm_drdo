"""
DRDO IVHM — config.py
Single source of truth. All paths, hyperparameters, and subsystem
definitions live here. Change once → everything updates.
"""
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ── Project root (auto-detects wherever this file sits) ──────
ROOT = Path(__file__).parent

# ── Data paths ────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_REAL      = ROOT / "data" / "real"

CMAPSS_DIR  = DATA_REAL / "CMAPSS"
BATTERY_DIR = DATA_REAL / "Battery"
FEMTO_DIR   = DATA_REAL / "FEMTO"

# ── Output paths ──────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
ANOMALY_DIR = ROOT / "models" / "anomaly"
FIGURES_DIR = ROOT / "reports" / "figures"

# Create all dirs on import
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, ANOMALY_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────
RANDOM_SEED       = 42
TEST_SIZE         = 0.2
CV_FOLDS          = 5
N_UAVS_SYNTHETIC  = 10
N_TIMESTEPS       = 500
FAULT_START_RANGE = (0.5, 0.8)   # fault onset between 50-80% of timeline

# ── Health class definitions ──────────────────────────────────
LABELS  = {0: "Healthy", 1: "Degraded", 2: "Faulty"}
PALETTE = ["#22C55E", "#F59E0B", "#EF4444"]
COLORS  = {"Healthy": "#22C55E", "Degraded": "#F59E0B", "Faulty": "#EF4444"}

# RUL thresholds (cycles) for CMAPSS binning
RUL_FAULTY   = 30    # RUL < 30  → Faulty
RUL_DEGRADED = 100   # RUL < 100 → Degraded

# ── Matplotlib dark theme (applied globally) ──────────────────
PLOT_STYLE = {
    "figure.facecolor": "#060E1A",
    "axes.facecolor":   "#0C1A2E",
    "axes.edgecolor":   "#1E3A5F",
    "axes.labelcolor":  "#6B8CAE",
    "xtick.color":      "#6B8CAE",
    "ytick.color":      "#6B8CAE",
    "text.color":       "#E0EAF5",
    "grid.color":       "#1E3A5F",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
}

GOLD = "#C9A227"

# ── Subsystem definitions (for synthetic generation + EDA) ────
SUBSYSTEMS = {
    "Airframe": {
        "sensors":  ["vibration_rms", "stress_mpa", "fatigue_cycles", "crack_index", "deformation_mm"],
        "units":    ["g", "MPa", "cycles", "index", "mm"],
        "normal":   [(0.1, 1.8),  (20, 120),   (100, 800),   (0.01, 0.08), (0.05, 0.40)],
        "degraded": [(2.0, 4.5),  (130, 180),  (900, 1500),  (0.10, 0.30), (0.50, 1.20)],
        "faulty":   [(5.0, 14.0), (200, 380),  (2000, 5000), (0.40, 1.00), (2.00, 5.00)],
        "noise":    [0.05, 0.03, 0.02, 0.04, 0.03],
        "source":   "real",   # NASA C-MAPSS
    },
    "Hydraulics": {
        "sensors":  ["pressure_bar", "flow_lpm", "temp_c", "fluid_level_pct", "pump_rpm"],
        "units":    ["bar", "L/min", "°C", "%", "RPM"],
        "normal":   [(185, 210), (8.5, 11.5), (42, 58),  (88, 100),  (2850, 3150)],
        "degraded": [(160, 180), (5.5, 8.0),  (62, 78),  (60, 85),   (2200, 2750)],
        "faulty":   [(90,  150), (1.5, 4.5),  (85, 120), (20, 55),   (900,  2000)],
        "noise":    [0.02, 0.04, 0.03, 0.02, 0.02],
        "source":   "real",   # FEMTO Bearing
    },
    "Fuel System": {
        "sensors":  ["fuel_pct", "flow_kgh", "pressure_psi", "temp_c", "contam_index"],
        "units":    ["%", "kg/h", "psi", "°C", "index"],
        "normal":   [(30, 100), (52, 78),   (42, 58), (18, 32), (0.01, 0.04)],
        "degraded": [(15, 28),  (85, 110),  (28, 40), (38, 55), (0.08, 0.25)],
        "faulty":   [(2,  12),  (120, 200), (8,  22), (62, 90), (0.40, 1.00)],
        "noise":    [0.01, 0.03, 0.03, 0.02, 0.05],
        "source":   "synthetic",
    },
    "Li-Ion Battery": {
        "sensors":  ["voltage_v", "current_a", "temp_c", "soc_pct", "resistance_mohm"],
        "units":    ["V", "A", "°C", "%", "mΩ"],
        "normal":   [(49, 53.5), (2,  45),  (22, 33), (25, 100), (11, 18)],
        "degraded": [(44, 48),   (55, 75),  (38, 52), (12, 22),  (22, 45)],
        "faulty":   [(36, 42),   (85, 150), (55, 82), (2,  10),  (55, 110)],
        "noise":    [0.01, 0.03, 0.02, 0.01, 0.04],
        "source":   "real",   # NASA Battery Dataset
    },
    "Electronics": {
        "sensors":  ["cpu_temp_c", "voltage_v", "current_ma", "error_rate", "snr_db"],
        "units":    ["°C", "V", "mA", "rate", "dB"],
        "normal":   [(42, 68),  (4.92, 5.08), (210, 390), (0.001, 0.009), (27, 39)],
        "degraded": [(72, 82),  (4.50, 4.88), (420, 580), (0.020, 0.080), (16, 24)],
        "faulty":   [(88, 112), (3.60, 4.30), (620, 950), (0.120, 0.500), (4,  13)],
        "noise":    [0.03, 0.01, 0.03, 0.05, 0.02],
        "source":   "synthetic",
    },
    "Actuators": {
        "sensors":  ["pos_error_deg", "torque_nm", "temp_c", "resp_time_ms", "duty_pct"],
        "units":    ["°", "Nm", "°C", "ms", "%"],
        "normal":   [(0.05, 0.45), (5.5, 14), (22, 48),  (12, 28),  (22, 58)],
        "degraded": [(0.60, 2.50), (18,  28), (55, 72),  (38, 80),  (65, 78)],
        "faulty":   [(4.00, 14.0), (32,  62), (82, 122), (110, 300),(82, 100)],
        "noise":    [0.04, 0.03, 0.02, 0.04, 0.02],
        "source":   "synthetic",
    },
    "ECS": {
        "sensors":  ["cabin_temp_c", "pressure_diff", "humidity_pct", "fan_rpm", "filter_dp"],
        "units":    ["°C", "Pa", "%", "RPM", "Pa"],
        "normal":   [(19, 24), (0.12, 0.28), (42, 58),  (1550, 1980), (0.12, 0.48)],
        "degraded": [(27, 33), (0.40, 0.70), (65, 78),  (1100, 1480), (0.70, 1.50)],
        "faulty":   [(37, 52), (1.10, 3.00), (82, 100), (450,  980),  (2.20, 5.00)],
        "noise":    [0.02, 0.04, 0.02, 0.02, 0.04],
        "source":   "synthetic",
    },
    "Electrical": {
        "sensors":  ["bus_voltage_v", "current_a", "power_factor", "freq_hz", "insulation_mohm"],
        "units":    ["V", "A", "PF", "Hz", "MΩ"],
        "normal":   [(27.2, 28.8), (11, 29),  (0.92, 1.00), (399, 401), (200, 1000)],
        "degraded": [(25.0, 27.0), (32, 48),  (0.78, 0.90), (394, 398), (30,  90)],
        "faulty":   [(20.0, 24.5), (52, 100), (0.48, 0.72), (382, 393), (2,   18)],
        "noise":    [0.01, 0.02, 0.01, 0.01, 0.03],
        "source":   "synthetic",
    },
}
