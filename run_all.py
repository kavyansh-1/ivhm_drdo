"""
DRDO IVHM — run_all.py  (updated)
Runs the complete 3-week pipeline end-to-end.

Usage:
  python run_all.py              ← full pipeline (days 1-5, 8, 10, 13)
  python run_all.py --day 1 2 3  ← specific days
  python run_all.py --dashboard  ← just launch the dashboard

Pipeline:
  Day 1  → Data generation (real + synthetic)
  Day 2  → EDA + feature engineering
  Day 3  → Classical ML  (RF, XGBoost, GBM)
  Day 4  → Anomaly detection (IF + LOF)
  Day 5  → SHAP explainability
  Day 8  → BiLSTM (health classification + RUL)
  Day 10 → LSTM Autoencoder (deep anomaly detection)
  Day 13 → Ensemble + report generation
"""

import sys, time, logging, subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ALL_DAYS = [1, 2, 3, 4, 5, 8, 10, 13]

DAY_MAP = {
    1:  ("day1_data_gen",      "Data Generation"),
    2:  ("day2_eda",           "EDA + Feature Engineering"),
    3:  ("day3_classical_ml",  "Classical ML (RF + XGB + GBM)"),
    4:  ("day4_anomaly",       "Anomaly Detection (IF + LOF)"),
    5:  ("day5_explainability","SHAP Explainability"),
    8:  ("day8_lstm",          "BiLSTM Multi-Task"),
    10: ("day10_autoencoder",  "LSTM Autoencoder"),
    13: ("day13_ensemble",     "Ensemble + Report"),
}


def run_day(day: int):
    if day not in DAY_MAP:
        log.error("Unknown day: %d. Valid: %s", day, list(DAY_MAP))
        return

    module, label = DAY_MAP[day]
    start = time.time()

    log.info("\n%s", "█" * 62)
    log.info("  DAY %d — %s", day, label)
    log.info("%s\n", "█" * 62)

    try:
        mod = __import__(module)
        mod.run()
    except ImportError as e:
        log.error("Import error: %s", e)
        return
    except Exception as e:
        log.error("Day %d failed: %s", day, e, exc_info=True)
        return

    log.info("\n✅ Day %d complete in %.1fs", day, time.time()-start)


def launch_dashboard():
    dash_path = Path(__file__).parent / "dashboard" / "app.py"
    try:
        import flask  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "flask"], check=True)
    log.info("Dashboard at http://localhost:5000 — Ctrl+C to stop")
    subprocess.run([sys.executable, str(dash_path)])


def print_summary():
    log.info("\n%s", "=" * 62)
    log.info("  DRDO IVHM — Pipeline Complete")
    log.info("%s", "=" * 62)
    outputs = [
        ("data/raw/",              "Real + synthetic CSVs"),
        ("data/processed/",        "Feature-engineered data"),
        ("models/*.pkl",           "RF + XGBoost + GBM models"),
        ("models/lstm/*.pt",       "BiLSTM weights"),
        ("models/autoencoder/*.pt","LSTM Autoencoder weights"),
        ("models/anomaly/*.pkl",   "IF + LOF detectors"),
        ("reports/ivhm_report.json","Dashboard JSON"),
        ("reports/ivhm_summary.csv","Results CSV"),
        ("reports/figures/",       "50+ plot PNGs"),
    ]
    for path, desc in outputs:
        log.info("  %-35s ← %s", path, desc)
    log.info("\n  Launch dashboard: python run_all.py --dashboard")
    log.info("%s", "=" * 62)


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--dashboard" in args:
        launch_dashboard()
        sys.exit(0)

    if "--day" in args:
        idx  = args.index("--day")
        days = [int(d) for d in args[idx+1:] if d.isdigit()]
    else:
        days = ALL_DAYS

    log.info("=" * 62)
    log.info("  DRDO IVHM — Full Pipeline | Days: %s", days)
    log.info("=" * 62)

    total = time.time()
    for day in days:
        run_day(day)

    print_summary()
    log.info("  Total time: %.1fs", time.time() - total)
