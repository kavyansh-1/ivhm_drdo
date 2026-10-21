# DRDO IVHM: Integrated Vehicle Health Management for UAVs

A multi-model machine learning pipeline that monitors 8 UAV subsystems, flags anomalies, classifies health (Healthy / Degraded / Faulty), estimates Remaining Useful Life (RUL), and serves the results on a live web dashboard.

Built as a demo for **DRDO DIA-CoE Research Problem 6: Algorithms for Integrated Vehicle Health Management (IVHM)**, ADE lab, Research Vertical ASDC (Aerospace Systems Design and Characterization). The problem statement's target is a model accuracy above 85%.

---

## Contents

1. [What this does](#what-this-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Datasets](#datasets)
5. [Setup](#setup)
6. [Running the pipeline](#running-the-pipeline)
7. [The models](#the-models)
8. [How the health score is computed](#how-the-health-score-is-computed)
9. [Dashboard](#dashboard)
10. [Outputs](#outputs)
11. [Known limitations](#known-limitations)
12. [Troubleshooting](#troubleshooting)
13. [Roadmap](#roadmap)

---

## What this does

For each of 8 UAV subsystems the pipeline:

1. Loads or generates multi-sensor time-series data
2. Engineers trend features (rolling stats, derivatives, z-scores, lags)
3. Trains supervised classifiers for health state
4. Trains unsupervised anomaly detectors on healthy-only data
5. Trains deep sequence models (BiLSTM and LSTM Autoencoder)
6. Combines everything in a weighted ensemble into one health score per subsystem
7. Explains predictions with SHAP
8. Writes a JSON report that the Flask dashboard displays

**Subsystems:** Airframe, Hydraulics, Fuel System, Li-Ion Battery, Electronics, Actuators, ECS (Environmental Control System), Electrical.

Each subsystem has 5 sensors (for example the battery has voltage, current, temperature, state of charge, internal resistance).

---

## Architecture

```
 Raw sensors (5 per subsystem)
            |
            v
 Day 1   Data loading / generation        -> data/raw/*.csv
            |
            v
 Day 2   Feature engineering (5 -> ~40)   -> data/processed/*.csv
            |
   +--------+-----------------+-------------------+
   v                          v                   v
 Day 3                      Day 8-9             Day 10-11
 RF / XGBoost / GBM         BiLSTM              LSTM Autoencoder
 (snapshot classifier)      (20-step sequence,  (20-step sequence,
                            class + RUL)        healthy-only training)
   |                          |                   |
   |        Day 4: Isolation Forest + LOF         |
   |        (healthy-only anomaly scoring)        |
   +--------------------------+-------------------+
                              v
 Day 13  Weighted ensemble -> health score, status, RUL
                              -> reports/ivhm_report.json
                              v
 Dashboard (Flask + Chart.js) -> http://localhost:5000

 Day 5   SHAP explainability (reads Day 3 models)
```

---

## Project structure

ivhm_drdo/
│
├── dashboard/
│   └── app.py                  # Flask backend + API
│
├── public/
│   └── index.html              # Main dashboard frontend
│
├── reports/
│   ├── figures/                # Generated plots and visualizations
│   ├── ivhm_report.json        # Dashboard/report data
│   └── ivhm_summary.csv        # Per-subsystem results
│
├── src/
│   ├── day1_data_gen.py        # Dataset loading + synthetic generation
│   ├── day2_eda.py             # EDA + feature engineering
│   ├── day3_classical_ml.py    # Random Forest, XGBoost, Gradient Boosting
│   ├── day4_anomaly.py         # Isolation Forest + LOF
│   ├── day5_explainability.py  # SHAP explainability
│   ├── day8_lstm.py            # BiLSTM health + RUL model
│   ├── day10_autoencoder.py    # LSTM Autoencoder
│   └── day13_ensemble.py       # Ensemble + final report
│
├── .gitignore
├── LICENSE
├── config.py                   # Project configuration
├── pyproject.toml              # Python project configuration
├── requirements-ml.txt         # ML dependencies
└── run_all.py                  # Master pipeline runner

## Datasets

| Subsystem | Intended source | Notes |
|---|---|---|
| Airframe | NASA C-MAPSS (FD001) | Turbofan run-to-failure; RUL binned into health classes |
| Li-Ion Battery | NASA Battery Dataset (B0005, B0006, B0007, B0018) | Health from capacity fade (<70% = Faulty) |
| Hydraulics | FEMTO-PRONOSTIA bearings | Vibration run-to-failure used as a proxy for pump wear |
| Fuel System, Electronics, Actuators, ECS, Electrical | Synthetic | No suitable public dataset; physics-inspired sigmoid degradation |

**Important:** the loaders fall back to synthetic data automatically if a real dataset is not found. The Day 1 log prints a `REAL` / `SYNTH` tag based on the *intended* source, not on what actually loaded. Scroll up in the Day 1 output and look for `not found` warnings to confirm which datasets really loaded. If every subsystem shows exactly 5,000 rows (10 UAVs x 500 steps), the real datasets were not picked up. See [Known limitations](#known-limitations).

### Downloading the real datasets (Windows PowerShell)

PowerShell's `wget` needs `-OutFile` and `-UseBasicParsing`, otherwise it only prints the response and saves nothing.

```powershell
wget "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -OutFile "data\real\CMAPSS\cmapss.zip" -UseBasicParsing
wget "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip" -OutFile "data\real\Battery\battery.zip" -UseBasicParsing
wget "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip" -OutFile "data\real\FEMTO\femto.zip" -UseBasicParsing

Expand-Archive data\real\CMAPSS\cmapss.zip -DestinationPath data\real\CMAPSS\
Expand-Archive data\real\Battery\battery.zip -DestinationPath data\real\Battery\
Expand-Archive data\real\FEMTO\femto.zip -DestinationPath data\real\FEMTO\
```

The extracted folders have long names (for example `6. Turbofan Engine Degradation Simulation Data Set`). The loaders search recursively, so you do not need to move files. Note that the NASA battery archive may contain nested zip files that need a second extraction.

Other public UAV datasets worth exploring: RflyMAD, Multi-Fault Multirotor UAV dataset (IEEE DataPort), DronePropA.

---

## Setup

Python 3.10+ is recommended.

```powershell
cd C:\Users\Asus\documents\ivhm_drdo
pip install -r requirements.txt

# PyTorch (choose one)
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121    # NVIDIA GPU
```

**Required, to avoid the pipeline hanging on plot windows:** make sure the top of `config.py` contains

```python
import matplotlib
matplotlib.use('Agg')
```

and that the scripts do not call `plt.show()` in a way that blocks. Without the `Agg` backend a matplotlib window opens and the script waits until you close it.

---

## Running the pipeline

Run from the project root, one window, in order:

```powershell
python day1_data_gen.py
python day2_eda.py
python day3_classical_ml.py
python day4_anomaly.py
python day5_explainability.py   # needs Day 3 models
python day8_lstm.py             # needs Day 2 data
python day10_autoencoder.py     # needs Day 2 data
python day13_ensemble.py        # uses whatever models exist
python dashboard/app.py         # then open http://localhost:5000
```

Or use the runner:

```powershell
python run_all.py               # days 1-5, 8, 10, 13
python run_all.py --day 1 2 3   # selected days
python run_all.py --dashboard   # launch dashboard only
```

Approximate time on a laptop CPU: 35-45 minutes end to end. Days 8 and 10 are the slowest.

Day 13 skips any model that is missing, so you can run it after Day 3 alone. The dashboard shows built-in demo data if `reports/ivhm_report.json` does not exist yet.

---

## The models

### Feature engineering (Day 2)

For every raw sensor `S`, seven features are added: `S_roll_mean`, `S_roll_std`, `S_diff1`, `S_diff2`, `S_zscore`, `S_lag5`, `S_lag10`. Rolling window is 10 steps. Features are computed per UAV so values never leak across aircraft. This takes 5 sensors to roughly 40 features per subsystem.

### Supervised classifiers (Day 3)

| Model | Input | Decision basis |
|---|---|---|
| Random Forest | One snapshot of ~40 features | 200 trees vote on feature thresholds |
| XGBoost | Same | 200 sequential trees, each correcting the previous errors |
| Gradient Boosting | Same | sklearn boosting, 150 trees |

The best model per subsystem (by test F1) is saved to `models/<Subsystem>.pkl`. The train/test split is **by UAV**, not by row, so adjacent timesteps from one aircraft never appear on both sides of the split. Cross-validation is 5-fold stratified.

### Unsupervised anomaly detection (Day 4)

Isolation Forest and Local Outlier Factor, both fitted on **healthy samples only**. Scores are normalised to 0-1 and averaged. Isolation Forest catches global outliers; LOF catches locally unusual points.

### BiLSTM (Day 8-9)

- Input: 20 consecutive timesteps x ~40 features
- Two stacked bidirectional LSTM layers (128 then 64 units), dropout 0.3, batch norm
- Two heads: health class (3 logits) and normalised RUL (sigmoid)
- Loss: cross-entropy + 0.3 x MSE on RUL; Adam, cosine schedule, early stopping (patience 8)

### LSTM Autoencoder (Day 10-11)

- Trained only on healthy 20-step windows
- Encoder (BiLSTM) to a 32-dim latent vector, decoder (LSTM) reconstructs the window
- Anomaly = high reconstruction error; threshold = mean + 2 x std on a healthy validation set

### Explainability (Day 5)

SHAP `TreeExplainer` on the saved tree models: beeswarm plots (Faulty class) and mean |SHAP| bar charts per subsystem.

---

## How the health score is computed

Day 13 combines the available models:

```
ensemble_probs = 0.5 x classical + 0.3 x BiLSTM + 0.2 x Autoencoder
                 (weights are renormalised if a model is missing)

health_score   = (1 - P(Faulty)) x max_class_probability x 100
status         = most frequent predicted class across the subsystem's timesteps
RUL (hours)    = max(0, 5.5 x avg_health - 50)      # heuristic, see limitations
```

The Autoencoder's reconstruction error is converted to pseudo-probabilities with a fixed mapping, so its contribution is an approximation rather than a calibrated probability.

---

## Dashboard

Flask backend + Chart.js frontend.

| Endpoint | Purpose |
|---|---|
| `GET /` | Dashboard page |
| `GET /api/report` | Full report JSON from Day 13 |
| `GET /api/live` | Report with +/-2% random jitter, simulating telemetry updates |
| `POST /api/predict` | Single-sample inference: `{"subsystem": "ECS", "sensors": {...}}` |

The page shows overall system health, anomaly count, critical systems, one card per subsystem (health, status, RUL, anomaly count), an alert log, and per-subsystem detail tabs.

---

## Outputs

| Path | Content |
|---|---|
| `data/raw/*.csv` | Raw sensor data per subsystem |
| `data/processed/*.csv` | Feature-engineered data; `*_health.csv` has anomaly and health scores |
| `models/*.pkl` | Best classical model + scaler + feature list per subsystem |
| `models/lstm/`, `models/autoencoder/`, `models/anomaly/` | Deep and anomaly models |
| `reports/ivhm_report.json` | Dashboard feed |
| `reports/ivhm_summary.csv` | Per-subsystem results table |
| `reports/figures/` | Distributions, PCA, confusion matrices, ROC curves, SHAP, fleet heatmap, training curves |

---

## Known limitations

Read these before presenting the results.

1. **Real vs synthetic data.** Check that the real datasets actually loaded (see [Datasets](#datasets)). If they fell back to synthetic, results describe the synthetic generator, not NASA data. Synthetic data is built with well-separated normal / degraded / faulty ranges, so classifiers can score near-perfect accuracy on it. That reflects the simulator's design and is not evidence of real-world performance.
2. **Health labels are derived, not measured.** Labels come from RUL bins, capacity fade, life fraction, or the degradation curve. A model that reproduces them is reproducing our labelling rule.
3. **RUL is a heuristic.** The `5.5 x health - 50` formula in Day 13 is a rough mapping, not a trained regressor. The BiLSTM RUL head uses a proxy target when no explicit RUL column exists (only C-MAPSS has one). Do not quote RUL values as validated predictions.
4. **Hydraulics and Airframe are proxies.** Bearing vibration stands in for hydraulic pump wear, and turbofan engine sensors stand in for airframe health. Several hydraulic channels (pressure, flow, temperature) are synthesised from bearing life fraction rather than measured.
5. **Dashboard motion is simulated.** The 24-hour trend chart and the `/api/live` jitter are generated for demonstration; there is no live telemetry source yet.
6. **The ~70% health everywhere reflects class mix.** Each subsystem score averages over all UAVs and timesteps, including the degraded and faulty portions, so scores cluster near the healthy fraction of the data. Per-UAV scores in the JSON are more informative.
7. **The ensemble weights (0.5 / 0.3 / 0.2) are hand-chosen**, not tuned on a validation set.
8. **No held-out evaluation of the full ensemble.** Day 13 scores the same processed data the models were trained on, so the ensemble numbers are not a clean test-set measurement. Per-model test metrics from Day 3 and Day 8 are the honest ones.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Script hangs at "Generating plots" | Add `matplotlib.use('Agg')` at the top of `config.py`, or close the hidden plot window |
| `wget` finishes but no file appears | Use `-OutFile <path> -UseBasicParsing` |
| `C-MAPSS train_FD001.txt not found` | Check extraction; files must exist somewhere under `data/real/CMAPSS/` |
| All subsystems have exactly 5,000 rows | Real datasets were not found; synthetic fallback is in use |
| `ModuleNotFoundError: torch` | Install PyTorch (see Setup) |
| Day 13 says "No results" | Run Days 1-3 first |
| `lof.score_samples` error on older scikit-learn | Upgrade to scikit-learn >= 1.3 (LOF is created with `novelty=True`) |
| SHAP fails for a subsystem | Make sure the Day 3 `.pkl` exists; SHAP needs a tree model |

---

## Roadmap

- Verify and document exactly which datasets loaded; add a run summary file that records the data source per subsystem
- Replace the RUL heuristic with a trained RUL regressor evaluated on held-out C-MAPSS engines
- Evaluate the ensemble on a strict held-out UAV split and tune the weights
- Add the Multi-Fault UAV / RflyMAD datasets for real flight-controller and motor signals
- Stream real or replayed telemetry into the dashboard instead of simulated jitter
- Write the technical report and slide deck

---

## References

- DRDO, *Research Problems for DIA-CoEs*, September 2026, Directorate of Futuristic Technology Management (Problem 6, ADE)
- NASA Prognostics Center of Excellence data repository (C-MAPSS, Battery)
- FEMTO-ST PRONOSTIA bearing dataset (IEEE PHM 2012 Challenge)
- Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP)
