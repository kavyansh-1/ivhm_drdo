# DRDO IVHM: Integrated Vehicle Health Management for UAVs

A multi-model machine learning pipeline that monitors **8 UAV subsystems**, detects anomalies, classifies health states (`Healthy`, `Degraded`, or `Faulty`), estimates Remaining Useful Life (RUL), and presents the results through a web dashboard.

Built as a demonstration for **DRDO DIA-CoE Research Problem 6: Algorithms for Integrated Vehicle Health Management (IVHM)**, ADE Lab, Research Vertical ASDC (Aerospace Systems Design and Characterization). The problem statement targets model accuracy above 85%.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Models and Methodology](#models-and-methodology)
- [Health Score Calculation](#health-score-calculation)
- [Dashboard](#dashboard)
- [Outputs](#outputs)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [References](#references)

## What This Project Does

For each of the eight UAV subsystems, the pipeline:

1. Loads public datasets or generates synthetic sensor time-series data.
2. Engineers temporal features, including rolling statistics, derivatives, z-scores, and lag values.
3. Trains supervised machine learning classifiers to predict health states.
4. Trains unsupervised anomaly detectors using healthy-only data.
5. Trains deep sequence models (BiLSTM and LSTM Autoencoder).
6. Combines available model outputs into an ensemble health assessment.
7. Uses SHAP to explain predictions from tree-based models.
8. Generates JSON and CSV reports for analysis and dashboard visualization.

### Monitored Subsystems

| Subsystem | Example sensor measurements |
|---|---|
| Airframe | Structural load, strain, vibration, temperature, stress |
| Hydraulics | Pressure, flow, temperature, vibration, pump condition |
| Fuel System | Fuel pressure, flow, temperature, level, pump condition |
| Li-Ion Battery | Voltage, current, temperature, state of charge, internal resistance |
| Electronics | Board temperature, voltage, current, signal quality, component condition |
| Actuators | Position, command error, current, temperature, vibration |
| ECS (Environmental Control System) | Temperature, pressure, airflow, humidity, cooling performance |
| Electrical | Voltage, current, frequency, temperature, power condition |

*Note: These are illustrative sensor categories. Actual channels depend on the selected dataset or synthetic data generator.*

## Architecture

The project follows a staged pipeline, from sensor data ingestion to health assessment and visualization.

```text
                    RAW SENSOR DATA
                           |
                           v
              +-------------------------+
              | Day 1: Data Loading      |
              | and Synthetic Generation|
              +-------------------------+
                           |
                           v
              +-------------------------+
              | Day 2: EDA and Feature   |
              | Engineering              |
              | 5 sensors -> ~40 features|
              +-------------------------+
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
      +-------------+ +-----------+ +----------------+
      | Day 3       | | Day 4     | | Day 8-9        |
      | Classical ML| | Anomaly   | | BiLSTM         |
      | RF, XGBoost,| | Detection | | Health + RUL   |
      | Gradient    | | IF + LOF  | |                |
      | Boosting    | |           | |                |
      +-------------+ +-----------+ +----------------+
             |             |             |
             |             |      +----------------+
             |             |      | Day 10-11      |
             |             |      | LSTM           |
             |             |      | Autoencoder    |
             |             |      +----------------+
             |             |             |
             +-------------+-------------+
                           |
                           v
              +-------------------------+
              | Day 13: Weighted Ensemble|
              | Health Score, Status, RUL|
              +-------------------------+
                           |
                           v
              +-------------------------+
              | JSON and CSV Reports    |
              +-------------------------+
                           |
                           v
              +-------------------------+
              | Flask + Chart.js        |
              | Web Dashboard           |
              +-------------------------+
                           |
                           v
                 http://localhost:5000
```

**Pipeline stages:**

- **Day 1:** Load available datasets or generate synthetic sensor data.
- **Day 2:** Perform exploratory data analysis (EDA) and feature engineering.
- **Day 3:** Train and evaluate classical supervised classifiers.
- **Day 4:** Train anomaly detection models on healthy samples.
- **Day 5:** Generate SHAP-based explanations for tree models.
- **Days 8–9:** Train the BiLSTM health classification and RUL model.
- **Days 10–11:** Train the LSTM Autoencoder for anomaly detection.
- **Day 13:** Combine available model outputs and generate final reports.

## Project Structure

The repository contains the source code, configuration, and dashboard implementation. **Large datasets, trained model artifacts, and generated outputs are maintained locally and excluded from Git version control because of their size.**

```text
ivhm_drdo/
│
├── dashboard/
│   └── app.py                    # Flask backend and REST API
│
├── public/
│   └── index.html                # Dashboard frontend (Chart.js)
│
├── data/                         # Local datasets (excluded from Git)
│   ├── raw/                      # Raw sensor time-series data
│   ├── processed/                # Engineered features and health data
│   └── real/                     # Downloaded public datasets
│       ├── CMAPSS/
│       ├── Battery/
│       └── FEMTO/
│
├── models/                       # Locally trained models (excluded from Git)
│   ├── *.pkl                     # Classical ML models and preprocessing objects
│   ├── lstm/                     # Trained BiLSTM models
│   ├── autoencoder/              # Trained LSTM Autoencoders
│   └── anomaly/                  # Trained anomaly detection models
│
├── reports/                      # Generated outputs (excluded from Git)
│   ├── figures/                  # Plots and visualizations
│   ├── ivhm_report.json          # Dashboard report data
│   └── ivhm_summary.csv          # Per-subsystem summary
│
├── src/
│   ├── day1_data_gen.py           # Dataset loading and synthetic generation
│   ├── day2_eda.py                # EDA and feature engineering
│   ├── day3_classical_ml.py       # Random Forest, XGBoost, Gradient Boosting
│   ├── day4_anomaly.py            # Isolation Forest and LOF
│   ├── day5_explainability.py     # SHAP explainability
│   ├── day8_lstm.py               # BiLSTM health classification and RUL
│   ├── day10_autoencoder.py       # LSTM Autoencoder anomaly detection
│   └── day13_ensemble.py          # Ensemble and report generation
│
├── .gitignore                    # Excludes large and generated artifacts
├── LICENSE
├── config.py                     # Shared project configuration
├── pyproject.toml                # Python project configuration
├── requirements-ml.txt           # Machine learning dependencies
└── run_all.py                    # Master pipeline runner
```

### Dataset and Model Availability

The `data/`, `models/`, and generated `reports/` directories are maintained locally and are not committed to the repository.

- **Datasets:** Public datasets are downloaded separately and loaded by the pipeline. Synthetic data can be generated automatically when the required real data is unavailable.
- **Trained models:** Classical ML models, BiLSTM weights, Autoencoder weights, and anomaly detection models are generated and stored locally.
- **Generated outputs:** Reports, CSV summaries, and visualizations are produced during pipeline execution.

The repository therefore documents the complete project structure without requiring large datasets or model files to be uploaded to GitHub. These files must be downloaded, generated, or otherwise obtained separately when setting up the project on another machine.

*The directory tree represents the intended local project layout. Some files and directories are created only after running the corresponding pipeline stages.*

## Datasets

The pipeline uses public prognostics datasets where available and synthetic data for subsystems without suitable public datasets.

| Subsystem | Intended dataset or source | Usage |
|---|---|---|
| Airframe | NASA C-MAPSS FD001 | Turbofan run-to-failure data used as an airframe-health proxy |
| Li-Ion Battery | NASA Battery Dataset: B0005, B0006, B0007, B0018 | Battery degradation and capacity fade |
| Hydraulics | FEMTO-ST PRONOSTIA bearing dataset | Bearing degradation used as a hydraulic pump-wear proxy |
| Fuel System | Synthetic | Physics-inspired degradation simulation |
| Electronics | Synthetic | Physics-inspired degradation simulation |
| Actuators | Synthetic | Physics-inspired degradation simulation |
| ECS | Synthetic | Physics-inspired degradation simulation |
| Electrical | Synthetic | Physics-inspired degradation simulation |

### Important: Verify the Actual Data Source

The loaders automatically fall back to synthetic data when a real dataset cannot be found.

The Day 1 `REAL` / `SYNTH` log tag reflects the intended source, not necessarily the data that was ultimately loaded. Check the preceding warnings to verify whether the real files were found.

If every subsystem contains exactly 5,000 rows (10 UAVs × 500 timesteps), the pipeline is likely using synthetic fallback data.

### Downloading the Public Datasets (Windows PowerShell)

Run these commands from the project root. Create the destination directories first if they do not already exist.

```powershell
New-Item -ItemType Directory -Force -Path `
    "data\real\CMAPSS", `
    "data\real\Battery", `
    "data\real\FEMTO"
```

Download the archives:

```powershell
Invoke-WebRequest `
  -Uri "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" `
  -OutFile "data\real\CMAPSS\cmapss.zip"

Invoke-WebRequest `
  -Uri "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip" `
  -OutFile "data\real\Battery\battery.zip"

Invoke-WebRequest `
  -Uri "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip" `
  -OutFile "data\real\FEMTO\femto.zip"
```

Extract the archives:

```powershell
Expand-Archive -Path "data\real\CMAPSS\cmapss.zip" -DestinationPath "data\real\CMAPSS" -Force
Expand-Archive -Path "data\real\Battery\battery.zip" -DestinationPath "data\real\Battery" -Force
Expand-Archive -Path "data\real\FEMTO\femto.zip" -DestinationPath "data\real\FEMTO" -Force
```

The extracted directories may have long names, such as `6. Turbofan Engine Degradation Simulation Data Set`. The loaders are intended to search recursively, so moving the files should not be necessary. The NASA Battery archive may contain nested ZIP files that require additional extraction.

Other public UAV datasets worth exploring include RflyMAD, the Multi-Fault Multirotor UAV dataset available through IEEE DataPort, and DronePropA.

## Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git (optional, for cloning the repository)
- PyTorch for deep learning stages
- An NVIDIA GPU is optional; CPU execution is supported

### 1. Navigate to the Project

```powershell
cd C:\Users\Asus\Documents\ivhm_drdo
```

### 2. Install Dependencies

If your repository contains `requirements.txt`:

```powershell
pip install -r requirements.txt
```

If your repository contains only `requirements-ml.txt`:

```powershell
pip install -r requirements-ml.txt
```

### 3. Install PyTorch

Choose the installation appropriate for your hardware.

**CPU:**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU with a compatible CUDA 12.1 setup:**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, consult the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Configure Matplotlib

To prevent scripts from blocking while waiting for plot windows to close, ensure that `config.py` selects a non-interactive Matplotlib backend before importing `matplotlib.pyplot`:

```python
import matplotlib
matplotlib.use("Agg")
```

Avoid blocking calls to `plt.show()` during automated pipeline execution.

## Running the Pipeline

Run commands from the project root, in the order shown below.

### Option 1: Run Each Stage Individually

```powershell
python src/day1_data_gen.py
python src/day2_eda.py
python src/day3_classical_ml.py
python src/day4_anomaly.py
python src/day5_explainability.py
python src/day8_lstm.py
python src/day10_autoencoder.py
python src/day13_ensemble.py
python dashboard/app.py
```

Then open [http://localhost:5000](http://localhost:5000).

*These commands assume the scripts are inside `src/`, as shown in the project tree. Adjust the paths if your actual repository layout differs.*

### Option 2: Use the Master Runner

```powershell
# Run all configured pipeline stages
python run_all.py

# Run selected stages
python run_all.py --day 1 2 3

# Launch only the dashboard
python run_all.py --dashboard
```

The master runner is intended to execute Days 1–5, 8, 10, and 13.

### Expected Runtime

The approximate end-to-end runtime on a laptop CPU is **35–45 minutes**, depending on hardware, dataset size, and model parameters. Days 8 and 10 are expected to be the slowest.

Day 13 can skip missing models, allowing a partial ensemble to be generated after the classical models have been trained. If `reports/ivhm_report.json` does not exist, the dashboard displays built-in demonstration data.

## Models and Methodology

### Feature Engineering (Day 2)

For each raw sensor \(S\), the pipeline derives seven temporal features:

- `S_roll_mean` — rolling mean
- `S_roll_std` — rolling standard deviation
- `S_diff1` — first difference
- `S_diff2` — second difference
- `S_zscore` — standardized value
- `S_lag5` — value from five timesteps earlier
- `S_lag10` — value from ten timesteps earlier

The rolling window contains 10 timesteps. Features are calculated separately for each UAV to prevent temporal features from crossing aircraft boundaries.

Five raw sensors produce approximately 40 features per subsystem, including the original sensor channels.

### Supervised Classification (Day 3)

The pipeline compares three classical machine learning models.

| Model | Input | Decision basis |
|---|---|---|
| Random Forest | One snapshot of approximately 40 features | An ensemble of 200 decision trees |
| XGBoost | Same feature vector | 200 boosting rounds, with each stage correcting earlier errors |
| Gradient Boosting | Same feature vector | Scikit-learn boosting with 150 estimators |

The best model for each subsystem is selected using test-set F1 score and saved under `models/<Subsystem>.pkl`.

The intended train/test split is by UAV rather than by individual row, preventing adjacent timesteps from the same aircraft from appearing in both sets. Five-fold stratified cross-validation is also used.

### Anomaly Detection (Day 4)

Two unsupervised methods are trained using healthy samples only:

- **Isolation Forest:** Detects observations that are unusually easy to isolate from the rest of the data.
- **Local Outlier Factor (LOF):** Identifies observations with unusually low local density compared with their neighbors.

Their normalized anomaly scores are averaged to obtain a combined anomaly score.

### Explainability (Day 5)

SHAP (SHapley Additive exPlanations) is applied to saved tree-based models using `TreeExplainer`.

Generated visualizations include:

- Beeswarm plots for the `Faulty` class
- Mean absolute SHAP feature-importance charts for each subsystem

These plots help identify which features contributed most to the model's predictions.

### BiLSTM (Days 8–9)

The Bidirectional Long Short-Term Memory (BiLSTM) network analyzes sequences rather than isolated snapshots.

**Architecture and training:**

- Input: 20 consecutive timesteps × approximately 40 features
- Two stacked bidirectional LSTM layers with 128 and 64 units
- Dropout: 0.3
- Batch normalization
- Two output heads:
  - Health classification: three logits
  - Normalized RUL: sigmoid output
- Loss: cross-entropy + 0.3 × mean squared error for RUL
- Optimizer: Adam
- Learning-rate schedule: cosine schedule
- Early stopping: patience of 8 epochs

The model jointly learns health-state classification and a remaining-life target. Where explicit RUL labels are unavailable, the target is a proxy and must not be treated as validated remaining-life estimation.

### LSTM Autoencoder (Days 10–11)

The LSTM Autoencoder learns patterns found in healthy operating sequences.

**Architecture and training:**

1. Train using healthy 20-timestep windows.
2. Encode each window into a 32-dimensional latent representation using a bidirectional LSTM encoder.
3. Reconstruct the input sequence using an LSTM decoder.
4. Calculate reconstruction error between the input and reconstructed sequence.
5. Flag sequences with unusually high reconstruction error as potential anomalies.

The anomaly threshold is calculated as:

\[
\text{Threshold} = \mu_{\text{healthy}} + 2\sigma_{\text{healthy}}
\]

Here, \(\mu_{\text{healthy}}\) and \(\sigma_{\text{healthy}}\) are the mean and standard deviation of reconstruction error on a healthy validation set.

## Health Score Calculation

Day 13 combines predictions from the available models into a final health assessment.

### 1. Ensemble Probabilities

The initial weights are:

\[
P_{\text{ensemble}} =
0.5P_{\text{classical}} +
0.3P_{\text{BiLSTM}} +
0.2P_{\text{autoencoder}}
\]

Weights are renormalized when one or more models are unavailable.

The Autoencoder's reconstruction error is converted to pseudo-probabilities using a fixed mapping. Its contribution is therefore approximate rather than a calibrated probability estimate.

### 2. Health Score

The reported score is:

\[
\text{Health Score} =
(1-P(\text{Faulty}))\times
P(\text{predicted class})_{\max}\times 100
\]

The score combines the probability of not being faulty with the maximum predicted class probability.

### 3. Health Status

The subsystem status is assigned using the most frequently predicted class across its timesteps:

- `Healthy`
- `Degraded`
- `Faulty`

### 4. Remaining Useful Life (RUL)

The current ensemble uses the following heuristic:

\[
\text{RUL (hours)} = \max(0,\;5.5\times \text{avg\_health}-50)
\]

This is a rough mapping rather than a trained and validated RUL estimator. The reported value should be treated as a demonstration output, not a reliable operational prediction.

## Dashboard

The project includes a Flask backend and a Chart.js frontend.

### Dashboard Features

- Overall system health
- Anomaly count
- Critical subsystem overview
- Individual cards for all eight subsystems
- Health score, status, RUL, and anomaly count per subsystem
- Alert log
- Per-subsystem detail tabs
- A demonstration trend chart

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the dashboard page |
| `GET` | `/api/report` | Returns the complete report generated by Day 13 |
| `GET` | `/api/live` | Returns the report with ±2% random jitter to simulate telemetry updates |
| `POST` | `/api/predict` | Performs single-sample inference |

Example request for `/api/predict`:

```json
{
  "subsystem": "ECS",
  "sensors": {
    "temperature": 35.0,
    "pressure": 101.0
  }
}
```

*The sensor names and values are illustrative. Use the exact sensor keys expected by the selected subsystem's model.*

Start the dashboard:

```powershell
python dashboard/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Important:** The current dashboard does not receive live aircraft telemetry. The `/api/live` endpoint and 24-hour trend chart use simulated updates for demonstration.

## Outputs

The pipeline generates the following artifacts locally.

| Path | Contents |
|---|---|
| `data/raw/*.csv` | Raw sensor time-series data |
| `data/processed/*.csv` | Feature-engineered datasets |
| `data/processed/*_health.csv` | Health and anomaly-related scores, where generated |
| `models/*.pkl` | Classical models, scalers, and feature metadata |
| `models/lstm/` | Trained BiLSTM artifacts |
| `models/autoencoder/` | Trained Autoencoder artifacts |
| `models/anomaly/` | Saved anomaly detection models |
| `reports/ivhm_report.json` | Dashboard and report data |
| `reports/ivhm_summary.csv` | Per-subsystem summary |
| `reports/figures/` | Distributions, PCA, confusion matrices, ROC curves, SHAP plots, fleet heatmaps, and training curves |

These artifacts are not committed to Git. They must be generated locally or obtained separately before running the full pipeline with trained models.

## Known Limitations

Read these limitations before interpreting or presenting the results.

### 1. Real vs. Synthetic Data

The loaders can fall back to synthetic data when real datasets are missing. Synthetic data uses deliberately separated healthy, degraded, and faulty ranges, which can produce near-perfect classification results.

High accuracy on synthetic data is not evidence of real-world UAV performance.

### 2. Derived Health Labels

Health labels are derived from rules such as RUL bins, battery capacity fade, life fraction, or a synthetic degradation curve. A model may simply learn to reproduce these labeling rules rather than discover independently verified faults.

### 3. RUL Is Not Yet Validated

The ensemble RUL formula is heuristic. The BiLSTM also uses a proxy target when explicit RUL data is unavailable. RUL values should not be quoted as validated predictions of actual component lifetime.

### 4. Airframe and Hydraulics Are Proxies

NASA C-MAPSS turbofan data is used as a proxy for airframe health. FEMTO-ST bearing data is used as a proxy for hydraulic pump wear. Several hydraulic channels, including pressure, flow, and temperature, are synthesized from bearing life fraction rather than directly measured.

These proxies do not fully represent real UAV subsystem behavior.

### 5. Dashboard Telemetry Is Simulated

The `/api/live` endpoint adds random jitter to report values, and the 24-hour trend chart is generated for demonstration. No real telemetry source is connected yet.

### 6. Aggregate Health Scores Reflect Dataset Composition

Subsystem scores near 70% can result from averaging across all UAVs and timesteps, including degraded and faulty periods. These aggregate scores do not necessarily represent the condition of a particular aircraft. Per-UAV results in the JSON report may provide more useful detail.

### 7. Ensemble Weights Are Hand-Chosen

The 0.5 / 0.3 / 0.2 ensemble weights have not been optimized on an independent validation set.

### 8. Full-Ensemble Evaluation Is Not Held Out

Day 13 scores the processed data available to the ensemble rather than performing a strict held-out evaluation of the complete pipeline. As a result, its aggregate scores are not an unbiased test-set measurement.

Per-model test metrics from Day 3 and Day 8 are more informative, provided the split is genuinely independent and preprocessing avoids data leakage.

### 9. The 85% Target Requires Independent Validation

The DRDO problem statement targets accuracy above 85%. This project should not claim to have met that target based solely on synthetic-data accuracy or in-sample ensemble results.

A credible performance claim requires clearly documented data sources, held-out evaluation, suitable metrics, and preferably validation on real UAV sensor data.

## Troubleshooting

| Problem | Possible solution |
|---|---|
| Script hangs while generating plots | Set `matplotlib.use("Agg")` before importing `pyplot` and avoid blocking `plt.show()` calls. |
| Download command finishes without creating a file | Use `Invoke-WebRequest -OutFile` and verify that the destination directory exists. |
| `train_FD001.txt` is not found | Check extraction and confirm the file exists somewhere under `data/real/CMAPSS/`. |
| Every subsystem has exactly 5,000 rows | Check whether the loaders fell back to synthetic data. |
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch using the appropriate CPU or CUDA command. |
| Day 13 reports no results | Run Days 1–3 first and confirm their expected output files exist. |
| `lof.score_samples` fails on an older scikit-learn version | Upgrade to scikit-learn 1.3 or later and verify LOF is configured with `novelty=True`. |
| SHAP fails for a subsystem | Confirm the corresponding Day 3 tree model exists and is compatible with SHAP. |
| Dashboard displays demo data instead of pipeline results | Run Day 13 and verify `reports/ivhm_report.json` exists at the expected path. |
| Pipeline cannot locate modules or files | Confirm commands are run from the repository root and that script paths match the actual directory structure. |

## Roadmap

- [ ] Verify and document the actual dataset loaded for each subsystem.
- [ ] Generate a run summary recording data source, row count, and preprocessing details.
- [ ] Replace the RUL heuristic with a trained regressor evaluated on held-out C-MAPSS engines.
- [ ] Evaluate the complete ensemble on a strict held-out UAV split.
- [ ] Tune ensemble weights using validation data.
- [ ] Integrate the Multi-Fault UAV and RflyMAD datasets for real flight-controller and motor signals.
- [ ] Replace simulated dashboard updates with real or replayed telemetry.
- [ ] Validate the complete pipeline against the DRDO accuracy target using independent evaluation.
- [ ] Prepare the technical report and presentation deck.

## References

1. **DRDO:** *Research Problems for DIA-CoEs*, September 2026. Directorate of Futuristic Technology Management. Problem 6: Algorithms for Integrated Vehicle Health Management (IVHM), ADE.
2. **NASA Prognostics Center of Excellence:** C-MAPSS Turbofan Engine Degradation Simulation and Battery datasets.
3. **FEMTO-ST Institute:** PRONOSTIA bearing degradation dataset, IEEE PHM 2012 Data Challenge.
4. **Lundberg, S. M., and Lee, S.-I.** “A Unified Approach to Interpreting Model Predictions.” NeurIPS, 2017. SHAP.

---

**Project status:** Research demonstration and prototype. Results depend on the datasets actually loaded, the evaluation procedure, and the availability of trained models. Real-world UAV validation remains future work.

