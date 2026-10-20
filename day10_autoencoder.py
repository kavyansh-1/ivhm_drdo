"""
DRDO IVHM — Day 10-11: LSTM Autoencoder for Deep Anomaly Detection
Trained ONLY on healthy data. Anomaly = high reconstruction error.
Architecture: LSTM Encoder → Latent → LSTM Decoder → Reconstruction
Threshold = mean + 2*std of reconstruction error on healthy validation set.
Compares with Day 4 IF+LOF results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from config import (SUBSYSTEMS, DATA_PROCESSED, MODELS_DIR,
                    FIGURES_DIR, LABELS, PALETTE, PLOT_STYLE, GOLD, RANDOM_SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
matplotlib.rcParams.update(PLOT_STYLE)

AE_DIR = MODELS_DIR / "autoencoder"
AE_DIR.mkdir(exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 20
LATENT   = 32
EPOCHS   = 40
BATCH    = 128
LR       = 1e-3


# ════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════

class LSTMAutoencoder(nn.Module):
    """
    Encoder compresses sequence → fixed latent vector.
    Decoder reconstructs sequence from latent vector.
    High reconstruction error → sample is anomalous (unlike training distribution).
    """
    def __init__(self, n_features: int, latent_dim: int = LATENT,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True,
                                bidirectional=True, dropout=dropout)
        self.enc_fc  = nn.Linear(hidden * 2, latent_dim)

        self.dec_fc  = nn.Linear(latent_dim, hidden)
        self.decoder = nn.LSTM(hidden, n_features, batch_first=True)

        self.seq_len = SEQ_LEN
        self.hidden  = hidden

    def forward(self, x):                          # x: (B, T, F)
        # Encode
        enc_out, _ = self.encoder(x)               # (B, T, 2H)
        latent     = self.enc_fc(enc_out[:, -1, :])# (B, latent)

        # Decode: repeat latent T times → feed into decoder
        dec_in  = self.dec_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        recon, _ = self.decoder(dec_in)            # (B, T, F)
        return recon, latent


# ════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════

def get_feature_cols(df):
    skip = {"health_label","health_status","uav_id","timestep",
            "subsystem","rul","capacity_ah","life_fraction","fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in skip]


def make_windows(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    return np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])


def prepare_data(name: str):
    path = DATA_PROCESSED / f"{name.replace(' ','_')}.csv"
    if not path.exists():
        return None

    df        = pd.read_csv(path).sort_values(["uav_id","timestep"])
    feat_cols = get_feature_cols(df)

    scaler = StandardScaler()
    X      = scaler.fit_transform(df[feat_cols].fillna(0))
    y      = df["health_label"].values

    # Train autoencoder ONLY on healthy samples
    healthy_mask = y == 0
    X_healthy    = X[healthy_mask]
    y_healthy    = y[healthy_mask]

    # Sliding windows
    X_h_win = make_windows(X_healthy)
    X_all_win = make_windows(X)
    y_win     = y[SEQ_LEN:]

    # Val split from healthy
    n_val      = max(20, int(len(X_h_win) * 0.15))
    idx        = np.random.permutation(len(X_h_win))
    train_idx  = idx[n_val:]
    val_idx    = idx[:n_val]

    X_train = torch.tensor(X_h_win[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X_h_win[val_idx],   dtype=torch.float32)
    X_all   = torch.tensor(X_all_win,           dtype=torch.float32)

    return X_train, X_val, X_all, y_win, scaler, feat_cols


# ════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════

def train_ae(X_train: torch.Tensor, X_val: torch.Tensor, n_features: int):
    model     = LSTMAutoencoder(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    loader    = DataLoader(TensorDataset(X_train), batch_size=BATCH, shuffle=True)

    best_val, best_state = float("inf"), None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = sum(
            criterion(model(xb.to(DEVICE))[0], xb.to(DEVICE)).item()
            for (xb,) in loader
        ) / len(loader)

        model.eval()
        with torch.no_grad():
            recon, _ = model(X_val.to(DEVICE))
            val_loss = criterion(recon, X_val.to(DEVICE)).item()

        history.append({"train": train_loss, "val": val_loss})

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            log.info("  Epoch %3d/%d | Train: %.5f | Val: %.5f", epoch, EPOCHS, train_loss, val_loss)

    model.load_state_dict(best_state)
    return model, history


def compute_reconstruction_error(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Per-sample mean squared reconstruction error."""
    model.eval()
    errors = []
    loader = DataLoader(TensorDataset(X), batch_size=256)
    with torch.no_grad():
        for (xb,) in loader:
            recon, _ = model(xb.to(DEVICE))
            err = ((recon - xb.to(DEVICE)) ** 2).mean(dim=(1, 2))
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def calibrate_threshold(model: nn.Module, X_healthy: torch.Tensor,
                          multiplier: float = 2.0) -> float:
    """
    Threshold = mean + 2*std on healthy validation data.
    Ensures ~95% of healthy samples are below threshold.
    """
    errors    = compute_reconstruction_error(model, X_healthy)
    threshold = errors.mean() + multiplier * errors.std()
    log.info("  Threshold calibrated: %.6f (mean=%.6f, std=%.6f)",
             threshold, errors.mean(), errors.std())
    return float(threshold)


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════

def plot_ae_training(history: list, name: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs  = range(1, len(history)+1)
    ax.plot(epochs, [h["train"] for h in history], color=GOLD, lw=1.8, label="Train")
    ax.plot(epochs, [h["val"]   for h in history], color=PALETTE[0], lw=1.8, label="Val")
    ax.set_title(f"{name} — Autoencoder Reconstruction Loss", color="white")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(labelcolor="white", facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"day10_ae_loss_{name.replace(' ','_')}.png",
                dpi=110, bbox_inches="tight")
    plt.close()


def plot_error_dist(errors: np.ndarray, y: np.ndarray,
                    threshold: float, name: str):
    """Reconstruction error distribution by health class + threshold line."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for label_id, color in enumerate(PALETTE):
        mask = y == label_id
        if mask.sum() > 5:
            pd.Series(errors[mask]).plot.kde(ax=ax, color=color,
                                              label=LABELS[label_id], lw=2)
    ax.axvline(threshold, color="#EF4444", ls="--", lw=1.5, label=f"Threshold={threshold:.5f}")
    ax.set_title(f"{name} — AE Reconstruction Error by Class", color="white", fontsize=11)
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#060E1A", edgecolor="#1E3A5F")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"day10_ae_dist_{name.replace(' ','_')}.png",
                dpi=110, bbox_inches="tight")
    plt.close()


def plot_roc_comparison(errors: np.ndarray, y: np.ndarray, name: str):
    """ROC curve: AE anomaly vs ground truth (binary: healthy vs not)."""
    y_bin = (y >= 1).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return
    try:
        fpr, tpr, _ = roc_curve(y_bin, errors)
        auc         = roc_auc_score(y_bin, errors)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=GOLD, lw=2, label=f"AE (AUC={auc:.3f})")
        ax.plot([0,1],[0,1],"k--", lw=0.8, alpha=0.5)
        ax.set_title(f"{name} — AE Anomaly ROC", color="white")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(labelcolor="white", facecolor="#060E1A", edgecolor="#1E3A5F")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"day10_ae_roc_{name.replace(' ','_')}.png",
                    dpi=110, bbox_inches="tight")
        plt.close()
        return auc
    except Exception:
        return None


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run():
    log.info("="*62)
    log.info("  DRDO IVHM — Day 10-11: LSTM Autoencoder")
    log.info("  Device: %s", DEVICE)
    log.info("="*62)

    results = {}

    for name in SUBSYSTEMS:
        log.info("\n── %s ──", name)
        prep = prepare_data(name)
        if prep is None:
            log.warning("  Skipping — missing data")
            continue

        X_train, X_val, X_all, y_win, scaler, feat_cols = prep
        n_features = X_train.shape[-1]
        log.info("  Train windows: %d | Features: %d", len(X_train), n_features)

        model, history = train_ae(X_train, X_val, n_features)
        threshold      = calibrate_threshold(model, X_val)
        all_errors     = compute_reconstruction_error(model, X_all)

        # Anomaly flag
        is_anomaly  = (all_errors > threshold).astype(int)
        y_bin       = (y_win >= 1).astype(int)
        n_correct   = (is_anomaly == y_bin).sum()
        binary_acc  = n_correct / len(y_win)

        auc = plot_roc_comparison(all_errors, y_win, name)
        plot_ae_training(history, name)
        plot_error_dist(all_errors, y_win, threshold, name)

        log.info("  ✅ Binary acc: %.4f | AUC: %s | Anomalies: %d/%d",
                 binary_acc, f"{auc:.4f}" if auc else "N/A",
                 is_anomaly.sum(), len(is_anomaly))

        results[name] = {
            "binary_acc": round(binary_acc, 4),
            "auc": round(auc, 4) if auc else None,
            "threshold": round(threshold, 6),
            "n_anomalies": int(is_anomaly.sum()),
        }

        # Save model + metadata
        torch.save(model.state_dict(), AE_DIR / f"{name.replace(' ','_')}.pt")
        joblib.dump({"scaler": scaler, "features": feat_cols,
                     "threshold": threshold, "seq_len": SEQ_LEN,
                     "n_features": n_features},
                    AE_DIR / f"{name.replace(' ','_')}_meta.pkl")

    print(f"\n{'='*55}")
    print("  AUTOENCODER RESULTS")
    print(f"{'='*55}")
    print(f"  {'Subsystem':<20} {'Bin Acc':>9} {'AUC':>8} {'Anomalies':>12}")
    print(f"  {'-'*52}")
    for name, m in results.items():
        print(f"  {name:<20} {m['binary_acc']:>9.4f} "
              f"{str(m['auc']):>8} {m['n_anomalies']:>12}")

    log.info("\n✅ Day 10-11 complete! Models → models/autoencoder/")
    return results


if __name__ == "__main__":
    run()
