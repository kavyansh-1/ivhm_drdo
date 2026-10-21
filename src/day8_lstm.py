"""
DRDO IVHM — Day 8-9: LSTM (Multi-Task: Health Classification + RUL Regression)
PyTorch implementation of Bidirectional LSTM.
Architecture:
  Input  : (batch, seq_len=20, n_features)
  Encoder: BiLSTM(128) → Dropout → BiLSTM(64) → Dropout
  Head 1 : FC(32) → FC(3)    [health class logits]
  Head 2 : FC(32) → FC(1)    [RUL regression]
Saves: models/lstm/<subsystem>.pt + scaler + metadata.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import (SUBSYSTEMS, DATA_PROCESSED, MODELS_DIR,
                    FIGURES_DIR, LABELS, PALETTE, PLOT_STYLE, GOLD, RANDOM_SEED)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
matplotlib.rcParams.update(PLOT_STYLE)

LSTM_DIR   = MODELS_DIR / "lstm"
LSTM_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 20      # lookback window (timesteps)
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 8       # early stopping


# ════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════

class UAVSequenceDataset(Dataset):
    """
    Sliding window dataset: each sample is SEQ_LEN consecutive timesteps.
    Label = health class of the LAST timestep in the window.
    RUL   = rul value of the last timestep (normalised 0-1).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 rul: np.ndarray, seq_len: int = SEQ_LEN):
        self.X, self.y, self.rul = [], [], []
        for i in range(len(X) - seq_len):
            self.X.append(X[i: i + seq_len])
            self.y.append(y[i + seq_len - 1])
            self.rul.append(rul[i + seq_len - 1])
        self.X   = torch.tensor(np.array(self.X),   dtype=torch.float32)
        self.y   = torch.tensor(np.array(self.y),   dtype=torch.long)
        self.rul = torch.tensor(np.array(self.rul), dtype=torch.float32)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.rul[i]


# ════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════

class IVHMLSTMNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3,
                 hidden1: int = 128, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.drop  = nn.Dropout(dropout)
        self.bn    = nn.BatchNorm1d(hidden2 * 2)

        # Health classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden2 * 2, 32),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
        # RUL regression head
        self.rul_head = nn.Sequential(
            nn.Linear(hidden2 * 2, 32),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()   # normalised RUL [0,1]
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out     = self.drop(out)
        out, _ = self.lstm2(out)
        last    = out[:, -1, :]        # last timestep
        last    = self.bn(last)
        cls_out = self.cls_head(last)
        rul_out = self.rul_head(last).squeeze(-1)
        return cls_out, rul_out


# ════════════════════════════════════════════════════════════
# DATA PREP
# ════════════════════════════════════════════════════════════

def get_feature_cols(df):
    skip = {"health_label","health_status","uav_id","timestep",
            "subsystem","rul","capacity_ah","life_fraction","fault_start"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in skip]


def build_rul(df: pd.DataFrame) -> np.ndarray:
    """Build normalised RUL proxy from health label if no explicit RUL column."""
    if "rul" in df.columns:
        rul = df["rul"].fillna(0).values.astype(float)
        max_rul = max(rul.max(), 1)
        return rul / max_rul

    # Proxy: fraction of remaining life based on health label
    # Healthy=1.0, Degraded=0.5, Faulty=0.1
    proxy_map = {0: 1.0, 1: 0.5, 2: 0.1}
    return df["health_label"].map(proxy_map).values.astype(float)


def prepare_sequences(name: str):
    path = DATA_PROCESSED / f"{name.replace(' ','_')}.csv"
    if not path.exists():
        log.warning("Missing %s — run day2 first", path)
        return None

    df        = pd.read_csv(path).sort_values(["uav_id","timestep"])
    feat_cols = get_feature_cols(df)
    rul       = build_rul(df)

    scaler = StandardScaler()
    X      = scaler.fit_transform(df[feat_cols].fillna(0))
    y      = df["health_label"].values

    # Split UAVs: 80% train, 20% test (same anti-leakage strategy as Day 3)
    uav_ids = df["uav_id"].values if "uav_id" in df.columns else np.zeros(len(df))
    unique  = np.unique(uav_ids)
    n_test  = max(1, int(len(unique) * 0.2))
    test_uavs = np.random.choice(unique, n_test, replace=False)

    train_mask = ~np.isin(uav_ids, test_uavs)
    test_mask  =  np.isin(uav_ids, test_uavs)

    # Build per-UAV sequences (don't cross UAV boundaries!)
    def build_dataset_per_uav(mask):
        all_X, all_y, all_rul = [], [], []
        for uid in np.unique(uav_ids[mask]):
            uid_mask = (uav_ids == uid) & mask
            ds = UAVSequenceDataset(X[uid_mask], y[uid_mask], rul[uid_mask])
            if len(ds) > 0:
                all_X.append(ds.X); all_y.append(ds.y); all_rul.append(ds.rul)
        if not all_X:
            return None
        return torch.cat(all_X), torch.cat(all_y), torch.cat(all_rul)

    train_data = build_dataset_per_uav(train_mask)
    test_data  = build_dataset_per_uav(test_mask)

    if train_data is None:
        return None

    return train_data, test_data, scaler, feat_cols


# ════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════

def train_model(train_data, test_data, n_features: int, name: str):
    Xtr, ytr, rul_tr = train_data
    train_ds = torch.utils.data.TensorDataset(Xtr, ytr, rul_tr)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = IVHMLSTMNet(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    cls_loss  = nn.CrossEntropyLoss()
    rul_loss  = nn.MSELoss()

    best_val_f1, patience_cnt = 0.0, 0
    history = {"train_loss": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for Xb, yb, rul_b in train_dl:
            Xb, yb, rul_b = Xb.to(DEVICE), yb.to(DEVICE), rul_b.to(DEVICE)
            optimizer.zero_grad()
            cls_out, rul_out = model(Xb)
            loss = cls_loss(cls_out, yb) + 0.3 * rul_loss(rul_out, rul_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # ── Validate ──
        if test_data is not None:
            Xte, yte, rul_te = test_data
            model.eval()
            with torch.no_grad():
                cls_out, rul_out = model(Xte.to(DEVICE))
                preds = cls_out.argmax(1).cpu().numpy()
                val_f1 = f1_score(yte.numpy(), preds, average="weighted", zero_division=0)
        else:
            val_f1 = 0.0

        history["train_loss"].append(epoch_loss / len(train_dl))
        history["val_f1"].append(val_f1)

        if epoch % 10 == 0:
            log.info("  Epoch %3d/%d | Loss: %.4f | Val F1: %.4f",
                     epoch, EPOCHS, history["train_loss"][-1], val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), LSTM_DIR / f"{name.replace(' ','_')}_best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log.info("  Early stop at epoch %d (best F1=%.4f)", epoch, best_val_f1)
                break

    # Load best weights
    model.load_state_dict(torch.load(LSTM_DIR / f"{name.replace(' ','_')}_best.pt",
                                      map_location=DEVICE))
    return model, history, best_val_f1


def evaluate(model: nn.Module, test_data) -> dict:
    if test_data is None:
        return {}
    Xte, yte, rul_te = test_data
    model.eval()
    with torch.no_grad():
        cls_out, rul_out = model(Xte.to(DEVICE))
        preds   = cls_out.argmax(1).cpu().numpy()
        rul_pred = rul_out.cpu().numpy()
        rul_true = rul_te.numpy()

    acc  = accuracy_score(yte.numpy(), preds)
    f1   = f1_score(yte.numpy(), preds, average="weighted", zero_division=0)
    rmse = float(np.sqrt(np.mean((rul_pred - rul_true)**2)))

    return {"test_acc": round(acc,4), "test_f1": round(f1,4),
            "rul_rmse": round(rmse,4)}


# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════

def plot_training(history: dict, name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{name} — LSTM Training", color="white", fontsize=12)

    ax1.plot(history["train_loss"], color=GOLD, lw=1.8)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
    ax1.grid(True)

    ax2.plot(history["val_f1"], color=PALETTE[0], lw=1.8)
    ax2.set_title("Validation F1"); ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1.05); ax2.grid(True)

    plt.tight_layout()
    out = FIGURES_DIR / f"day8_lstm_training_{name.replace(' ','_')}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run():
    log.info("="*62)
    log.info("  DRDO IVHM — Day 8-9: BiLSTM Multi-Task Training")
    log.info("  Device: %s", DEVICE)
    log.info("="*62)

    results = {}

    for name in SUBSYSTEMS:
        log.info("\n── %s ──", name)
        prep = prepare_sequences(name)
        if prep is None:
            continue

        train_data, test_data, scaler, feat_cols = prep
        n_features = train_data[0].shape[-1]
        log.info("  Train sequences: %d | Features: %d | Device: %s",
                 len(train_data[0]), n_features, DEVICE)

        model, history, best_f1 = train_model(
            train_data, test_data, n_features, name)

        metrics = evaluate(model, test_data)
        results[name] = metrics

        log.info("  ✅ Best F1: %.4f | Test F1: %.4f | RUL RMSE: %.4f",
                 best_f1, metrics.get("test_f1",0), metrics.get("rul_rmse",0))

        plot_training(history, name)

        # Save full artefact
        joblib.dump({"scaler": scaler, "features": feat_cols,
                     "seq_len": SEQ_LEN, "n_features": n_features},
                    LSTM_DIR / f"{name.replace(' ','_')}_meta.pkl")

    # Summary
    print(f"\n{'='*55}")
    print("  LSTM RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Subsystem':<20} {'Test F1':>9} {'RUL RMSE':>10}")
    print(f"  {'-'*42}")
    for name, m in results.items():
        print(f"  {name:<20} {m.get('test_f1',0):>9.4f} {m.get('rul_rmse',0):>10.4f}")

    log.info("\n✅ Day 8-9 complete! Models → models/lstm/")
    return results


if __name__ == "__main__":
    run()
