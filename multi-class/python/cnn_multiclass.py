import os
import sys
import gc
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
OUTPUT_BASE  = "./output"
MODEL_DIR    = "./cnn_models"
PLOT_DIR     = "./cnn_plots"

CHUNK_SIZE   = 2_000_000
READ_WORKERS = 4
RANDOM_SEED  = 42
LABEL_COL    = "Label"

# Training config
BATCH_SIZE     = 1024
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5
PATIENCE       = 5      # early stopping
NUM_WORKERS    = 4      # DataLoader workers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DAY_CONFIG = {
    "day1": dict(
        max_per_class    = 300_000,
        min_per_class    = 50_000,
        smote_threshold  = 1_000,
        dropout          = 0.3,
        weight_decay     = 1e-5,
    ),
    "day2": dict(
        max_per_class    = 400_000,
        min_per_class    = 50_000,
        smote_threshold  = 500,
        dropout          = 0.25,
        weight_decay     = 1e-5,
    ),
}

BASE_FEATURES = [
    "Packet Length Mean", "Subflow Fwd Bytes", "Flow Bytes/s",
    "Fwd Packets/s", "ACK Flag Count", "Flow IAT Mean",
    "Init_Win_bytes_forward", "Flow Duration", "Packet Length Std",
    "Fwd Packet Length Std", "Bwd Packet Length Min",
    "Packet Length Variance", "Bwd Packet Length Mean",
    "Fwd Header Length", "Bwd Packets/s", "min_seg_size_forward",
    "Down/Up Ratio", "Init_Win_bytes_backward", "Bwd Packet Length Max",
    "Subflow Fwd Packets", "Subflow Bwd Packets", "Bwd IAT Total",
    "Idle Max", "Fwd IAT Min", "Bwd IAT Max", "Bwd Header Length",
    "Bwd IAT Min", "Subflow Bwd Bytes", "Active Max",
    "Active Mean", "Active Min",
]

PORT_FEATURES = [
    "is_port_dns", "is_port_ntp", "is_port_netbios", "is_port_snmp",
    "is_port_ldap", "is_port_mssql", "is_port_ssdp",
    "is_port_chargen", "is_port_rpc",
    "port_category", "src_port_log",
]

LOG_FEATURES = {
    "Subflow Fwd Bytes", "Flow Bytes/s", "Fwd Packets/s", "Bwd Packets/s",
    "Flow IAT Mean", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Max",
    "Bwd IAT Min", "Flow Duration", "Fwd Header Length", "Bwd Header Length",
    "Active Mean", "Active Max", "Active Min", "Idle Max",
}

ENGINEERED_FEATURES = [
    "bytes_per_fwd_packet", "fwd_bwd_bytes_ratio", "packet_length_cv",
    "ack_flag_ratio", "win_size_ratio", "ack_flag_squared",
    "header_length_ratio", "subflow_density",
    "amplification_factor", "burst_intensity",
]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ══════════════════════════════════════════════
# 1D CNN MODEL
# ══════════════════════════════════════════════
class Conv1DBlock(nn.Module):
    """Conv1D → BN → ReLU → Dropout block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class DDoSConv1D(nn.Module):
    """
    Input shape : (batch, n_features) → unsqueeze → (batch, 1, n_features)
    Output shape: (batch, n_classes)
    """
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features

        # 3 conv blocks
        self.block1 = Conv1DBlock(1,   64,  kernel_size=3, dropout=dropout)
        self.block2 = Conv1DBlock(64,  128, kernel_size=3, dropout=dropout)
        self.block3 = Conv1DBlock(128, 256, kernel_size=3, dropout=dropout)

        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        # Classifier head
        self.fc1 = nn.Linear(256 * 2, 128)  # GAP + GMP concat → 512
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, n_features) → (B, 1, n_features)
        x = x.unsqueeze(1)
        x = self.block1(x)   # (B, 64, n_features)
        x = self.block2(x)   # (B, 128, n_features)
        x = self.block3(x)   # (B, 256, n_features)

        # Pooling: kết hợp avg + max 
        avg = self.gap(x).squeeze(-1)   # (B, 256)
        mx  = self.gmp(x).squeeze(-1)   # (B, 256)
        x = torch.cat([avg, mx], dim=1)  # (B, 512)

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def safe_div(a, b):
    return (a / (b.abs() + 1e-9)).clip(-1e9, 1e9).fillna(0)


def engineer_features(df):
    cols = df.columns
    if "Subflow Fwd Bytes" in cols and "Subflow Fwd Packets" in cols:
        df["bytes_per_fwd_packet"] = safe_div(
            df["Subflow Fwd Bytes"], df["Subflow Fwd Packets"])
        df["subflow_density"] = safe_div(
            df["Subflow Fwd Bytes"], df["Subflow Fwd Packets"] + 1)
    if "Subflow Fwd Bytes" in cols and "Subflow Bwd Bytes" in cols:
        df["fwd_bwd_bytes_ratio"] = safe_div(
            df["Subflow Fwd Bytes"], df["Subflow Bwd Bytes"])
    if "Packet Length Mean" in cols and "Packet Length Std" in cols:
        df["packet_length_cv"] = safe_div(
            df["Packet Length Std"], df["Packet Length Mean"])
    if "ACK Flag Count" in cols and "Subflow Fwd Packets" in cols:
        total = df["Subflow Fwd Packets"] + df.get("Subflow Bwd Packets", 0)
        df["ack_flag_ratio"] = safe_div(df["ACK Flag Count"], total)
        df["ack_flag_squared"] = df["ACK Flag Count"] ** 2
    if "Init_Win_bytes_forward" in cols and "Init_Win_bytes_backward" in cols:
        df["win_size_ratio"] = safe_div(
            df["Init_Win_bytes_forward"], df["Init_Win_bytes_backward"])
    if "Fwd Header Length" in cols and "Bwd Header Length" in cols:
        df["header_length_ratio"] = safe_div(
            df["Fwd Header Length"], df["Bwd Header Length"])
    if "Bwd Packet Length Mean" in cols and "Packet Length Mean" in cols:
        df["amplification_factor"] = safe_div(
            df["Bwd Packet Length Mean"], df["Packet Length Mean"])
    if "Fwd Packets/s" in cols and "Bwd Packets/s" in cols:
        df["burst_intensity"] = safe_div(
            df["Fwd Packets/s"], df["Bwd Packets/s"] + 1)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def apply_log(df):
    cols = [c for c in LOG_FEATURES if c in df.columns]
    df[cols] = np.log1p(df[cols].clip(lower=0))
    return df


def smote_lite(sub, target_n, feat_cols, rng):
    n = len(sub)
    n_gen = target_n - n
    num_cols = [c for c in feat_cols if c in sub.columns]
    X = sub[num_cols].values.astype(np.float32)
    rows = []
    for _ in range(n_gen):
        i, j = rng.integers(0, n, size=2)
        alpha = rng.random()
        rows.append(X[i] + alpha * (X[j] - X[i]))
    syn = pd.DataFrame(rows, columns=num_cols)
    syn[LABEL_COL] = sub[LABEL_COL].iloc[0]
    return pd.concat([sub, syn], ignore_index=True)


def balance_df(df, feat_cols, cfg):
    rng = np.random.default_rng(RANDOM_SEED)
    max_n, min_n = cfg["max_per_class"], cfg["min_per_class"]
    smote_thr = cfg["smote_threshold"]
    frames = []
    for cls in df[LABEL_COL].unique():
        sub = df[df[LABEL_COL] == cls].copy()
        n = len(sub)
        if n == 0:
            continue
        if n > max_n:
            sub = sub.sample(max_n, random_state=RANDOM_SEED)
            print(f"  ↓ {cls:<25s}  {n:>10,} → {max_n:,}")
        elif n < smote_thr:
            target = min(min_n, n * 20)
            sub = smote_lite(sub, target, feat_cols, rng)
            print(f"  ✦ {cls:<25s}  {n:>10,} → {len(sub):,}  [SMOTE]")
        elif n < min_n:
            target = min(min_n, n * 5)
            factor = int(np.ceil(target / n))
            sub_rep = pd.concat([sub] * factor, ignore_index=True).iloc[:target]
            num_cols = [c for c in sub_rep.columns if c != LABEL_COL]
            noise = rng.normal(0, 1e-4, size=(len(sub_rep), len(num_cols)))
            sub_rep[num_cols] = sub_rep[num_cols].values + noise
            sub = sub_rep
            print(f"  ↑ {cls:<25s}  {n:>10,} → {len(sub):,}  [dup]")
        else:
            print(f"    {cls:<25s}  {n:>10,}  (keep)")
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def _read_worker(args):
    path, skiprows, nrows, col_names, all_feats = args
    try:
        df = pd.read_csv(path, skiprows=skiprows, nrows=nrows,
                          header=None, names=col_names,
                          low_memory=False, on_bad_lines="skip")
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    if LABEL_COL not in df.columns:
        return None
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[~df[LABEL_COL].isin({"", "nan", "Label"})]
    if df.empty:
        return None
    keep = [c for c in all_feats if c in df.columns] + [LABEL_COL]
    df = df[keep]
    num_cols = [c for c in df.columns if c != LABEL_COL]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def load_csv(path, all_feats, desc="Loading"):
    with open(path, "rb") as f:
        n_data = sum(1 for _ in f) - 1
    col_names = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    args_list, offset = [], 1
    while offset <= n_data:
        nrows = min(CHUNK_SIZE, n_data - offset + 1)
        args_list.append((path, offset, nrows, col_names, all_feats))
        offset += nrows
    frames = [None] * len(args_list)
    pbar = tqdm(total=len(args_list), desc=f"  {desc}", unit="chunk")
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as ex:
        fmap = {ex.submit(_read_worker, a): i for i, a in enumerate(args_list)}
        for future in as_completed(fmap):
            frames[fmap[future]] = future.result()
            pbar.update(1)
    pbar.close()
    valid = [f for f in frames if f is not None and not f.empty]
    return pd.concat(valid, ignore_index=True, copy=False)


def stratified_split_indices(y, val_frac=0.05, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    tr, vl = [], []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        rng.shuffle(pos)
        nv = max(1, int(len(pos) * val_frac))
        vl.extend(pos[:nv]); tr.extend(pos[nv:])
    return np.array(tr), np.array(vl)


# ══════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y_batch).sum().item()
        total += y_batch.size(0)

        if total % 50000 < BATCH_SIZE:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                              acc=f"{correct/total*100:.2f}%")
    return total_loss / total, correct / total


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    return (total_loss / total, correct / total,
            np.concatenate(all_targets), np.concatenate(all_preds))


def predict_streaming(model, X, batch_size=4096, device=DEVICE):
    """Predict on numpy array, return class indices."""
    model.eval()
    preds = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device,
                                                            non_blocking=True)
            out = model(batch)
            _, pred = out.max(1)
            preds[i:i+batch_size] = pred.cpu().numpy()
    return preds


# ══════════════════════════════════════════════
# TRAIN 1 DAY
# ══════════════════════════════════════════════
def train_one_day(day_name: str):
    print("\n" + "█" * 70)
    print(f"  TRAINING {day_name.upper()}  (1D CNN)")
    print("█" * 70)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    t_start = time.time()

    cfg = DAY_CONFIG[day_name]
    day_dir = os.path.join(OUTPUT_BASE, day_name)
    train_path = os.path.join(day_dir, "train.csv")
    test_path  = os.path.join(day_dir, "test.csv")

    if not os.path.exists(train_path):
        print(f"  [ERROR] {train_path} unavailable")
        return None

    # Scan
    print(f"\n[1/7]  Scan {day_name}...")
    header = pd.read_csv(train_path, nrows=0)
    file_cols = set(header.columns.str.strip())
    base_feats = [f for f in BASE_FEATURES if f in file_cols]
    port_feats = [f for f in PORT_FEATURES if f in file_cols]
    all_load_feats = base_feats + port_feats
    print(f"  Base / Port features: {len(base_feats)} / {len(port_feats)}")

    label_counts = Counter()
    for chunk in pd.read_csv(train_path, usecols=[LABEL_COL],
                              chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        vals = chunk[LABEL_COL].astype(str).str.strip()
        vals = vals[~vals.isin({"", "nan", "Label"})]
        label_counts.update(vals.tolist())
    all_classes = sorted(label_counts.keys())
    n_classes = len(all_classes)
    print(f"  Classes: {n_classes}")
    for cls, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<25s}  {cnt:>10,}")

    # Load + engineer
    print(f"\n[2/7]  Load + feature engineering...")
    df = load_csv(train_path, all_load_feats, desc="Train")
    df = engineer_features(df)
    df = apply_log(df)
    new_eng = [c for c in ENGINEERED_FEATURES if c in df.columns]
    feat_cols = base_feats + port_feats + new_eng
    print(f"  Total features: {len(feat_cols)}")

    # Balance
    print(f"\n[3/7]  Balance classes...")
    df_bal = balance_df(df, feat_cols, cfg)
    print(f"\n  Sau balance: {len(df_bal):,}")
    del df; gc.collect()

    # Prep tensors
    print(f"\n[4/7]  Prep tensors + StandardScaler...")
    le = LabelEncoder()
    le.fit(all_classes)
    X = df_bal[feat_cols].values.astype(np.float32)
    y = le.transform(df_bal[LABEL_COL].values).astype(np.int64)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    tr_idx, vl_idx = stratified_split_indices(y, val_frac=0.05)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_vl, y_vl = X[vl_idx], y[vl_idx]
    print(f"  Train: {len(X_tr):,}  Val: {len(X_vl):,}")
    del df_bal, X
    gc.collect()

    # Class weights for cross entropy
    class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    class_weights = np.sqrt(class_counts.max() / (class_counts + 1e-9))
    class_weights = class_weights / class_weights.mean()
    cw_tensor = torch.from_numpy(class_weights).float().to(DEVICE)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=(DEVICE.type == "cuda"),
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=(DEVICE.type == "cuda"))

    # Model
    print(f"\n[5/7]  Build 1D CNN model...")
    model = DDoSConv1D(n_features=len(feat_cols),
                        n_classes=n_classes,
                        dropout=cfg["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    print(model)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=LEARNING_RATE,
                                    weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training
    print(f"\n[6/7]  Training {NUM_EPOCHS} epochs (early stop patience={PATIENCE})...")
    best_val_acc = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_path = os.path.join(MODEL_DIR, f"cnn_{day_name}_v16_best.pt")

    for epoch in range(NUM_EPOCHS):
        t_ep = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                            criterion, optimizer, DEVICE)
        vl_loss, vl_acc, _, _ = evaluate_model(model, val_loader,
                                                 criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        improved = vl_acc > best_val_acc
        marker = "  ★ new best" if improved else ""
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}%  "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc*100:.2f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}  "
              f"({time.time()-t_ep:.1f}s){marker}")

        if improved:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": vl_acc,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    print(f"\n  Load best model (val_acc={best_val_acc*100:.4f}%)...")
    ckpt = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Free memory
    del train_ds, val_ds, train_loader, val_loader
    del X_tr, y_tr, X_vl, y_vl
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Evaluate on test set
    print(f"\n[7/7]  Evaluate trên test.csv...")
    t0 = time.time()
    all_true, all_pred = [], []

    reader = pd.read_csv(test_path, chunksize=CHUNK_SIZE,
                          low_memory=False, on_bad_lines="skip")
    for chunk in tqdm(reader, desc="  Eval", unit="chunk"):
        chunk.columns = chunk.columns.str.strip()
        chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()
        chunk = chunk[~chunk[LABEL_COL].isin({"", "nan", "Label"})]
        if chunk.empty:
            continue
        num_cols = [c for c in chunk.columns if c != LABEL_COL]
        chunk[num_cols] = chunk[num_cols].apply(pd.to_numeric, errors="coerce")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)
        chunk = engineer_features(chunk)
        chunk = apply_log(chunk)
        for c in feat_cols:
            if c not in chunk.columns:
                chunk[c] = 0.0
        X_test = chunk[feat_cols].values.astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        y_test = le.transform(chunk[LABEL_COL].values)

        yp = predict_streaming(model, X_test, batch_size=4096)
        all_true.extend(y_test)
        all_pred.extend(yp)

    yt = np.array(all_true)
    yp = np.array(all_pred)

    acc = accuracy_score(yt, yp)
    report = classification_report(yt, yp,
                                    target_names=le.classes_,
                                    zero_division=0, digits=4)
    f1s = f1_score(yt, yp, average=None, zero_division=0)

    print(f"\n  {day_name.upper()} CNN Accuracy : {acc*100:.4f}%")
    print(f"  Macro F1                : {f1s.mean():.4f}  ({time.time()-t0:.1f}s)\n")
    print(report)

    print("  Per-class F1:")
    for i, cls in enumerate(le.classes_):
        bar = "█" * int(f1s[i] * 30)
        flag = " ←" if f1s[i] < 0.7 else ""
        print(f"    {cls:<25s}  {f1s[i]:.4f}  {bar}{flag}")

    # Save
    print(f"\n[SAVE]  Lưu model + scaler + label encoder...")
    final_model_path = os.path.join(MODEL_DIR, f"cnn_{day_name}_v16_model.pt")
    torch.save({
        "model_state" : model.state_dict(),
        "n_features"  : len(feat_cols),
        "n_classes"   : n_classes,
        "dropout"     : cfg["dropout"],
    }, final_model_path)
    joblib.dump(scaler,    os.path.join(MODEL_DIR, f"cnn_{day_name}_v16_scaler.pkl"))
    joblib.dump(le,        os.path.join(MODEL_DIR, f"cnn_{day_name}_v16_le.pkl"))
    joblib.dump(feat_cols, os.path.join(MODEL_DIR, f"cnn_{day_name}_v16_features.pkl"))

    report_path = os.path.join(OUTPUT_BASE,
                                f"cnn_{day_name}_v16_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Day: {day_name} (1D CNN v16)\n")
        f.write(f"Model parameters: {n_params:,}\n")
        f.write(f"Best val accuracy: {best_val_acc*100:.4f}%\n")
        f.write(f"Test accuracy   : {acc*100:.4f}%\n")
        f.write(f"Macro F1        : {f1s.mean():.4f}\n\n")
        f.write(report)
    print(f"  ✓ {report_path}")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   markersize=4)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"CNN {day_name.upper()} – Training Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["train_acc"]],
                  "b-o", label="Train", markersize=4)
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],
                  "r-o", label="Val",   markersize=4)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"CNN {day_name.upper()} – Training Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(PLOT_DIR, f"cnn_{day_name}_v16_training_history.png")
    plt.savefig(hist_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {hist_path}")

    # Confusion matrix
    cm = confusion_matrix(yt, yp)
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    n = len(le.classes_)
    fig, axes = plt.subplots(1, 2, figsize=(n * 2 + 2, n + 1))
    for ax, data, title, fmt in zip(
        axes, [cm, cmn], ["Count", "Normalized"], [".0f", ".2f"]
    ):
        disp = ConfusionMatrixDisplay(data, display_labels=le.classes_)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=True,
                  cmap="Blues", values_format=fmt)
        ax.set_title(f"CNN {day_name.upper()} v16 – {title}",
                     fontsize=11, pad=10)
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, f"cnn_{day_name}_v16_confusion_matrix.png")
    plt.savefig(cm_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {cm_path}")

    print(f"\n  [DONE {day_name.upper()}]  "
          f"CNN Accuracy = {acc*100:.4f}%  "
          f"({(time.time()-t_start)/60:.1f} min)")
    return acc


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cnn_multiclass.py day1")
        print("  python cnn_multiclass.py day2")
        print("  python cnn_multiclass.py both")
        sys.exit(1)

    arg = sys.argv[1].lower()
    if arg == "both":
        days = ["day1", "day2"]
    elif arg in ("day1", "day2"):
        days = [arg]
    else:
        print(f"  [ERROR] Invalid: {arg}")
        sys.exit(1)

    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True

    results = {}
    for day in days:
        acc = train_one_day(day)
        results[day] = acc

    print("\n" + "═" * 70)
    print("  CNN MULTI-CLASS REPORT ")
    print("═" * 70)
    for day, acc in results.items():
        if acc is not None:
            print(f"  {day}  →  Accuracy = {acc*100:.4f}%")


if __name__ == "__main__":
    main()