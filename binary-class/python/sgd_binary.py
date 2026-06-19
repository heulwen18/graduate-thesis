import os
import sys
import gc
import time
import warnings
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

OUTPUT_BASE  = "./output"
MODEL_DIR    = "./sgd_models"
PLOT_DIR     = "./sgd_plots"

CHUNK_SIZE   = 200_000     # nhỏ hơn vì streaming
RANDOM_SEED  = 42
LABEL_COL    = "Label"

BENIGN_LABEL = "BENIGN"
ATTACK_LABEL = "ATTACK"

DAY_CONFIG = {
    "day1": dict(
        n_epochs       = 5,         # số lần đi qua toàn bộ data
        learning_rate  = "optimal", # "constant" | "optimal" | "invscaling" | "adaptive"
        eta0           = 0.01,
        alpha          = 1e-5,      # L2 regularization strength
        loss           = "log_loss",
    ),
    "day2": dict(
        n_epochs       = 5,
        learning_rate  = "optimal",
        eta0           = 0.01,
        alpha          = 1e-5,
        loss           = "log_loss",
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
# HELPERS
# ══════════════════════════════════════════════
def safe_div(a, b):
    return (a / (b.abs() + 1e-9)).clip(-1e9, 1e9).fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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


def apply_log(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LOG_FEATURES if c in df.columns]
    df[cols] = np.log1p(df[cols].clip(lower=0))
    return df


def to_binary(label_series: pd.Series) -> pd.Series:
    return label_series.apply(lambda x: BENIGN_LABEL if x == BENIGN_LABEL
                                else ATTACK_LABEL)


# ══════════════════════════════════════════════
# STREAMING PREPROCESSING
# ══════════════════════════════════════════════
def process_chunk(chunk: pd.DataFrame, all_load_feats: list,
                   feat_cols: list) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process 1 chunk: clean → binary label → engineer → log → return (X, y).
    """
    chunk.columns = chunk.columns.str.strip()
    if LABEL_COL not in chunk.columns:
        return None
    chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()
    chunk = chunk[~chunk[LABEL_COL].isin({"", "nan", "Label"})]
    if chunk.empty:
        return None

    # Binary label
    chunk[LABEL_COL] = to_binary(chunk[LABEL_COL])

    # Keep features + label
    keep = [c for c in all_load_feats if c in chunk.columns] + [LABEL_COL]
    chunk = chunk[keep].copy()

    num_cols = [c for c in chunk.columns if c != LABEL_COL]
    chunk[num_cols] = chunk[num_cols].apply(pd.to_numeric, errors="coerce")
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(0, inplace=True)

    # Engineer + log
    chunk = engineer_features(chunk)
    chunk = apply_log(chunk)

    for c in feat_cols:
        if c not in chunk.columns:
            chunk[c] = 0.0

    X = chunk[feat_cols].values.astype(np.float32)
    y_raw = chunk[LABEL_COL].values
    # 0=BENIGN, 1=ATTACK
    y = (y_raw == ATTACK_LABEL).astype(np.int8)
    return X, y


# ══════════════════════════════════════════════
# STREAMING SCALER FIT (1 pass)
# ══════════════════════════════════════════════
def fit_scaler_streaming(train_path: str, all_load_feats: list,
                          feat_cols: list) -> StandardScaler:
    """Fit StandardScaler bằng partial_fit (Welford's online algorithm)."""
    print(f"  [Scaler] Fitting StandardScaler streaming...")
    t0 = time.time()
    scaler = StandardScaler()
    reader = pd.read_csv(train_path, chunksize=CHUNK_SIZE,
                          low_memory=False, on_bad_lines="skip")
    total_rows = 0
    for chunk in tqdm(reader, desc="    Scaler fit", unit="chunk"):
        result = process_chunk(chunk, all_load_feats, feat_cols)
        if result is None:
            continue
        X, _ = result
        scaler.partial_fit(X)
        total_rows += len(X)
    print(f"  ✓ Scaler fit done on {total_rows:,} rows ({time.time()-t0:.1f}s)")
    return scaler


# ══════════════════════════════════════════════
# STREAMING TRAIN
# ══════════════════════════════════════════════
def train_sgd_streaming(train_path: str, all_load_feats: list,
                         feat_cols: list, scaler: StandardScaler,
                         cfg: dict) -> Tuple[SGDClassifier, np.ndarray]:
   
    print(f"\n  [Train] SGDClassifier streaming, {cfg['n_epochs']} epochs")
    t0 = time.time()

    classes = np.array([0, 1])  # BENIGN, ATTACK

    # Count class distribution
    print(f"  Pass 0: counting class distribution...")
    counter = Counter()
    reader = pd.read_csv(train_path, usecols=[LABEL_COL],
                          chunksize=CHUNK_SIZE, low_memory=False)
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        vals = chunk[LABEL_COL].astype(str).str.strip()
        vals = vals[~vals.isin({"", "nan", "Label"})]
        vals = to_binary(vals)
        counter.update(vals.tolist())
    n_benign = counter.get(BENIGN_LABEL, 0)
    n_attack = counter.get(ATTACK_LABEL, 0)
    print(f"  Class counts: BENIGN={n_benign:,}  ATTACK={n_attack:,}")
    print(f"  Imbalance ratio: 1 : {n_attack / max(n_benign, 1):.1f}")

    # Class weights (balanced)
    cw = compute_class_weight("balanced",
                                classes=classes,
                                y=np.concatenate([
                                    np.zeros(n_benign, dtype=np.int8),
                                    np.ones(n_attack, dtype=np.int8)
                                ]))
    # CW dict (label -> weight)
    cw_dict = {0: cw[0], 1: cw[1]}
    print(f"  Class weights: BENIGN={cw[0]:.3f}  ATTACK={cw[1]:.3f}")

    # SGD model
    model = SGDClassifier(
        loss          = cfg["loss"],
        penalty       = "l2",
        alpha         = cfg["alpha"],
        learning_rate = cfg["learning_rate"],
        eta0          = cfg["eta0"],
        random_state  = RANDOM_SEED,
        n_jobs        = -1,
        verbose       = 0,

    )

    # Train with n epoch
    for epoch in range(cfg["n_epochs"]):
        print(f"\n  Epoch {epoch+1}/{cfg['n_epochs']}...")
        reader = pd.read_csv(train_path, chunksize=CHUNK_SIZE,
                              low_memory=False, on_bad_lines="skip")
        epoch_rows = 0
        epoch_correct = 0

        rng = np.random.default_rng(RANDOM_SEED + epoch)

        for chunk in tqdm(reader, desc=f"  Ep{epoch+1}", unit="chunk"):
            result = process_chunk(chunk, all_load_feats, feat_cols)
            if result is None:
                continue
            X, y = result

            # Scale
            X = scaler.transform(X).astype(np.float32)

            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]

            # Sample weights by class
            sw = np.where(y == 0, cw_dict[0], cw_dict[1]).astype(np.float32)

            # Partial fit
            model.partial_fit(X, y, classes=classes, sample_weight=sw)

            # Quick accuracy estimate
            y_pred = model.predict(X)
            epoch_correct += (y_pred == y).sum()
            epoch_rows += len(y)

        ep_acc = epoch_correct / max(epoch_rows, 1)
        print(f"  Epoch {epoch+1} done: train_acc ≈ {ep_acc*100:.4f}%  "
              f"({epoch_rows:,} rows)")

    print(f"\n  ✓ Training done ({time.time()-t0:.1f}s)")
    return model, cw


# ══════════════════════════════════════════════
# STREAMING EVALUATE
# ══════════════════════════════════════════════
def evaluate_streaming(model, scaler, test_path: str,
                        all_load_feats: list, feat_cols: list):
    print(f"\n  [Eval] Streaming evaluation on test.csv...")
    t0 = time.time()
    all_true = []
    all_pred = []
    all_score = []  # decision_function score (cho ROC)

    reader = pd.read_csv(test_path, chunksize=CHUNK_SIZE,
                          low_memory=False, on_bad_lines="skip")
    for chunk in tqdm(reader, desc="    Eval", unit="chunk"):
        result = process_chunk(chunk, all_load_feats, feat_cols)
        if result is None:
            continue
        X, y = result
        X = scaler.transform(X).astype(np.float32)

        # decision_function cho ROC/AUC
        scores = model.decision_function(X)
        y_pred = (scores >= 0).astype(np.int8)

        all_true.append(y)
        all_pred.append(y_pred)
        all_score.append(scores)

    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    ss = np.concatenate(all_score)

    # Sigmoid 
    pp = 1.0 / (1.0 + np.exp(-np.clip(ss, -500, 500)))

    print(f"  ✓ Evaluated {len(yt):,} rows ({time.time()-t0:.1f}s)")
    return yt, yp, pp, ss


# ══════════════════════════════════════════════
# TRAIN 1 DAY
# ══════════════════════════════════════════════
def train_one_day(day_name: str):
    print("\n" + "█" * 70)
    print(f"  BINARY SGD CLASSIFICATION {day_name.upper()}")
    print("█" * 70)
    t_start = time.time()

    cfg = DAY_CONFIG[day_name]
    day_dir = os.path.join(OUTPUT_BASE, day_name)
    train_path = os.path.join(day_dir, "train.csv")
    test_path  = os.path.join(day_dir, "test.csv")

    if not os.path.exists(train_path):
        print(f"  [ERROR] {train_path} invalid")
        return None

    # Scan features
    print(f"\n[1/4]  Scan {day_name}...")
    header = pd.read_csv(train_path, nrows=0)
    file_cols = set(header.columns.str.strip())
    base_feats = [f for f in BASE_FEATURES if f in file_cols]
    port_feats = [f for f in PORT_FEATURES if f in file_cols]
    all_load_feats = base_feats + port_feats
    feat_cols = base_feats + port_feats + ENGINEERED_FEATURES
    print(f"  Base / Port / Engineered: "
          f"{len(base_feats)} / {len(port_feats)} / {len(ENGINEERED_FEATURES)}")
    print(f"  Total features: {len(feat_cols)}")
    print(f"  Config: {cfg}")

    # Fit scaler (1 pass)
    print(f"\n[2/4]  Fit StandardScaler streaming...")
    scaler = fit_scaler_streaming(train_path, all_load_feats, feat_cols)

    # Train SGD (multiple passes)
    print(f"\n[3/4]  Train SGD streaming...")
    model, cw = train_sgd_streaming(train_path, all_load_feats,
                                      feat_cols, scaler, cfg)

    # Evaluate
    print(f"\n[4/4]  Evaluate on test.csv...")
    yt, yp, pp, ss = evaluate_streaming(model, scaler, test_path,
                                          all_load_feats, feat_cols)

    # Metrics
    acc = accuracy_score(yt, yp)
    auc = roc_auc_score(yt, pp)
    ap  = average_precision_score(yt, pp)
    report = classification_report(yt, yp,
                                    target_names=[BENIGN_LABEL, ATTACK_LABEL],
                                    zero_division=0, digits=4)
    f1s = f1_score(yt, yp, average=None, zero_division=0)

    cm = confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm.ravel()
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_attack = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n  {day_name.upper()} SGD BINARY RESULTS:")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Accuracy            : {acc*100:.4f}%             │")
    print(f"  │  AUC-ROC             : {auc:.6f}              │")
    print(f"  │  Avg Precision (PR)  : {ap:.6f}              │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Detection Rate (TPR): {detection_rate*100:.4f}%             │")
    print(f"  │  False Positive Rate : {fpr_rate*100:.4f}%             │")
    print(f"  │  False Negative Rate : {fnr_rate*100:.4f}%             │")
    print(f"  │  Precision (ATTACK)  : {precision_attack*100:.4f}%             │")
    print(f"  └─────────────────────────────────────────────────┘")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"               BENIGN    ATTACK")
    print(f"  Actual BENIGN  {tn:>8,}  {fp:>8,}")
    print(f"        ATTACK   {fn:>8,}  {tp:>8,}")
    print(f"\n{report}")

    # Save
    print(f"\n[SAVE]  Saving model...")
    joblib.dump(model,     os.path.join(MODEL_DIR, f"sgd_{day_name}_binary.pkl"))
    joblib.dump(scaler,    os.path.join(MODEL_DIR, f"sgd_{day_name}_scaler.pkl"))
    joblib.dump(feat_cols, os.path.join(MODEL_DIR, f"sgd_{day_name}_features.pkl"))

    report_path = os.path.join(OUTPUT_BASE,
                                f"sgd_{day_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: SGDClassifier (loss={cfg['loss']}, "
                f"alpha={cfg['alpha']}, n_epochs={cfg['n_epochs']})\n\n")
        f.write(f"Accuracy            : {acc*100:.4f}%\n")
        f.write(f"AUC-ROC             : {auc:.6f}\n")
        f.write(f"Avg Precision       : {ap:.6f}\n")
        f.write(f"Detection Rate (TPR): {detection_rate*100:.4f}%\n")
        f.write(f"False Positive Rate : {fpr_rate*100:.4f}%\n")
        f.write(f"False Negative Rate : {fnr_rate*100:.4f}%\n")
        f.write(f"Precision (ATTACK)  : {precision_attack*100:.4f}%\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"               Predicted\n")
        f.write(f"              BENIGN    ATTACK\n")
        f.write(f"  BENIGN  {tn:>10,}  {fp:>10,}\n")
        f.write(f"  ATTACK  {fn:>10,}  {tp:>10,}\n\n")
        f.write(report)
    print(f"  ✓ {report_path}")

    # ── Plots ──
    # 1. Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    for ax, data, title, fmt in zip(
        axes, [cm, cmn], ["Count", "Normalized"], [".0f", ".2%"]
    ):
        disp = ConfusionMatrixDisplay(data,
                                        display_labels=[BENIGN_LABEL, ATTACK_LABEL])
        disp.plot(ax=ax, colorbar=True, cmap="Greens", values_format=fmt)
        ax.set_title(f"SGD Binary {day_name.upper()} v18 – {title}")
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, f"sgd_{day_name}_v18_binary_cm.png")
    plt.savefig(cm_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {cm_path}")

    # 2. ROC curve
    fpr_curve, tpr_curve, _ = roc_curve(yt, pp)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_curve, tpr_curve, "g-", lw=2,
            label=f"SGD (AUC = {auc:.6f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {day_name.upper()} SGD Binary")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_path = os.path.join(PLOT_DIR, f"sgd_{day_name}_v18_binary_roc.png")
    plt.savefig(roc_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {roc_path}")

    # 3. PR curve
    prec, rec, _ = precision_recall_curve(yt, pp)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, "b-", lw=2, label=f"SGD (AP = {ap:.6f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve – {day_name.upper()} SGD")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    pr_path = os.path.join(PLOT_DIR, f"sgd_{day_name}_v18_binary_pr.png")
    plt.savefig(pr_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {pr_path}")

    # 4. Feature importance 
    coef = model.coef_[0]  # shape (n_features,)
    importance = np.abs(coef)
    idx = np.argsort(importance)[::-1][:25]
    names_sorted = [feat_cols[i] for i in idx]
    values_sorted = importance[idx]
    coef_sorted = coef[idx]

    # Color by sign: red = ATTACK, blue = BENIGN
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_sorted[::-1]]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(len(names_sorted)), values_sorted[::-1], color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted[::-1], fontsize=9)
    ax.set_xlabel("|Coefficient| (linear model importance)")
    ax.set_title(f"SGD {day_name.upper()} – Top 25 Features\n"
                 "(red = pro-ATTACK, blue = pro-BENIGN)")
    for bar, c_val in zip(bars, coef_sorted[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{c_val:+.3f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(PLOT_DIR, f"sgd_{day_name}_v18_binary_features.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fi_path}")

    print(f"\n  [DONE {day_name.upper()}]  "
          f"SGD Binary Accuracy = {acc*100:.4f}%  "
          f"AUC = {auc:.6f}  "
          f"({(time.time()-t_start)/60:.1f} min)")
    return acc, auc


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python sgd_binary.py day1")
        print("  python sgd_binary.py day2")
        print("  python sgd_binary.py both")
        sys.exit(1)

    arg = sys.argv[1].lower()
    if arg == "both":
        days = ["day1", "day2"]
    elif arg in ("day1", "day2"):
        days = [arg]
    else:
        print(f"  [ERROR] Invalid: {arg}")
        sys.exit(1)

    results = {}
    for day in days:
        result = train_one_day(day)
        if result is not None:
            results[day] = result

    print("\n" + "═" * 70)
    print("  TỔNG KẾT SGD BINARY v18")
    print("═" * 70)
    for day, (acc, auc) in results.items():
        print(f"  {day}  →  Accuracy = {acc*100:.4f}%  AUC = {auc:.6f}")


if __name__ == "__main__":
    main()