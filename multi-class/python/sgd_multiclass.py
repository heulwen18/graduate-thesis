import os
import time
import warnings
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
TRAIN_PATH = "./output/train.csv"
TEST_PATH  = "./output/test.csv"
MODEL_DIR  = "./models"
PLOT_DIR   = "./plots"
OUT_DIR    = "./output"

CHUNK_SIZE  = 200_000
N_EPOCHS    = 3
RANDOM_SEED = 42
LABEL_COL   = "Label"

SGD_PARAMS = dict(
    loss         = "modified_huber",  # helping predict_proba, robust with outlier
    penalty      = "l2",              # l1 | l2 | elasticnet
    alpha        = 1e-4,              # regularization parameter
    max_iter     = 1,                 
    tol          = None,              # turn-off early stopping
    random_state = RANDOM_SEED,

    n_jobs       = -1,
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ══════════════════════════════════════════════
# PASS 1: SCAN METADATA
# ══════════════════════════════════════════════
def scan_metadata(train_path: str):

    print("─" * 60)
    print("[PASS 1]  Scan feature names & labels...")

    # feature names from header
    header_df  = pd.read_csv(train_path, nrows=0)
    all_cols   = header_df.columns.str.strip().tolist()
    feat_cols  = [c for c in all_cols if c != LABEL_COL]
    print(f"  Tổng số feature  : {len(feat_cols)}")

    # Scan all label
    all_labels = set()
    for chunk in pd.read_csv(train_path, usecols=[LABEL_COL],
                              chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        vals = chunk[LABEL_COL].dropna().astype(str).str.strip()
        all_labels.update(vals[~vals.isin({"", "nan", "Label"})].unique())

    all_classes = sorted(all_labels)
    print(f"  Số lớp           : {len(all_classes)}")
    for cls in all_classes:
        print(f"    • {cls}")

    return feat_cols, all_classes


# ══════════════════════════════════════════════
# PASS 1b: COUNT CLASS COUNTS → CALCULATE CLASS WEIGHTS
# ══════════════════════════════════════════════
def compute_class_weights(train_path: str, all_classes: list, le: LabelEncoder) -> dict:

    print("\n" + "─" * 60)
    print("[PASS 1b]  Đếm class counts để tính class weights...")

    from collections import Counter
    counts = Counter()
    for chunk in pd.read_csv(train_path, usecols=[LABEL_COL],
                              chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        vals = chunk[LABEL_COL].astype(str).str.strip()
        vals = vals[vals.isin(all_classes)]
        counts.update(vals.tolist())

    y_full = []
    for cls in all_classes:
        y_full.extend([cls] * counts[cls])
    y_enc = le.transform(y_full)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=le.transform(all_classes),
        y=y_enc,
    )
    weight_dict = dict(zip(le.transform(all_classes), weights))

    print(f"  Class weights:")
    for cls, idx in zip(all_classes, le.transform(all_classes)):
        print(f"    [{idx}] {cls:<30s}  count={counts[cls]:>10,}  weight={weight_dict[idx]:.4f}")

    return weight_dict


# ══════════════════════════════════════════════
# PASS 2: FIT SCALER ONLINE
# ══════════════════════════════════════════════
def fit_scaler(train_path: str, feat_cols: list) -> StandardScaler:

    print("\n" + "─" * 60)
    print("[PASS 2]  Fit StandardScaler (online)...")

    scaler = StandardScaler()
    reader = pd.read_csv(train_path, chunksize=CHUNK_SIZE,
                         low_memory=False, on_bad_lines="skip")
    pbar = tqdm(reader, desc="  Scaler", unit="chunk")

    for chunk in pbar:
        chunk.columns = chunk.columns.str.strip()
        X = _extract_features(chunk, feat_cols)
        if X is not None and len(X) > 0:
            scaler.partial_fit(X)

    print("  ✓ Scaler fit xong")
    return scaler


# ══════════════════════════════════════════════
# HELPER: EXTRACT & VALIDATE FEATURES
# ══════════════════════════════════════════════
def _extract_features(chunk: pd.DataFrame,
                       feat_cols: list) -> Optional[np.ndarray]:
    chunk.columns = chunk.columns.str.strip()

    for c in feat_cols:
        if c not in chunk.columns:
            chunk[c] = 0.0

    X = chunk[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    if X.empty:
        return None
    return X.values.astype(np.float32)


def _extract_labels(chunk: pd.DataFrame,
                     le: LabelEncoder,
                     all_classes: list):
    """Trích xuất và encode nhãn, bỏ qua nhãn không hợp lệ."""
    chunk.columns = chunk.columns.str.strip()
    y_raw = chunk[LABEL_COL].astype(str).str.strip()
    mask  = y_raw.isin(all_classes)
    return y_raw[mask].values, mask.values


# ══════════════════════════════════════════════
# PASS 3..N+2: TRAINING EPOCHS
# ══════════════════════════════════════════════
def train_epochs(train_path: str,
                 feat_cols: list,
                 all_classes: list,
                 scaler: StandardScaler,
                 weight_dict: dict):

    print("\n" + "─" * 60)
    print(f"[PASS 3..{N_EPOCHS+2}]  Training SGD  "
          f"({N_EPOCHS} epoch × chunk={CHUNK_SIZE:,})")
    print(f"  Params: {SGD_PARAMS}")

    le  = LabelEncoder()
    le.classes_ = np.array(all_classes)
    classes_enc = le.transform(all_classes)

    clf = SGDClassifier(**SGD_PARAMS)

    learning_curve = []   # list of dict per chunk
    t_total = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n  ── Epoch {epoch}/{N_EPOCHS} ──")
        t_epoch      = time.time()
        epoch_preds  = 0
        epoch_correct= 0

        reader = pd.read_csv(train_path, chunksize=CHUNK_SIZE,
                             low_memory=False, on_bad_lines="skip")
        pbar = tqdm(reader, desc=f"  Epoch {epoch}", unit="chunk")

        for chunk_idx, chunk in enumerate(pbar, 1):
            chunk.columns = chunk.columns.str.strip()

            # Extract valid labels
            y_raw_all = chunk[LABEL_COL].astype(str).str.strip()
            valid_mask = y_raw_all.isin(all_classes)
            chunk = chunk[valid_mask]
            if chunk.empty:
                continue

            X = _extract_features(chunk, feat_cols)
            if X is None or len(X) == 0:
                continue

            y = le.transform(chunk[LABEL_COL].astype(str).str.strip().values)

            # Scale
            X_sc = scaler.transform(X)

            # Calculate sample_weight by class weights
            sample_w = np.array([weight_dict[yi] for yi in y], dtype=np.float32)

            # Partial fit
            clf.partial_fit(X_sc, y, classes=classes_enc, sample_weight=sample_w)

            # Calculate accuracy approximately
            y_pred = clf.predict(X_sc)
            acc    = accuracy_score(y, y_pred)
            epoch_correct += (y_pred == y).sum()
            epoch_preds   += len(y)

            learning_curve.append({
                "epoch"    : epoch,
                "chunk"    : chunk_idx,
                "acc"      : acc,
            })
            pbar.set_postfix({"chunk_acc": f"{acc:.4f}"})

        epoch_acc = epoch_correct / epoch_preds if epoch_preds > 0 else 0
        print(f"  Epoch {epoch} done │ "
              f"train_acc≈{epoch_acc:.4f} │ "
              f"time={time.time()-t_epoch:.1f}s")

    print(f"\n  Tổng thời gian training : {time.time()-t_total:.1f}s")
    return clf, le, learning_curve


# ══════════════════════════════════════════════
# EVALUATION ON TEST SET
# ══════════════════════════════════════════════
def evaluate(clf, scaler, le, feat_cols, all_classes):
    print("\n" + "─" * 60)
    print("[EVAL]  Đánh giá trên test.csv (chunk-based)...")

    all_true = []
    all_pred = []

    reader = pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE,
                         low_memory=False, on_bad_lines="skip")
    pbar = tqdm(reader, desc="  Eval", unit="chunk")

    for chunk in pbar:
        chunk.columns = chunk.columns.str.strip()

        y_raw_all  = chunk[LABEL_COL].astype(str).str.strip()
        valid_mask = y_raw_all.isin(all_classes)
        chunk      = chunk[valid_mask]
        if chunk.empty:
            continue

        X = _extract_features(chunk, feat_cols)
        if X is None or len(X) == 0:
            continue

        y    = le.transform(chunk[LABEL_COL].astype(str).str.strip().values)
        X_sc = scaler.transform(X)
        yhat = clf.predict(X_sc)

        all_true.extend(y)
        all_pred.extend(yhat)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        zero_division=0,
        digits=4,
    )

    print(f"\n  Accuracy : {acc*100:.4f}%\n")
    print(report)

    # Lưu report
    report_path = os.path.join(OUT_DIR, "sgd_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc*100:.4f}%\n\n")
        f.write(report)
    print(f"  ✓ Đã lưu: {report_path}")

    return y_true, y_pred, acc


# ══════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, le):
    print("\n[PLOT]  Confusion Matrix...")
    cm  = confusion_matrix(y_true, y_pred)
    n   = len(le.classes_)
    sz  = max(8, n)

    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title("SGD – Confusion Matrix (Test Set)", fontsize=14, pad=14)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "sgd_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_learning_curve(learning_curve: list):
    print("  Learning Curve...")
    df = pd.DataFrame(learning_curve)

    fig, ax = plt.subplots(figsize=(14, 4))
    colors  = plt.cm.tab10.colors
    epochs  = sorted(df["epoch"].unique())

    for ep in epochs:
        sub  = df[df["epoch"] == ep].reset_index(drop=True)
        smooth = sub["acc"].rolling(window=30, min_periods=1).mean()
        ax.plot(sub.index, smooth,
                label=f"Epoch {ep}",
                color=colors[(ep - 1) % 10],
                linewidth=1.5)
        ax.plot(sub.index, sub["acc"],
                color=colors[(ep - 1) % 10],
                alpha=0.15, linewidth=0.8)

    ax.set_xlabel("Chunk index (trong epoch)")
    ax.set_ylabel("Train Accuracy (per chunk)")
    ax.set_title("SGD – Learning Curve  (solid = rolling avg window=30)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "sgd_learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_feature_importance(clf, feat_cols: list, top_n: int = 25):
    print("  Feature Importance...")
    mean_abs_coef = np.abs(clf.coef_).mean(axis=0)
    idx           = np.argsort(mean_abs_coef)[::-1][:top_n]

    top_names  = [feat_cols[i] for i in idx]
    top_values = mean_abs_coef[idx]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(top_n), top_values[::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |Coefficient| across classes")
    ax.set_title(f"SGD – Top {top_n} Feature Importances")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "sgd_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════
# SAVED MODEL
# ══════════════════════════════════════════════
def save_artifacts(clf, scaler, le, feat_cols):
    print("\n[SAVE]  Lưu model & artifacts...")
    joblib.dump(clf,       os.path.join(MODEL_DIR, "sgd_model.pkl"))
    joblib.dump(scaler,    os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le,        os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(feat_cols, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"  ✓ Đã lưu vào {MODEL_DIR}/")
    for f in ["sgd_model.pkl", "scaler.pkl",
              "label_encoder.pkl", "feature_names.pkl"]:
        size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
        print(f"    {f:<30s}  {size:>8.1f} KB")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  STEP 2  –  TRAIN SGDClassifier  (partial_fit / online)")
    print("=" * 60)
    print(f"  TRAIN      : {TRAIN_PATH}")
    print(f"  TEST       : {TEST_PATH}")
    print(f"  N_EPOCHS   : {N_EPOCHS}")
    print(f"  CHUNK_SIZE : {CHUNK_SIZE:,}")
    print(f"  SGD loss   : {SGD_PARAMS['loss']}")
    print(f"  SGD alpha  : {SGD_PARAMS['alpha']}")
    print(f"  SGD penalty: {SGD_PARAMS['penalty']}")

    # Pass 1: scan metadata
    feat_cols, all_classes = scan_metadata(TRAIN_PATH)

    # Pass 1b: calculate class weights
    le_tmp = LabelEncoder()
    le_tmp.classes_ = np.array(all_classes)
    weight_dict = compute_class_weights(TRAIN_PATH, all_classes, le_tmp)

    # Pass 2: fit scaler
    scaler = fit_scaler(TRAIN_PATH, feat_cols)

    # Pass 3..N+2: train
    clf, le, lc = train_epochs(TRAIN_PATH, feat_cols, all_classes, scaler, weight_dict)

    # Eval on test set
    y_true, y_pred, acc = evaluate(clf, scaler, le, feat_cols, all_classes)

    # Plots
    print("\n[PLOT]  Vẽ biểu đồ...")
    plot_confusion_matrix(y_true, y_pred, le)
    plot_learning_curve(lc)
    plot_feature_importance(clf, feat_cols, top_n=25)

    # Saving model
    save_artifacts(clf, scaler, le, feat_cols)

    print("\n" + "=" * 60)
    print(f"  [DONE]  Accuracy trên test set : {acc*100:.4f}%")
    print(f"  Model  → {MODEL_DIR}/sgd_model.pkl")
    print(f"  Plots  → {PLOT_DIR}/")
    print(f"  Report → {OUT_DIR}/sgd_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()