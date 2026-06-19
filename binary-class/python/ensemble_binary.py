"""
====================================
4-Model Ensemble cho BINARY DDoS detection:
  XGBoost + LightGBM + CatBoost + 1D CNN

Voting strategy: SOFT VOTING with learned weights
  - Each model predict P(ATTACK)
  - Ensemble = weighted avg(P)
  - Weights = softmax(val_accuracy * temperature)

"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

OUTPUT_BASE = "./output"
MODEL_DIR   = "./ensemble_models"
CHUNK_SIZE  = 2_000_000
LABEL_COL   = "Label"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FEATURES = {
    "Subflow Fwd Bytes", "Flow Bytes/s", "Fwd Packets/s", "Bwd Packets/s",
    "Flow IAT Mean", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Max",
    "Bwd IAT Min", "Flow Duration", "Fwd Header Length", "Bwd Header Length",
    "Active Mean", "Active Max", "Active Min", "Idle Max",
}


class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class DDoSBinaryConv1D(nn.Module):
    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.block1 = Conv1DBlock(1, 64, 3, dropout)
        self.block2 = Conv1DBlock(64, 128, 3, dropout)
        self.block3 = Conv1DBlock(128, 256, 3, dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256 * 2, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        avg = self.gap(x).squeeze(-1); mx = self.gmp(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)
        x = self.dropout_fc(F.relu(self.bn_fc(self.fc1(x))))
        return self.fc2(x)


def safe_div(a, b):
    return (a / (b.abs() + 1e-9)).clip(-1e9, 1e9).fillna(0)


def engineer_features(df):
    cols = df.columns
    if "Subflow Fwd Bytes" in cols and "Subflow Fwd Packets" in cols:
        df["bytes_per_fwd_packet"] = safe_div(df["Subflow Fwd Bytes"], df["Subflow Fwd Packets"])
        df["subflow_density"] = safe_div(df["Subflow Fwd Bytes"], df["Subflow Fwd Packets"] + 1)
    if "Subflow Fwd Bytes" in cols and "Subflow Bwd Bytes" in cols:
        df["fwd_bwd_bytes_ratio"] = safe_div(df["Subflow Fwd Bytes"], df["Subflow Bwd Bytes"])
    if "Packet Length Mean" in cols and "Packet Length Std" in cols:
        df["packet_length_cv"] = safe_div(df["Packet Length Std"], df["Packet Length Mean"])
    if "ACK Flag Count" in cols and "Subflow Fwd Packets" in cols:
        total = df["Subflow Fwd Packets"] + df.get("Subflow Bwd Packets", 0)
        df["ack_flag_ratio"] = safe_div(df["ACK Flag Count"], total)
        df["ack_flag_squared"] = df["ACK Flag Count"] ** 2
    if "Init_Win_bytes_forward" in cols and "Init_Win_bytes_backward" in cols:
        df["win_size_ratio"] = safe_div(df["Init_Win_bytes_forward"], df["Init_Win_bytes_backward"])
    if "Fwd Header Length" in cols and "Bwd Header Length" in cols:
        df["header_length_ratio"] = safe_div(df["Fwd Header Length"], df["Bwd Header Length"])
    if "Bwd Packet Length Mean" in cols and "Packet Length Mean" in cols:
        df["amplification_factor"] = safe_div(df["Bwd Packet Length Mean"], df["Packet Length Mean"])
    if "Fwd Packets/s" in cols and "Bwd Packets/s" in cols:
        df["burst_intensity"] = safe_div(df["Fwd Packets/s"], df["Bwd Packets/s"] + 1)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def apply_log(df):
    cols = [c for c in LOG_FEATURES if c in df.columns]
    df[cols] = np.log1p(df[cols].clip(lower=0))
    return df


def to_binary_label(s):
    return (s != "BENIGN").astype(np.int64)


def predict_proba_cnn(model, X, batch_size=4096):
    model.eval()
    probas = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(DEVICE)
            out = model(batch).squeeze(-1)
            probas[i:i+batch_size] = torch.sigmoid(out).cpu().numpy()
    return probas


def full_metrics(yt, yp):
    """Tính FULL metrics từ y_true và y_pred."""
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
    # ATTACK = positive class
    prec_a = tp / (tp + fp) if (tp + fp) else 0.0
    rec_a  = tp / (tp + fn) if (tp + fn) else 0.0   # = TPR = recall ATTACK
    f1_a   = 2*prec_a*rec_a/(prec_a+rec_a) if (prec_a+rec_a) else 0.0
    # BENIGN = negative class
    prec_b = tn / (tn + fn) if (tn + fn) else 0.0
    rec_b  = tn / (tn + fp) if (tn + fp) else 0.0   # = TNR = specificity
    f1_b   = 2*prec_b*rec_b/(prec_b+rec_b) if (prec_b+rec_b) else 0.0
    macro_f1 = (f1_a + f1_b) / 2
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    # weighted (theo support)
    n = tp + tn + fp + fn
    n_a = tp + fn; n_b = tn + fp
    w_prec = (prec_a*n_a + prec_b*n_b) / n if n else 0.0
    w_rec  = (rec_a*n_a + rec_b*n_b) / n if n else 0.0
    w_f1   = (f1_a*n_a + f1_b*n_b) / n if n else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, acc=acc,
                prec_a=prec_a, rec_a=rec_a, f1_a=f1_a,
                prec_b=prec_b, rec_b=rec_b, f1_b=f1_b,
                macro_f1=macro_f1, fpr=fpr, fnr=fnr,
                w_prec=w_prec, w_rec=w_rec, w_f1=w_f1,
                specificity=rec_b)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ensemble_voting.py {day1|day2}")
        sys.exit(1)
    day_name = sys.argv[1].lower()

    print("█" * 70)
    print(f"  FULL REPORT ENSEMBLE VOTING ")
    print("█" * 70)

    prefix = os.path.join(MODEL_DIR, f"ens_binary_{day_name}")

    # Load models + artifacts
    print(f"\n[1/4]  Load models + weights + thresholds...")
    clf_xgb = xgb.XGBClassifier(); clf_xgb.load_model(f"{prefix}_xgb.ubj")
    clf_lgb = joblib.load(f"{prefix}_lgb.pkl")
    clf_cat = CatBoostClassifier(); clf_cat.load_model(f"{prefix}_cat.cbm")
    cnn_ckpt = torch.load(f"{prefix}_cnn.pt", map_location=DEVICE)
    cnn_model = DDoSBinaryConv1D(cnn_ckpt["n_features"], cnn_ckpt["dropout"]).to(DEVICE)
    cnn_model.load_state_dict(cnn_ckpt["model_state"]); cnn_model.eval()
    scaler    = joblib.load(f"{prefix}_scaler.pkl")
    feat_cols = joblib.load(f"{prefix}_features.pkl")
    weights   = joblib.load(f"{prefix}_weights.pkl")
    thresholds = joblib.load(f"{prefix}_per_model_thresholds.pkl")
    thr_ens   = joblib.load(f"{prefix}_ensemble_threshold.pkl")

    names = ["xgb", "lgb", "cat", "cnn"]
    name_disp = {"xgb": "XGBoost", "lgb": "LightGBM",
                 "cat": "CatBoost", "cnn": "1D CNN"}
    print(f"  Weights: " + "  ".join(
        f"{name_disp[k]}={weights[i]:.4f}" for i, k in enumerate(names)))
    print(f"  Per-model thr: " + "  ".join(
        f"{name_disp[k]}={thresholds[k]:.4f}" for k in names))
    print(f"  Ensemble thr: {thr_ens:.4f}")

    # Predict full test
    print(f"\n[2/4]  Predict test.csv...")
    test_path = os.path.join(OUTPUT_BASE, day_name, "test.csv")
    store = {k: [] for k in names}
    yt_all = []
    reader = pd.read_csv(test_path, chunksize=CHUNK_SIZE,
                          low_memory=False, on_bad_lines="skip")
    for chunk in tqdm(reader, desc="  Predict", unit="chunk"):
        chunk.columns = chunk.columns.str.strip()
        chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()
        chunk = chunk[~chunk[LABEL_COL].isin({"", "nan", "Label"})]
        if chunk.empty:
            continue
        y = to_binary_label(chunk[LABEL_COL]).values
        num_cols = [c for c in chunk.columns if c != LABEL_COL]
        chunk[num_cols] = chunk[num_cols].apply(pd.to_numeric, errors="coerce")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)
        chunk = engineer_features(chunk); chunk = apply_log(chunk)
        for c in feat_cols:
            if c not in chunk.columns:
                chunk[c] = 0.0
        X = scaler.transform(chunk[feat_cols].values.astype(np.float32)).astype(np.float32)
        store["xgb"].append(clf_xgb.predict_proba(X)[:, 1].astype(np.float32))
        store["lgb"].append(clf_lgb.predict_proba(X)[:, 1].astype(np.float32))
        store["cat"].append(clf_cat.predict_proba(X)[:, 1].astype(np.float32))
        store["cnn"].append(predict_proba_cnn(cnn_model, X).astype(np.float32))
        yt_all.append(y.astype(np.int8))
    for k in names:
        store[k] = np.concatenate(store[k])
    yt = np.concatenate(yt_all)

    # Regenerate EVAL half
    print(f"\n[3/4]  Tái tạo EVAL half (seed={RANDOM_SEED})...")
    rng = np.random.default_rng(RANDOM_SEED)
    tune_mask = np.zeros(len(yt), dtype=bool)
    for c in [0, 1]:
        idx = np.where(yt == c)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        tune_mask[idx[:half]] = True
    eval_mask = ~tune_mask
    yt_e = yt[eval_mask]
    n_b = int((yt_e == 0).sum()); n_a = int((yt_e == 1).sum())
    print(f"  EVAL: {len(yt_e):,}  (BENIGN={n_b:,}  ATTACK={n_a:,})")

    # AUC/AP (threshold-independent) on EVAL
    aucs = {k: roc_auc_score(yt_e, store[k][eval_mask]) for k in names}
    aps  = {k: average_precision_score(yt_e, store[k][eval_mask]) for k in names}

    # Per-model metrics
    print(f"\n[4/4]  Tính full metrics trên EVAL half...\n")
    results = {}
    for k in names:
        yp = (store[k][eval_mask] > thresholds[k]).astype(np.int8)
        results[k] = full_metrics(yt_e, yp)
        results[k]["auc"] = aucs[k]
        results[k]["ap"]  = aps[k]

    # Ensemble
    p_ens = sum(weights[i] * store[names[i]][eval_mask] for i in range(4))
    yp_ens = (p_ens > thr_ens).astype(np.int8)
    results["ens"] = full_metrics(yt_e, yp_ens)
    results["ens"]["auc"] = roc_auc_score(yt_e, p_ens)
    results["ens"]["ap"]  = average_precision_score(yt_e, p_ens)
    name_disp["ens"] = "ENSEMBLE"

    # Summary table
    order = ["xgb", "lgb", "cat", "cnn", "ens"]
    print(f"  {'Model':<10s} {'Acc':>9s} {'Prec':>8s} {'Rec':>8s} "
          f"{'F1':>8s} {'FPR':>8s} {'FNR':>10s} {'AUC':>8s}")
    print(f"  {'-'*74}")
    for k in order:
        r = results[k]
        marker = "  ←" if k == "ens" else ""
        print(f"  {name_disp[k]:<10s} {r['acc']*100:>8.4f}% "
              f"{r['w_prec']:>8.4f} {r['w_rec']:>8.4f} {r['w_f1']:>8.4f} "
              f"{r['fpr']*100:>7.3f}% {r['fnr']*100:>9.4f}% "
              f"{r['auc']:>8.4f}{marker}")

    # Ensemble detail
    r = results["ens"]
    print(f"\n  {'='*60}")
    print(f"  ENSEMBLE threshold={thr_ens:.4f}")
    print(f"  {'='*60}")
    print(f"    Accuracy    : {r['acc']*100:.4f}%")
    print(f"    Precision   : {r['w_prec']:.4f} (weighted)")
    print(f"    Recall      : {r['w_rec']:.4f} (weighted)")
    print(f"    F1          : {r['w_f1']:.4f} (weighted)")
    print(f"    Macro F1    : {r['macro_f1']:.4f}")
    print(f"    FPR         : {r['fpr']*100:.4f}%")
    print(f"    FNR         : {r['fnr']*100:.6f}%")
    print(f"    Specificity : {r['specificity']:.4f}")
    print(f"    ROC-AUC     : {r['auc']:.4f}")
    print(f"    PR-AUC      : {r['ap']:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"                    Pred BENIGN   Pred ATTACK")
    print(f"      True BENIGN  {r['tn']:>12,}  {r['fp']:>12,}")
    print(f"      True ATTACK  {r['fn']:>12,}  {r['tp']:>12,}")

    print(f"\n    Classification report (ENSEMBLE):")
    print(f"    {'':12s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>10s}")
    print(f"    {'BENIGN':<12s}{r['prec_b']:>10.4f}{r['rec_b']:>10.4f}"
          f"{r['f1_b']:>10.4f}{n_b:>10,}")
    print(f"    {'ATTACK':<12s}{r['prec_a']:>10.4f}{r['rec_a']:>10.4f}"
          f"{r['f1_a']:>10.4f}{n_a:>10,}")
    print(f"    {'accuracy':<12s}{'':>10s}{'':>10s}{r['acc']:>10.4f}{n_b+n_a:>10,}")
    print(f"    {'macro avg':<12s}{(r['prec_a']+r['prec_b'])/2:>10.4f}"
          f"{(r['rec_a']+r['rec_b'])/2:>10.4f}{r['macro_f1']:>10.4f}{n_b+n_a:>10,}")

    # Save report
    rp = os.path.join(OUTPUT_BASE, f"{day_name}_ensemble_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(f"FULL REPORT ENSEMBLE VOTING — {day_name}\n{'='*70}\n\n")
        f.write(f"Weights: " + "  ".join(
            f"{name_disp[k]}={weights[i]:.4f}" for i, k in enumerate(names)) + "\n")
        f.write(f"Ensemble threshold: {thr_ens:.4f}\n\n")
        f.write(f"{'Model':<10s} {'Acc':>9s} {'Prec':>8s} {'Rec':>8s} "
                f"{'F1':>8s} {'FPR':>8s} {'FNR':>10s} {'AUC':>8s} {'thr':>7s}\n")
        f.write(f"{'-'*84}\n")
        for k in order:
            r = results[k]
            thr = thr_ens if k == "ens" else thresholds[k]
            f.write(f"{name_disp[k]:<10s} {r['acc']*100:>8.4f}% "
                    f"{r['w_prec']:>8.4f} {r['w_rec']:>8.4f} {r['w_f1']:>8.4f} "
                    f"{r['fpr']*100:>7.3f}% {r['fnr']*100:>9.4f}% "
                    f"{r['auc']:>8.4f} {thr:>7.4f}\n")
        f.write(f"\n=== ENSEMBLE details ===\n")
        r = results["ens"]
        f.write(f"Accuracy   : {r['acc']*100:.4f}%\n")
        f.write(f"Precision  : {r['w_prec']:.4f}\n")
        f.write(f"Recall     : {r['w_rec']:.4f}\n")
        f.write(f"F1         : {r['w_f1']:.4f}\n")
        f.write(f"Macro F1   : {r['macro_f1']:.4f}\n")
        f.write(f"FPR        : {r['fpr']*100:.4f}%\n")
        f.write(f"FNR        : {r['fnr']*100:.6f}%\n")
        f.write(f"ROC-AUC    : {r['auc']:.4f}\n")
        f.write(f"PR-AUC     : {r['ap']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TN={r['tn']:,}  FP={r['fp']:,}\n")
        f.write(f"  FN={r['fn']:,}  TP={r['tp']:,}\n")
        f.write(f"\nClassification report:\n")
        f.write(f"{'':12s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>10s}\n")
        f.write(f"{'BENIGN':<12s}{r['prec_b']:>10.4f}{r['rec_b']:>10.4f}"
                f"{r['f1_b']:>10.4f}{n_b:>10,}\n")
        f.write(f"{'ATTACK':<12s}{r['prec_a']:>10.4f}{r['rec_a']:>10.4f}"
                f"{r['f1_a']:>10.4f}{n_a:>10,}\n")
        f.write(f"{'accuracy':<12s}{'':>10s}{'':>10s}{r['acc']:>10.4f}{n_b+n_a:>10,}\n")
        f.write(f"{'macro avg':<12s}{(r['prec_a']+r['prec_b'])/2:>10.4f}"
                f"{(r['rec_a']+r['rec_b'])/2:>10.4f}{r['macro_f1']:>10.4f}{n_b+n_a:>10,}\n")
    print(f"\n  ✓ Saved: {rp}")


if __name__ == "__main__":
    main()