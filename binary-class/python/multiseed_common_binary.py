"""
multiseed_common_binary.py
==========================
Module PROCESSING DATA for binary multi-seed.

Split 70/10/20 stratified.

"""

import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from multiseed_common import (
    SEEDS, RESULTS_DIR, LABEL_COL,
    BASE_FEATURES, PORT_FEATURES, ENGINEERED_FEATURES,
    engineer_features, apply_log, split_3way, _load_csv, OUTPUT_BASE,
)

# ── Config ──
DAY_CONFIG_BIN = {
    "day1": dict(max_attack_samples=2_000_000, max_benign_samples=100_000,
                 min_benign_samples=100_000, dropout=0.3, weight_decay=1e-5),
    "day2": dict(max_attack_samples=1_500_000, max_benign_samples=100_000,
                 min_benign_samples=100_000, dropout=0.25, weight_decay=1e-5),
}


def to_binary_label(s):
    return (s != "BENIGN").astype(np.int64)


def balance_binary(df, feat_cols, cfg, seed):
    """Downsample ATTACK, upsample BENIGN """
    rng = np.random.default_rng(seed)
    df_benign = df[df["binary_label"] == 0].copy()
    df_attack = df[df["binary_label"] == 1].copy()

    max_attack = cfg["max_attack_samples"]
    if len(df_attack) > max_attack:
        df_attack = df_attack.sample(max_attack, random_state=seed)

    min_benign = cfg["min_benign_samples"]
    if len(df_benign) < min_benign:
        n = len(df_benign)
        factor = int(np.ceil(min_benign / n))
        rep = pd.concat([df_benign] * factor, ignore_index=True).iloc[:min_benign]
        num_cols = [c for c in rep.columns if c not in {LABEL_COL, "binary_label"}]
        noise = rng.normal(0, 1e-4, size=(len(rep), len(num_cols)))
        rep[num_cols] = rep[num_cols].values + noise
        df_benign = rep
    elif len(df_benign) > cfg["max_benign_samples"]:
        df_benign = df_benign.sample(cfg["max_benign_samples"], random_state=seed)

    out = pd.concat([df_benign, df_attack], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_day_data_bin(day_name):

    train_path = os.path.join(OUTPUT_BASE, day_name, "train.csv")
    test_path  = os.path.join(OUTPUT_BASE, day_name, "test.csv")
    header = pd.read_csv(train_path, nrows=0)
    file_cols = set(header.columns.str.strip())
    base_feats = [f for f in BASE_FEATURES if f in file_cols]
    port_feats = [f for f in PORT_FEATURES if f in file_cols]
    all_load = base_feats + port_feats

    print(f"  Load + gộp train + test ({day_name})...")
    df1 = _load_csv(train_path, all_load, desc="train")
    df2 = _load_csv(test_path,  all_load, desc="test")
    df_full = pd.concat([df1, df2], ignore_index=True)
    del df1, df2; gc.collect()

    df_full["binary_label"] = to_binary_label(df_full[LABEL_COL])
    df_full = engineer_features(df_full)
    df_full = apply_log(df_full)
    new_eng = [c for c in ENGINEERED_FEATURES if c in df_full.columns]
    feat_cols = base_feats + port_feats + new_eng

    n_b = int((df_full["binary_label"] == 0).sum())
    n_a = int((df_full["binary_label"] == 1).sum())
    print(f"  Total: {len(df_full):,} rows  {len(feat_cols)} feats  "
          f"(BENIGN={n_b:,}  ATTACK={n_a:,})")
    return df_full, feat_cols


def make_split_bin(df_full, feat_cols, cfg, seed):
    """
    Split 70/10/20 stratified by binary_label + balance train.
    """
    y_bin = df_full["binary_label"].values
    tr_idx, vl_idx, te_idx = split_3way(y_bin, seed)

    # Balance
    df_tr_bal = balance_binary(df_full.iloc[tr_idx], feat_cols, cfg, seed)
    Xr_tr = df_tr_bal[feat_cols].values.astype(np.float32)
    y_tr  = df_tr_bal["binary_label"].values.astype(np.int64)
    del df_tr_bal

    Xr_vl = df_full.iloc[vl_idx][feat_cols].values.astype(np.float32)
    y_vl  = y_bin[vl_idx].astype(np.int64)
    Xr_te = df_full.iloc[te_idx][feat_cols].values.astype(np.float32)
    y_te  = y_bin[te_idx].astype(np.int64)

    # Scale
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(Xr_tr).astype(np.float32)
    Xs_vl = scaler.transform(Xr_vl).astype(np.float32)
    Xs_te = scaler.transform(Xr_te).astype(np.float32)
    del Xr_tr, Xr_vl, Xr_te

    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    return dict(Xs_tr=Xs_tr, y_tr=y_tr, Xs_vl=Xs_vl, y_vl=y_vl,
                Xs_te=Xs_te, y_te=y_te,
                scale_pos_weight=scale_pos_weight,
                n_features=Xs_tr.shape[1],
                n_train=len(tr_idx), n_val=len(vl_idx), n_test=len(te_idx))


def binary_metrics(model_name, day, seed, y_te, proba, threshold=0.5):
    """Calculate metrics binary by proba + threshold."""
    yp = (proba > threshold).astype(np.int8)
    cm = confusion_matrix(y_te, yp, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    try:
        auc = roc_auc_score(y_te, proba)
    except Exception:
        auc = float("nan")
    return dict(
        model=model_name, day=day, seed=seed, threshold=threshold,
        accuracy=accuracy_score(y_te, yp),
        precision=precision_score(y_te, yp, zero_division=0),
        recall=recall_score(y_te, yp, zero_division=0),
        f1=f1_score(y_te, yp, zero_division=0),
        fpr=fpr, fnr=fnr, auc=auc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def append_result_bin(model_key, day, row):
    path = os.path.join(RESULTS_DIR, f"binary_{model_key}_{day}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["seed"] != row["seed"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("seed").reset_index(drop=True)
    df.to_csv(path, index=False)


def summarize_bin(model_key, day):
    path = os.path.join(RESULTS_DIR, f"binary_{model_key}_{day}.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    print(f"\n  {'='*60}")
    print(f"  {model_key} — {day} ({len(d)} seeds)")
    print(f"  {'='*60}")
    for m in ["accuracy", "precision", "recall", "f1", "fpr", "fnr", "auc"]:
        mean, std = d[m].mean(), d[m].std()
        if m in ("accuracy", "fpr", "fnr"):
            print(f"  {m:<10s}: {mean*100:.4f}% ± {std*100:.4f}%")
        else:
            print(f"  {m:<10s}: {mean:.4f} ± {std:.4f}")