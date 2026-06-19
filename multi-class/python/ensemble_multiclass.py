import os
import sys
import gc
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
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
MODEL_DIR    = "./ensemble_models"
PLOT_DIR     = "./ensemble_plots"

CHUNK_SIZE   = 2_000_000
READ_WORKERS = 4
RANDOM_SEED  = 42
LABEL_COL    = "Label"

DEFAULT_WEIGHTS = {"xgb": 0.4, "lgb": 0.35, "cat": 0.25}

DAY_CONFIG = {
    "day1": dict(
        max_per_class     = 300_000,
        min_per_class     = 50_000,
        smote_threshold   = 1_000,
        weight_mode       = "sqrt",
    ),
    "day2": dict(
        max_per_class     = 400_000,
        min_per_class     = 50_000,
        smote_threshold   = 500,
        weight_mode       = "sqrt",
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


def compute_weights(y, n_classes, mode):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    max_c = counts.max()
    if mode == "balanced":
        w = max_c / (counts + 1e-9)
    elif mode == "sqrt":
        w = np.sqrt(max_c / (counts + 1e-9))
    else:
        w = np.ones(n_classes)
    return (w / w.mean())[y]


def stratified_split(y, val_frac=0.05):
    rng = np.random.default_rng(RANDOM_SEED)
    tr, vl = [], []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        rng.shuffle(pos)
        nv = max(1, int(len(pos) * val_frac))
        vl.extend(pos[:nv]); tr.extend(pos[nv:])
    return np.array(tr), np.array(vl)


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


# ══════════════════════════════════════════════
# TRAIN 3 MODELS
# ══════════════════════════════════════════════
def train_xgboost(X_tr, y_tr, X_vl, y_vl, n_classes, sw_tr):
    print("\n  [Model 1/3]  XGBoost...")
    t0 = time.time()
    params = dict(
        n_estimators          = 1000,
        learning_rate         = 0.025,
        max_depth             = 10,
        max_leaves            = 511,
        objective             = "multi:softprob",
        eval_metric           = "mlogloss",
        num_class             = n_classes,
        subsample             = 0.85,
        colsample_bytree      = 0.85,
        min_child_weight      = 2,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
        tree_method           = "hist",
        grow_policy           = "lossguide",
        max_bin               = 256,
        device                = "cpu",  
        n_jobs                = -1,
        random_state          = RANDOM_SEED,
        verbosity             = 0,
        early_stopping_rounds = 50,
    )
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr, sample_weight=sw_tr,
            eval_set=[(X_vl, y_vl)], verbose=50)
    print(f"  ✓ XGB done  ({time.time()-t0:.1f}s)  best={clf.best_iteration}")
    return clf


def train_lightgbm(X_tr, y_tr, X_vl, y_vl, n_classes, sw_tr):
    print("\n  [Model 2/3]  LightGBM...")
    t0 = time.time()
    params = dict(
        n_estimators       = 1000,
        learning_rate      = 0.03,
        num_leaves         = 255,
        max_depth          = 10,
        objective          = "multiclass",
        num_class          = n_classes,
        metric             = "multi_logloss",
        subsample          = 0.85,
        colsample_bytree   = 0.85,
        min_child_samples  = 20,
        reg_alpha          = 0.1,
        reg_lambda         = 1.5,
        max_bin            = 256,
        n_jobs             = -1,
        random_state       = RANDOM_SEED,
        verbosity          = -1,
        device             = "cpu", 
    )
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr, sample_weight=sw_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    print(f"  ✓ LGB done  ({time.time()-t0:.1f}s)  best={clf.best_iteration_}")
    return clf


def train_catboost(X_tr, y_tr, X_vl, y_vl, n_classes, sw_tr):
    print("\n  [Model 3/3]  CatBoost...")
    t0 = time.time()
    clf = CatBoostClassifier(
        iterations             = 800,
        learning_rate          = 0.05,
        depth                  = 9,
        l2_leaf_reg            = 3,
        loss_function          = "MultiClass",
        eval_metric            = "MultiClass",
        random_seed            = RANDOM_SEED,
        early_stopping_rounds  = 50,
        verbose                = 100,
        thread_count           = -1,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw_tr,
            eval_set=(X_vl, y_vl))
    print(f"  ✓ CatBoost done  ({time.time()-t0:.1f}s)  best={clf.best_iteration_}")
    return clf


def predict_proba_xgb(clf, X):
    proba = clf.predict_proba(X)
    return proba


def predict_proba_lgb(clf, X):
    proba = clf.predict_proba(X)
    return proba


def predict_proba_cat(clf, X):
    proba = clf.predict_proba(X)
    return proba


def soft_vote(probas_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    # Weighted soft voting → argmax
    assert len(probas_list) == len(weights)
    avg = np.zeros_like(probas_list[0])
    for p, w in zip(probas_list, weights):
        avg += w * p
    return np.argmax(avg, axis=1)


def compute_ensemble_weights(clf_xgb, clf_lgb, clf_cat, X_vl, y_vl):
    # Compute weights
    print("\n  Computing ensemble weights từ validation accuracy...")
    proba_xgb = clf_xgb.predict_proba(X_vl)
    proba_lgb = clf_lgb.predict_proba(X_vl)
    proba_cat = clf_cat.predict_proba(X_vl)

    acc_xgb = accuracy_score(y_vl, np.argmax(proba_xgb, axis=1))
    acc_lgb = accuracy_score(y_vl, np.argmax(proba_lgb, axis=1))
    acc_cat = accuracy_score(y_vl, np.argmax(proba_cat, axis=1))

    print(f"    XGB acc      : {acc_xgb*100:.4f}%")
    print(f"    LGB acc      : {acc_lgb*100:.4f}%")
    print(f"    CatBoost acc : {acc_cat*100:.4f}%")

    # Softmax over accuracies with low temperature
    scores = np.array([acc_xgb, acc_lgb, acc_cat])

    temperature = 50.0  
    weights = np.exp((scores - scores.mean()) * temperature)
    weights = weights / weights.sum()

    print(f"    Weights      : XGB={weights[0]:.3f}  "
          f"LGB={weights[1]:.3f}  CAT={weights[2]:.3f}")

    return weights


def train_one_day(day_name: str):
    print("\n" + "█" * 70)
    print(f"  ENSEMBLE TRAINING {day_name.upper()}  (XGB + LGB + CatBoost)")
    print("█" * 70)
    t_start = time.time()

    cfg = DAY_CONFIG[day_name]
    day_dir = os.path.join(OUTPUT_BASE, day_name)
    train_path = os.path.join(day_dir, "train.csv")
    test_path  = os.path.join(day_dir, "test.csv")

    if not os.path.exists(train_path):
        print(f"  [ERROR] {train_path} không tồn tại")
        return None

    # Scan
    print(f"\n[1/7]  Scan {day_name}...")
    header = pd.read_csv(train_path, nrows=0)
    file_cols = set(header.columns.str.strip())
    base_feats = [f for f in BASE_FEATURES if f in file_cols]
    port_feats = [f for f in PORT_FEATURES if f in file_cols]
    print(f"  Base / Port features: {len(base_feats)} / {len(port_feats)}")
    all_load_feats = base_feats + port_feats

    label_counts = Counter()
    for chunk in pd.read_csv(train_path, usecols=[LABEL_COL],
                              chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        vals = chunk[LABEL_COL].astype(str).str.strip()
        vals = vals[~vals.isin({"", "nan", "Label"})]
        label_counts.update(vals.tolist())
    all_classes = sorted(label_counts.keys())
    print(f"  Classes: {len(all_classes)}")

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

    # Prep matrices
    print(f"\n[4/7]  Prep matrices...")
    le = LabelEncoder()
    le.fit(all_classes)
    X = df_bal[feat_cols].values.astype(np.float32)
    y = le.transform(df_bal[LABEL_COL].values)
    sw = compute_weights(y, len(all_classes), cfg["weight_mode"])
    tr, vl = stratified_split(y)
    X_tr, y_tr, sw_tr = X[tr], y[tr], sw[tr]
    X_vl, y_vl        = X[vl], y[vl]
    print(f"  Train: {len(X_tr):,}  Val: {len(X_vl):,}")
    del df_bal, X
    gc.collect()

    # Train 3 models
    print(f"\n[5/7]  Train 3 base models...")
    clf_xgb = train_xgboost(X_tr, y_tr, X_vl, y_vl, len(all_classes), sw_tr)
    clf_lgb = train_lightgbm(X_tr, y_tr, X_vl, y_vl, len(all_classes), sw_tr)
    clf_cat = train_catboost(X_tr, y_tr, X_vl, y_vl, len(all_classes), sw_tr)

    # Compute ensemble weights
    weights = compute_ensemble_weights(clf_xgb, clf_lgb, clf_cat, X_vl, y_vl)
    del X_tr, y_tr, sw_tr, X_vl, y_vl, sw
    gc.collect()

    # Evaluate
    print(f"\n[6/7]  Evaluate ensemble trên test set...")
    t0 = time.time()
    all_true = []
    all_pred_ens = []
    all_pred_xgb = []
    all_pred_lgb = []
    all_pred_cat = []

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
        X = chunk[feat_cols].values.astype(np.float32)
        y = le.transform(chunk[LABEL_COL].values)

        proba_xgb = clf_xgb.predict_proba(X)
        proba_lgb = clf_lgb.predict_proba(X)
        proba_cat = clf_cat.predict_proba(X)

        yp_ens = soft_vote([proba_xgb, proba_lgb, proba_cat], weights)
        yp_xgb = np.argmax(proba_xgb, axis=1)
        yp_lgb = np.argmax(proba_lgb, axis=1)
        yp_cat = np.argmax(proba_cat, axis=1)

        all_true.extend(y)
        all_pred_ens.extend(yp_ens)
        all_pred_xgb.extend(yp_xgb)
        all_pred_lgb.extend(yp_lgb)
        all_pred_cat.extend(yp_cat)

    yt = np.array(all_true)
    yp_ens = np.array(all_pred_ens)
    yp_xgb = np.array(all_pred_xgb)
    yp_lgb = np.array(all_pred_lgb)
    yp_cat = np.array(all_pred_cat)

    # Report for each model + ensemble
    print(f"\n{'='*65}")
    print(f"  PER-MODEL ACCURACY ON TEST SET")
    print(f"{'='*65}")
    acc_xgb = accuracy_score(yt, yp_xgb)
    acc_lgb = accuracy_score(yt, yp_lgb)
    acc_cat = accuracy_score(yt, yp_cat)
    acc_ens = accuracy_score(yt, yp_ens)
    print(f"  XGBoost  : {acc_xgb*100:.4f}%")
    print(f"  LightGBM : {acc_lgb*100:.4f}%")
    print(f"  CatBoost : {acc_cat*100:.4f}%")
    print(f"  ENSEMBLE : {acc_ens*100:.4f}%  ← final")

    # Detailed ensemble report
    report = classification_report(yt, yp_ens, target_names=le.classes_,
                                    zero_division=0, digits=4)
    f1s = f1_score(yt, yp_ens, average=None, zero_division=0)
    print(f"\n  Macro F1 (ensemble): {f1s.mean():.4f}")
    print(f"  ({time.time()-t0:.1f}s)\n")
    print(report)

    print("  Per-class F1 (ensemble):")
    for i, cls in enumerate(le.classes_):
        bar = "█" * int(f1s[i] * 30)
        flag = " ←" if f1s[i] < 0.7 else ""
        print(f"    {cls:<25s}  {f1s[i]:.4f}  {bar}{flag}")

    # Save
    print(f"\n[7/7]  Save artifacts...")
    clf_xgb.save_model(os.path.join(MODEL_DIR, f"xgb_{day_name}_v14_xgb.ubj"))
    joblib.dump(clf_lgb,   os.path.join(MODEL_DIR, f"xgb_{day_name}_v14_lgb.pkl"))
    clf_cat.save_model(    os.path.join(MODEL_DIR, f"xgb_{day_name}_v14_cat.cbm"))
    joblib.dump(le,        os.path.join(MODEL_DIR, f"xgb_{day_name}_v14_le.pkl"))
    joblib.dump(feat_cols, os.path.join(MODEL_DIR, f"xgb_{day_name}_v14_features.pkl"))
    joblib.dump(weights,   os.path.join(MODEL_DIR, f"xgb_{day_name}_v14_weights.pkl"))

    report_path = os.path.join(OUTPUT_BASE,
                                f"xgb_{day_name}_v14_ensemble_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Day: {day_name} (ENSEMBLE XGB+LGB+CAT)\n")
        f.write(f"Weights: XGB={weights[0]:.4f}  "
                f"LGB={weights[1]:.4f}  CAT={weights[2]:.4f}\n\n")
        f.write(f"=== Per-model accuracy ===\n")
        f.write(f"XGBoost  : {acc_xgb*100:.4f}%\n")
        f.write(f"LightGBM : {acc_lgb*100:.4f}%\n")
        f.write(f"CatBoost : {acc_cat*100:.4f}%\n")
        f.write(f"ENSEMBLE : {acc_ens*100:.4f}%\n")
        f.write(f"Macro F1 : {f1s.mean():.4f}\n\n")
        f.write("=== Ensemble classification report ===\n")
        f.write(report)
    print(f"  ✓ {report_path}")

    # Plot confusion matrix
    cm = confusion_matrix(yt, yp_ens)
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    n = len(le.classes_)
    fig, axes = plt.subplots(1, 2, figsize=(n * 2 + 2, n + 1))
    for ax, data, title, fmt in zip(
        axes, [cm, cmn], ["Count", "Normalized"], [".0f", ".2f"]
    ):
        disp = ConfusionMatrixDisplay(data, display_labels=le.classes_)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=True,
                  cmap="Blues", values_format=fmt)
        ax.set_title(f"Ensemble {day_name.upper()} v14 – {title}",
                     fontsize=11, pad=10)
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, f"xgb_{day_name}_v14_confusion_matrix.png")
    plt.savefig(cm_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {cm_path}")

    # Comparison plot: per-model accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["XGBoost", "LightGBM", "CatBoost", "ENSEMBLE"]
    accs   = [acc_xgb*100, acc_lgb*100, acc_cat*100, acc_ens*100]
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22"]
    bars = ax.bar(models, accs, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Day {day_name[-1]} v14 – Model Comparison")
    ax.set_ylim(min(accs) - 2, max(accs) + 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{acc:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    cmp_path = os.path.join(PLOT_DIR, f"xgb_{day_name}_v14_model_comparison.png")
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {cmp_path}")

    print(f"\n  [DONE {day_name.upper()}]  "
          f"Ensemble Accuracy = {acc_ens*100:.4f}%  "
          f"({(time.time()-t_start)/60:.1f} min)")
    return acc_ens


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python step2_train_ensemble_v14.py day1")
        print("  python step2_train_ensemble_v14.py day2")
        print("  python step2_train_ensemble_v14.py both")
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
        acc = train_one_day(day)
        results[day] = acc

    print("\n" + "═" * 70)
    print("  TỔNG KẾT v14 (ENSEMBLE)")
    print("═" * 70)
    for day, acc in results.items():
        if acc is not None:
            print(f"  {day}  →  Ensemble Accuracy = {acc*100:.4f}%")


if __name__ == "__main__":
    main()