"""
Usage:
    python multiseed_ensemble_verbose.py day2
    python multiseed_ensemble_verbose.py day1
    python multiseed_ensemble_verbose.py both
"""

import os
import sys
import gc
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score)
from tqdm import tqdm

warnings.filterwarnings("ignore")

OUTPUT_BASE  = "./output"
RESULTS_DIR  = "./results_multiseed"
CHUNK_SIZE   = 2_000_000
READ_WORKERS = 4
LABEL_COL    = "Label"

SEEDS = [42, 52, 62, 72, 82]
TEST_FRAC = 0.20
VAL_FRAC  = 0.10
MODEL_NAME = "Ensemble(XGB+LGB+CAT)"

N_EST = 800

DAY_CONFIG = {
    "day1": dict(max_per_class=300_000, min_per_class=50_000,
                 smote_threshold=1_000, weight_mode="sqrt",
                 xgb_depth=10, xgb_lr=0.025, cat_depth=9),
    "day2": dict(max_per_class=400_000, min_per_class=50_000,
                 smote_threshold=500, weight_mode="sqrt",
                 xgb_depth=9, xgb_lr=0.03, cat_depth=9),
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
    "is_port_chargen", "is_port_rpc", "port_category", "src_port_log",
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

os.makedirs(RESULTS_DIR, exist_ok=True)


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


def smote_lite(sub, target_n, feat_cols, rng):
    n = len(sub); n_gen = target_n - n
    num_cols = [c for c in feat_cols if c in sub.columns]
    X = sub[num_cols].values.astype(np.float32)
    # Vectorized SMOTE 
    ii = rng.integers(0, n, size=n_gen)
    jj = rng.integers(0, n, size=n_gen)
    alpha = rng.random((n_gen, 1)).astype(np.float32)
    syn_X = X[ii] + alpha * (X[jj] - X[ii])
    syn = pd.DataFrame(syn_X, columns=num_cols)
    syn[LABEL_COL] = sub[LABEL_COL].iloc[0]
    return pd.concat([sub, syn], ignore_index=True)


def balance_df(df, feat_cols, cfg, seed):
    rng = np.random.default_rng(seed)
    max_n, min_n = cfg["max_per_class"], cfg["min_per_class"]
    smote_thr = cfg["smote_threshold"]
    frames = []
    for cls in df[LABEL_COL].unique():
        sub = df[df[LABEL_COL] == cls].copy()
        n = len(sub)
        if n == 0:
            continue
        if n > max_n:
            sub = sub.sample(max_n, random_state=seed)
        elif n < smote_thr:
            target = min(min_n, n * 20)
            sub = smote_lite(sub, target, feat_cols, rng)
        elif n < min_n:
            target = min(min_n, n * 5)
            factor = int(np.ceil(target / n))
            sub_rep = pd.concat([sub] * factor, ignore_index=True).iloc[:target]
            num_cols = [c for c in sub_rep.columns if c != LABEL_COL]
            noise = rng.normal(0, 1e-4, size=(len(sub_rep), len(num_cols)))
            sub_rep[num_cols] = sub_rep[num_cols].values + noise
            sub = sub_rep
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


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


def split_3way(y, test_frac, val_frac, seed):
    rng = np.random.default_rng(seed)
    tr, vl, te = [], [], []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        rng.shuffle(pos)
        n = len(pos)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        te.extend(pos[:n_test])
        vl.extend(pos[n_test:n_test + n_val])
        tr.extend(pos[n_test + n_val:])
    return np.array(tr), np.array(vl), np.array(te)


def _read_worker(args):
    path, skiprows, nrows, col_names, all_feats = args
    try:
        df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, header=None,
                          names=col_names, low_memory=False, on_bad_lines="skip")
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
        for fut in as_completed(fmap):
            frames[fmap[fut]] = fut.result(); pbar.update(1)
    pbar.close()
    valid = [f for f in frames if f is not None and not f.empty]
    return pd.concat(valid, ignore_index=True, copy=False)


def train_xgb(X_tr, y_tr, X_vl, y_vl, n_cls, sw, cfg, seed):
    print(f"      Training XGBoost ...", end="", flush=True)
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=N_EST, learning_rate=cfg["xgb_lr"], max_depth=cfg["xgb_depth"],
        max_leaves=511, objective="multi:softprob", eval_metric="mlogloss",
        num_class=n_cls, subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        reg_alpha=0.1, reg_lambda=1.5, tree_method="hist", grow_policy="lossguide",
        max_bin=256, device="cpu", n_jobs=-1, random_state=seed, verbosity=0,
        early_stopping_rounds=40)
    clf.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_vl, y_vl)], verbose=False)
    print(f" xong ({time.time()-t0:.0f}s, best_iter={clf.best_iteration})")
    return clf


def train_lgb(X_tr, y_tr, X_vl, y_vl, n_cls, sw, seed):
    print(f"      Training LightGBM ...", end="", flush=True)
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=N_EST, learning_rate=0.03, num_leaves=255, max_depth=10,
        objective="multiclass", num_class=n_cls, metric="multi_logloss",
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.5, max_bin=256, n_jobs=-1,
        random_state=seed, verbosity=-1, device="cpu")
    clf.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    print(f" xong ({time.time()-t0:.0f}s, best_iter={clf.best_iteration_})")
    return clf


def train_cat(X_tr, y_tr, X_vl, y_vl, sw, cfg, seed):
    print(f"      Training CatBoost ...", end="", flush=True)
    t0 = time.time()
    clf = CatBoostClassifier(
        iterations=700, learning_rate=0.05, depth=cfg["cat_depth"], l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass", random_seed=seed,
        early_stopping_rounds=40, verbose=0, thread_count=-1)
    clf.fit(X_tr, y_tr, sample_weight=sw, eval_set=(X_vl, y_vl))
    print(f" xong ({time.time()-t0:.0f}s, best_iter={clf.best_iteration_})")
    return clf


def run_one_seed(df_full, X_all, y_all, feat_cols, all_classes, cfg,
                 day_name, seed, seed_i, n_seeds):
    print(f"\n  {'─'*62}")
    print(f"  SEED = {seed}   ({seed_i}/{n_seeds})")
    print(f"  {'─'*62}")
    t_seed = time.time()

    le = LabelEncoder(); le.fit(all_classes)
    n_cls = len(all_classes)

    # Split 70/10/20
    tr_idx, vl_idx, te_idx = split_3way(y_all, TEST_FRAC, VAL_FRAC, seed)
    print(f"    Split: train={len(tr_idx):,}  val={len(vl_idx):,}  test={len(te_idx):,}")

    # Balance
    print(f"    Balancing train ...", end="", flush=True)
    t0 = time.time()
    df_tr = df_full.iloc[tr_idx].copy()
    df_tr_bal = balance_df(df_tr, feat_cols, cfg, seed)
    X_tr = df_tr_bal[feat_cols].values.astype(np.float32)
    y_tr = le.transform(df_tr_bal[LABEL_COL].values)
    sw   = compute_weights(y_tr, n_cls, cfg["weight_mode"])
    del df_tr, df_tr_bal; gc.collect()
    print(f" xong ({time.time()-t0:.0f}s, {len(X_tr):,} dòng sau balance)")

    # Val/test from X_all by index 
    X_vl, y_vl = X_all[vl_idx], y_all[vl_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]

    # Train 3 models
    clf_xgb = train_xgb(X_tr, y_tr, X_vl, y_vl, n_cls, sw, cfg, seed)
    clf_lgb = train_lgb(X_tr, y_tr, X_vl, y_vl, n_cls, sw, seed)
    clf_cat = train_cat(X_tr, y_tr, X_vl, y_vl, sw, cfg, seed)
    del X_tr, y_tr, sw; gc.collect()

    # Ensemble weights
    print(f"    Tính ensemble weights + predict test ...", end="", flush=True)
    t0 = time.time()
    pv_x = clf_xgb.predict_proba(X_vl)
    pv_l = clf_lgb.predict_proba(X_vl)
    pv_c = clf_cat.predict_proba(X_vl)
    accs = [accuracy_score(y_vl, p.argmax(1)) for p in (pv_x, pv_l, pv_c)]
    sc = (np.array(accs) - np.mean(accs)) * 50
    w = np.exp(np.clip(sc, -50, 50)); w = w / w.sum()
    del pv_x, pv_l, pv_c

    pt_x = clf_xgb.predict_proba(X_te)
    pt_l = clf_lgb.predict_proba(X_te)
    pt_c = clf_cat.predict_proba(X_te)
    p_ens = w[0]*pt_x + w[1]*pt_l + w[2]*pt_c
    print(f" xong ({time.time()-t0:.0f}s)")

    def _metrics(proba, name):
        yp = proba.argmax(1)
        return dict(model=name, day=day_name, seed=seed,
                    accuracy=accuracy_score(y_te, yp),
                    macro_precision=precision_score(y_te, yp, average="macro", zero_division=0),
                    macro_recall=recall_score(y_te, yp, average="macro", zero_division=0),
                    macro_f1=f1_score(y_te, yp, average="macro", zero_division=0),
                    weighted_precision=precision_score(y_te, yp, average="weighted", zero_division=0),
                    weighted_recall=recall_score(y_te, yp, average="weighted", zero_division=0),
                    weighted_f1=f1_score(y_te, yp, average="weighted", zero_division=0))

    results = [
        _metrics(pt_x, "XGBoost"),
        _metrics(pt_l, "LightGBM"),
        _metrics(pt_c, "CatBoost"),
        _metrics(p_ens, MODEL_NAME),
    ]
    print(f"    Kết quả seed {seed}:")
    for r in results:
        print(f"      {r['model']:<22s}  Acc={r['accuracy']*100:.4f}%  "
              f"W-Prec={r['weighted_precision']:.4f}  "
              f"W-Rec={r['weighted_recall']:.4f}  "
              f"W-F1={r['weighted_f1']:.4f}  MacroF1={r['macro_f1']:.4f}")
    print(f"    >>> Seed {seed} xong trong {(time.time()-t_seed)/60:.1f} phút")

    del X_te, y_te, pt_x, pt_l, pt_c, clf_xgb, clf_lgb, clf_cat
    gc.collect()
    return results


def run_day(day_name):
    print("\n" + "█" * 70)
    print(f"  MULTI-SEED ENSEMBLE — {day_name.upper()}  (seeds={SEEDS})")
    print(f"  Split 70/10/20   |   N_EST={N_EST}")
    print("█" * 70)

    cfg = DAY_CONFIG[day_name]
    train_path = os.path.join(OUTPUT_BASE, day_name, "train.csv")
    test_path  = os.path.join(OUTPUT_BASE, day_name, "test.csv")

    header = pd.read_csv(train_path, nrows=0)
    file_cols = set(header.columns.str.strip())
    base_feats = [f for f in BASE_FEATURES if f in file_cols]
    port_feats = [f for f in PORT_FEATURES if f in file_cols]
    all_load = base_feats + port_feats

    print(f"\n  Load + gộp train.csv + test.csv...")
    df1 = load_csv(train_path, all_load, desc="train")
    df2 = load_csv(test_path,  all_load, desc="test")
    df_full = pd.concat([df1, df2], ignore_index=True)
    del df1, df2; gc.collect()

    df_full = engineer_features(df_full)
    df_full = apply_log(df_full)
    new_eng = [c for c in ENGINEERED_FEATURES if c in df_full.columns]
    feat_cols = base_feats + port_feats + new_eng
    all_classes = sorted(df_full[LABEL_COL].unique().tolist())
    print(f"  Total: {len(df_full):,} rows  |  {len(feat_cols)} features  |  "
          f"{len(all_classes)} classes")

    # Build X_all, y_all
    print(f"  Build X_all (1 lần)...", end="", flush=True)
    le0 = LabelEncoder(); le0.fit(all_classes)
    X_all = df_full[feat_cols].values.astype(np.float32)
    y_all = le0.transform(df_full[LABEL_COL].values)
    print(f" xong ({X_all.shape})")

    rows = []
    for i, seed in enumerate(SEEDS, 1):
        seed_results = run_one_seed(df_full, X_all, y_all, feat_cols,
                                     all_classes, cfg, day_name, seed,
                                     i, len(SEEDS))
        rows.extend(seed_results)
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, f"ensemble_{day_name}.csv"), index=False)
        print(f"  [Đã lưu tạm sau seed {seed}]")

    # Summary
    dfr = pd.DataFrame(rows)
    print(f"\n  {'='*78}")
    print(f"  TỔNG KẾT {day_name.upper()} ({len(SEEDS)} seeds)  —  mean ± std (ddof=1)")
    print(f"  {'='*78}")
    print(f"  {'Model':<22s} {'Accuracy(%)':>15s} {'W-Prec':>15s} "
          f"{'W-Recall':>15s}")
    print(f"  {'-'*70}")
    for model_name in ["XGBoost", "LightGBM", "CatBoost", MODEL_NAME]:
        g = dfr[dfr["model"] == model_name]
        nm = "ENSEMBLE" if model_name == MODEL_NAME else model_name
        acc = f"{g['accuracy'].mean()*100:.3f}±{g['accuracy'].std(ddof=1)*100:.3f}"
        wp  = f"{g['weighted_precision'].mean():.4f}±{g['weighted_precision'].std(ddof=1):.4f}"
        wr  = f"{g['weighted_recall'].mean():.4f}±{g['weighted_recall'].std(ddof=1):.4f}"
        print(f"  {nm:<22s} {acc:>15s} {wp:>15s} {wr:>15s}")
    print(f"  {'-'*70}")
    print(f"  {'Model':<22s} {'W-F1':>15s} {'Macro-F1':>15s}")
    print(f"  {'-'*70}")
    for model_name in ["XGBoost", "LightGBM", "CatBoost", MODEL_NAME]:
        g = dfr[dfr["model"] == model_name]
        nm = "ENSEMBLE" if model_name == MODEL_NAME else model_name
        wf  = f"{g['weighted_f1'].mean():.4f}±{g['weighted_f1'].std(ddof=1):.4f}"
        mf  = f"{g['macro_f1'].mean():.4f}±{g['macro_f1'].std(ddof=1):.4f}"
        print(f"  {nm:<22s} {wf:>15s} {mf:>15s}")
    print(f"\n  ✓ Saved: {RESULTS_DIR}/ensemble_{day_name}.csv")

    del df_full, X_all, y_all; gc.collect()


def main():
    if len(sys.argv) < 2:
        print("Usage: python multiseed_ensemble_multiclass.py {day1|day2|both}")
        sys.exit(1)
    arg = sys.argv[1].lower()
    days = ["day1", "day2"] if arg == "both" else [arg]
    for day in days:
        run_day(day)


if __name__ == "__main__":
    main()