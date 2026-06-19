import os
import gc
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
OUTPUT_BASE  = "./output"
RESULTS_DIR  = "./results_multiseed"
CHUNK_SIZE   = 2_000_000
READ_WORKERS = 4
LABEL_COL    = "Label"

SEEDS = [42, 52, 62, 72, 82]
TEST_FRAC = 0.20
VAL_FRAC  = 0.10   # → train = 0.70

DAY_CONFIG = {
    "day1": dict(max_per_class=300_000, min_per_class=50_000,
                 smote_threshold=1_000, weight_mode="sqrt",
                 xgb_depth=10, xgb_lr=0.025, cat_depth=9,
                 dropout=0.3, weight_decay=1e-5),
    "day2": dict(max_per_class=400_000, min_per_class=50_000,
                 smote_threshold=500, weight_mode="sqrt",
                 xgb_depth=9, xgb_lr=0.03, cat_depth=9,
                 dropout=0.25, weight_decay=1e-5),
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


# ══════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
# BALANCE + SPLIT + WEIGHTS
# ══════════════════════════════════════════════
def smote_lite(sub, target_n, feat_cols, rng):
    n = len(sub); n_gen = target_n - n
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


def compute_weights_arr(y, n_classes, mode):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    max_c = counts.max()
    if mode == "sqrt":
        w = np.sqrt(max_c / (counts + 1e-9))
    elif mode == "balanced":
        w = max_c / (counts + 1e-9)
    else:
        w = np.ones(n_classes)
    return w / w.mean()


def split_3way(y, seed):
    rng = np.random.default_rng(seed)
    tr, vl, te = [], [], []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        rng.shuffle(pos)
        n = len(pos)
        n_test = max(1, int(n * TEST_FRAC))
        n_val  = max(1, int(n * VAL_FRAC))
        te.extend(pos[:n_test])
        vl.extend(pos[n_test:n_test + n_val])
        tr.extend(pos[n_test + n_val:])
    return np.array(tr), np.array(vl), np.array(te)


# ══════════════════════════════════════════════
# LOAD CSV
# ══════════════════════════════════════════════
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


def _load_csv(path, all_feats, desc="Loading"):
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


# ══════════════════════════════════════════════
# MAIN APi
# ══════════════════════════════════════════════
def load_day_data(day_name):

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

    df_full = engineer_features(df_full)
    df_full = apply_log(df_full)
    new_eng = [c for c in ENGINEERED_FEATURES if c in df_full.columns]
    feat_cols = base_feats + port_feats + new_eng
    all_classes = sorted(df_full[LABEL_COL].unique().tolist())
    le = LabelEncoder(); le.fit(all_classes)
    print(f"  Total: {len(df_full):,} rows  {len(feat_cols)} feats  "
          f"{len(all_classes)} classes")
    return df_full, feat_cols, all_classes, le


def make_split(df_full, feat_cols, le, cfg, seed):

    n_cls = len(le.classes_)
    y_all = le.transform(df_full[LABEL_COL].values)
    tr_idx, vl_idx, te_idx = split_3way(y_all, seed)

    df_tr_bal = balance_df(df_full.iloc[tr_idx], feat_cols, cfg, seed)
    Xr_tr = df_tr_bal[feat_cols].values.astype(np.float32)
    y_tr  = le.transform(df_tr_bal[LABEL_COL].values)
    del df_tr_bal

    Xr_vl = df_full.iloc[vl_idx][feat_cols].values.astype(np.float32)
    y_vl  = y_all[vl_idx]
    Xr_te = df_full.iloc[te_idx][feat_cols].values.astype(np.float32)
    y_te  = y_all[te_idx]

    cw_arr = compute_weights_arr(y_tr, n_cls, cfg["weight_mode"])
    sw_sample = cw_arr[y_tr]

    return dict(Xr_tr=Xr_tr, y_tr=y_tr, cw_arr=cw_arr, sw_sample=sw_sample,
                Xr_vl=Xr_vl, y_vl=y_vl, Xr_te=Xr_te, y_te=y_te,
                n_cls=n_cls,
                n_train=len(tr_idx), n_val=len(vl_idx), n_test=len(te_idx))


def scale_split(split):

    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(split["Xr_tr"]).astype(np.float32)
    Xs_vl = scaler.transform(split["Xr_vl"]).astype(np.float32)
    Xs_te = scaler.transform(split["Xr_te"]).astype(np.float32)
    return Xs_tr, Xs_vl, Xs_te


def metrics_dict(model_name, day, seed, y_te, yp):
    return dict(model=model_name, day=day, seed=seed,
                accuracy=accuracy_score(y_te, yp),
                macro_f1=f1_score(y_te, yp, average="macro", zero_division=0),
                weighted_f1=f1_score(y_te, yp, average="weighted", zero_division=0))


def append_result(model_key, day, row):
    path = os.path.join(RESULTS_DIR, f"{model_key}_{day}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["seed"] != row["seed"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("seed").reset_index(drop=True)
    df.to_csv(path, index=False)


def summarize(model_key, day):
    path = os.path.join(RESULTS_DIR, f"{model_key}_{day}.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    print(f"\n  {'='*58}")
    print(f"  SUMMARY {model_key} — {day} ({len(d)} seeds)")
    print(f"  {'='*58}")
    print(f"  Accuracy   : {d['accuracy'].mean()*100:.4f}% ± "
          f"{d['accuracy'].std()*100:.4f}%")
    print(f"  Macro F1   : {d['macro_f1'].mean():.4f} ± {d['macro_f1'].std():.4f}")
    print(f"  Weighted F1: {d['weighted_f1'].mean():.4f} ± "
          f"{d['weighted_f1'].std():.4f}")