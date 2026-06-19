import sys
import time
import gc
import warnings

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from multiseed_common_binary import (
    SEEDS, DAY_CONFIG_BIN,
    load_day_data_bin, make_split_bin,
    binary_metrics, append_result_bin, summarize_bin,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_BATCH = 1024
CNN_EPOCHS = 25
CNN_LR = 1e-3
CNN_PATIENCE = 4
NUM_WORKERS = 4


WEIGHT_TEMPERATURE = 0.01   
WEIGHT_FLOOR = 0.05        
WEIGHT_CAP   = 0.60         

# Per-model threshold + ensemble threshold t(max macro-F1)
TUNE_THRESHOLDS = True      # False = 0.5


# ── CNN binary  ──
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


def predict_proba_cnn(model, X, batch_size=4096):
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(DEVICE)
            out[i:i+batch_size] = torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()
    return out


def train_cnn(Xs_tr, y_tr, Xs_vl, y_vl, cfg, n_features, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32).to(DEVICE)
    tr_ds = TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(y_tr))
    vl_ds = TensorDataset(torch.from_numpy(Xs_vl), torch.from_numpy(y_vl))
    tr_ld = DataLoader(tr_ds, batch_size=CNN_BATCH, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                       drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=CNN_BATCH*2, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    model = DDoSBinaryConv1D(n_features, cfg["dropout"]).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EPOCHS, eta_min=1e-6)

    best_acc = 0; best_state = None; no_imp = 0
    for ep in range(CNN_EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).float()
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
        sched.step()
        model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in vl_ld:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).float()
                pred = (torch.sigmoid(model(xb).squeeze(-1)) > 0.5).long()
                correct += pred.eq(yb.long()).sum().item(); total += yb.size(0)
        va = correct / total
        if va > best_acc:
            best_acc = va; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= CNN_PATIENCE:
                break
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    del tr_ds, vl_ds, tr_ld, vl_ld; gc.collect()
    return model


def find_best_threshold(proba, y, grid=None):
    from sklearn.metrics import f1_score as _f1
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 0.90, 50),
                                np.linspace(0.90, 0.999, 80)])
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        yp = (proba > thr).astype(np.int8)
        f1m = _f1(y, yp, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1, best_thr = f1m, thr
    return best_thr, best_f1


def compute_smoothed_weights(f1_list):
    f1_arr = np.array(f1_list)
    z = (f1_arr - f1_arr.max()) / WEIGHT_TEMPERATURE
    z = np.clip(z, -50, 0)
    w = np.exp(z); w = w / w.sum()
    for _ in range(10):   # áp floor + cap then renormalize until stable
        w = np.clip(w, WEIGHT_FLOOR, WEIGHT_CAP)
        w = w / w.sum()
    return w


def run_seed(split, cfg, day, seed):
    Xs_tr, y_tr = split["Xs_tr"], split["y_tr"]
    Xs_vl, y_vl = split["Xs_vl"], split["y_vl"]
    Xs_te, y_te = split["Xs_te"], split["y_te"]
    spw = split["scale_pos_weight"]
    t0 = time.time()

    # XGB
    clf_x = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=8, max_leaves=255,
        objective="binary:logistic", eval_metric="logloss",
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        reg_alpha=0.1, reg_lambda=1.5, tree_method="hist", grow_policy="lossguide",
        max_bin=256, scale_pos_weight=spw, device="cpu", n_jobs=-1,
        random_state=seed, verbosity=0, early_stopping_rounds=50)
    clf_x.fit(Xs_tr, y_tr, eval_set=[(Xs_vl, y_vl)], verbose=False)

    # LGB
    clf_l = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=255, max_depth=10,
        objective="binary", metric="binary_logloss",
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.5, max_bin=256, scale_pos_weight=spw,
        n_jobs=-1, random_state=seed, verbosity=-1, device="cpu")
    clf_l.fit(Xs_tr, y_tr, eval_set=[(Xs_vl, y_vl)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    # CAT
    clf_c = CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        loss_function="Logloss", eval_metric="Logloss", scale_pos_weight=spw,
        random_seed=seed, early_stopping_rounds=50, verbose=0, thread_count=-1)
    clf_c.fit(Xs_tr, y_tr, eval_set=(Xs_vl, y_vl))

    # CNN
    cnn = train_cnn(Xs_tr, y_tr, Xs_vl, y_vl, cfg, split["n_features"], seed)

    # ── Predict on VAL (real distribution) to tune weights + thresholds ──
    pv_x = clf_x.predict_proba(Xs_vl)[:, 1]
    pv_l = clf_l.predict_proba(Xs_vl)[:, 1]
    pv_c = clf_c.predict_proba(Xs_vl)[:, 1]
    pv_n = predict_proba_cnn(cnn, Xs_vl)
    val_probas = {"xgb": pv_x, "lgb": pv_l, "cat": pv_c, "cnn": pv_n}

    # Per-model threshold + macro-F1 on VAL
    thr = {}; f1_val = {}
    for k, p in val_probas.items():
        if TUNE_THRESHOLDS:
            t, f1m = find_best_threshold(p, y_vl)
        else:
            from sklearn.metrics import f1_score as _f1
            t = 0.5; f1m = _f1(y_vl, (p > 0.5).astype(int), average="macro", zero_division=0)
        thr[k] = t; f1_val[k] = f1m

    # Weights = softmax(macroF1/T) + floor + cap 
    w = compute_smoothed_weights([f1_val["xgb"], f1_val["lgb"],
                                   f1_val["cat"], f1_val["cnn"]])

    # ── Predict test ──
    pt_x = clf_x.predict_proba(Xs_te)[:, 1]
    pt_l = clf_l.predict_proba(Xs_te)[:, 1]
    pt_c = clf_c.predict_proba(Xs_te)[:, 1]
    pt_n = predict_proba_cnn(cnn, Xs_te)
    pt_ens = w[0]*pt_x + w[1]*pt_l + w[2]*pt_c + w[3]*pt_n

    # Ensemble threshold tune on VAL
    pv_ens = w[0]*pv_x + w[1]*pv_l + w[2]*pv_c + w[3]*pv_n
    if TUNE_THRESHOLDS:
        thr_ens, _ = find_best_threshold(pv_ens, y_vl)
    else:
        thr_ens = 0.5

    model_specs = [
        ("XGBoost", "xgb", pt_x, thr["xgb"]),
        ("LightGBM", "lgb", pt_l, thr["lgb"]),
        ("CatBoost", "cat", pt_c, thr["cat"]),
        ("CNN", "cnn", pt_n, thr["cnn"]),
        ("Ensemble", "ensemble", pt_ens, thr_ens),
    ]
    rows = []
    for name, key, proba, t in model_specs:
        row = binary_metrics(name, day, seed, y_te, proba, t)
        row["w_xgb"], row["w_lgb"] = w[0], w[1]
        row["w_cat"], row["w_cnn"] = w[2], w[3]
        append_result_bin(key, day, row)
        rows.append((name, row))

    # print
    print(f"    weights: XGB={w[0]:.3f} LGB={w[1]:.3f} CAT={w[2]:.3f} CNN={w[3]:.3f}")
    print(f"    thr: XGB={thr['xgb']:.3f} LGB={thr['lgb']:.3f} "
          f"CAT={thr['cat']:.3f} CNN={thr['cnn']:.3f} ENS={thr_ens:.3f}")
    for name, r in rows:
        marker = "  ←" if name == "Ensemble" else ""
        print(f"    {name:<10s} acc={r['accuracy']*100:.4f}%  "
              f"FPR={r['fpr']*100:.3f}%  FNR={r['fnr']*100:.4f}%  "
              f"AUC={r['auc']:.4f}{marker}")
    print(f"    ({time.time()-t0:.0f}s)")

    del clf_x, clf_l, clf_c, cnn; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_day(day):
    print("\n" + "█" * 70)
    print(f"  MULTI-SEED BINARY ENSEMBLE — {day.upper()}  seeds={SEEDS}")
    print(f"  tune_thresholds={TUNE_THRESHOLDS}  device={DEVICE}")
    print("█" * 70)
    cfg = DAY_CONFIG_BIN[day]
    df_full, feat_cols = load_day_data_bin(day)
    for seed in SEEDS:
        print(f"\n  ── SEED {seed} ──")
        split = make_split_bin(df_full, feat_cols, cfg, seed)
        print(f"    Split: train={split['n_train']:,}  "
              f"val={split['n_val']:,}  test={split['n_test']:,}  "
              f"spw={split['scale_pos_weight']:.2f}")
        run_seed(split, cfg, day, seed)
        del split; gc.collect()

    # summary per model
    for key in ["xgb", "lgb", "cat", "cnn", "ensemble"]:
        summarize_bin(key, day)


def main():
    if len(sys.argv) < 2:
        print("Usage: python multiseed_binary_ensemble.py {day1|day2|both}")
        sys.exit(1)
    arg = sys.argv[1].lower()
    days = ["day1", "day2"] if arg == "both" else [arg]
    for day in days:
        run_day(day)


if __name__ == "__main__":
    main()