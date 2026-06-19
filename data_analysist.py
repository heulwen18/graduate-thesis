import os
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════
TRAIN_PATH   = "./output/train.csv"
ANALYSIS_DIR = "./analysis"
CHUNK_SIZE   = 1_000_000
SAMPLE_SIZE  = 3_000_000
RANDOM_SEED  = 42
LABEL_COL    = "Label"

NZV_THRESH   = 0.01
CORR_THRESH  = 0.95
TOP_N_PLOT   = 20
MI_TOP_N     = 40

# UMAP
UMAP_SAMPLE  = 50_000   # số mẫu dùng cho UMAP (tăng nếu RAM đủ)
UMAP_NEIGHBORS = 30     # n_neighbors: lớn → cấu trúc toàn cục rõ hơn
UMAP_MIN_DIST  = 0.1    # min_dist: nhỏ → cluster chặt hơn

os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# [CÁC BƯỚC 1–9 GIỮ NGUYÊN – không thay đổi]
# ══════════════════════════════════════════════
def load_sample(train_path):
    print("─" * 62)
    print(f"[STEP 1]  Load sample  (target={SAMPLE_SIZE:,} dòng)...")
    t0 = time.time()
    with open(train_path, "rb") as f:
        total_lines = sum(1 for _ in f)
    n_data = total_lines - 1
    sample_ratio = min(1.0, SAMPLE_SIZE / n_data)
    print(f"  Tổng dòng: {n_data:,}  |  Sample ratio: {sample_ratio*100:.1f}%")

    frames = []
    rng = np.random.default_rng(RANDOM_SEED)
    reader = pd.read_csv(train_path, chunksize=CHUNK_SIZE,
                         low_memory=False, on_bad_lines="skip")
    collected = 0
    for chunk in tqdm(reader, desc="  Loading", unit="chunk"):
        chunk.columns = chunk.columns.str.strip()
        if sample_ratio < 1.0:
            mask = rng.random(len(chunk)) < sample_ratio
            chunk = chunk[mask]
        if LABEL_COL in chunk.columns:
            chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()
            chunk = chunk[~chunk[LABEL_COL].isin({"", "nan", "Label"})]
        frames.append(chunk)
        collected += len(chunk)
        if collected >= SAMPLE_SIZE:
            break

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  ✓ Loaded {len(df):,} rows  ({time.time()-t0:.1f}s)")
    vc = df[LABEL_COL].value_counts()
    for lbl, cnt in vc.items():
        print(f"    {lbl:<30s}  {cnt:>8,}  ({cnt/len(df)*100:5.2f}%)")
    return df


def prep_features(df):
    print("\n" + "─" * 62)
    print("[STEP 2]  Prep features...")
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL].astype(str).str.strip())
    print(f"  Features: {len(feat_cols)}  |  Samples: {len(X):,}  |  Classes: {len(le.classes_)}")
    return X, y, feat_cols, le


def analyze_basic_stats(X, y, feat_cols, le):
    print("\n" + "─" * 62)
    print("[STEP 3]  Thống kê cơ bản...")
    stats = X.describe().T
    stats.index = feat_cols
    stats["cv"]         = stats["std"] / (stats["mean"].abs() + 1e-9)
    stats["null_count"] = (X == 0).sum().values
    stats["zero_ratio"] = stats["null_count"] / len(X)
    path = os.path.join(ANALYSIS_DIR, "1_basic_stats.csv")
    stats.to_csv(path)
    print(f"  ✓ {path}")
    return stats


def analyze_variance(X, feat_cols, basic_stats):
    print("\n" + "─" * 62)
    print("[STEP 4]  Variance & Near-Zero Variance analysis...")
    var_df = pd.DataFrame({
        "feature"     : feat_cols,
        "variance"    : X.var().values,
        "std"         : X.std().values,
        "mean"        : X.mean().values,
        "zero_ratio"  : (X == 0).mean().values,
        "unique_ratio": [X[c].nunique() / len(X) for c in feat_cols],
    })
    var_df["is_constant"]  = var_df["variance"] < 1e-10
    var_df["is_near_zero"] = (var_df["std"] / (var_df["mean"].abs() + 1e-9) < NZV_THRESH) & ~var_df["is_constant"]
    var_df["is_high_zero"] = var_df["zero_ratio"] > 0.99
    var_df.sort_values("variance", ascending=False, inplace=True)
    print(f"  Constant: {var_df['is_constant'].sum()}  |  Near-zero: {var_df['is_near_zero'].sum()}  |  High-zero: {var_df['is_high_zero'].sum()}")
    path = os.path.join(ANALYSIS_DIR, "2_variance_report.csv")
    var_df.to_csv(path, index=False)
    print(f"  ✓ {path}")
    return var_df


def analyze_correlation(X, feat_cols):
    print("\n" + "─" * 62)
    print("[STEP 5]  Correlation analysis...")
    t0 = time.time()
    corr  = X.corr(method="pearson")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = []
    for col in upper.columns:
        partners = upper[col][upper[col].abs() > CORR_THRESH]
        for partner, val in partners.items():
            high_corr.append({"feature_a": col, "feature_b": partner, "correlation": val})
    df_corr = pd.DataFrame(high_corr).sort_values("correlation", ascending=False, key=abs)
    print(f"  Cặp correlation > {CORR_THRESH}: {len(df_corr)}")
    top40 = X.var().nlargest(40).index.tolist()
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(X[top40].corr(), ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=False, linewidths=0.3)
    ax.set_title("Correlation Heatmap – Top 40 features by Variance", fontsize=13, pad=12)
    plt.xticks(fontsize=7, rotation=45, ha="right"); plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "3_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.join(ANALYSIS_DIR, '3_correlation_heatmap.png')}  ({time.time()-t0:.1f}s)")
    return df_corr


def analyze_mutual_information(X, y, feat_cols):
    print("\n" + "─" * 62)
    print("[STEP 6]  Mutual Information (MI)...")
    t0 = time.time()
    mi_scores = mutual_info_classif(X.values, y, discrete_features=False,
                                     random_state=RANDOM_SEED, n_neighbors=3)
    mi_df = pd.DataFrame({"feature": feat_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False).reset_index(drop=True)
    mi_df["mi_rank"] = mi_df.index + 1
    top_mi = mi_df.head(MI_TOP_N)
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(len(top_mi)), top_mi["mi_score"].values[::-1],
                   color=plt.cm.plasma(np.linspace(0.1, 0.9, len(top_mi))))
    ax.set_yticks(range(len(top_mi)))
    ax.set_yticklabels(top_mi["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Top {MI_TOP_N} Features – Mutual Information với Label")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "4_mutual_information.png"), dpi=150, bbox_inches="tight")
    plt.close()
    mi_df.to_csv(os.path.join(ANALYSIS_DIR, "4_mutual_information.csv"), index=False)
    print(f"  ✓ {os.path.join(ANALYSIS_DIR, '4_mutual_information.csv')}  ({time.time()-t0:.1f}s)")
    return mi_df


def analyze_anova(X, y, feat_cols):
    print("\n" + "─" * 62)
    print("[STEP 7]  ANOVA F-test...")
    t0 = time.time()
    f_scores, p_values = f_classif(X.values, y)
    anova_df = pd.DataFrame({"feature": feat_cols, "f_score": f_scores, "p_value": p_values})
    anova_df = anova_df.sort_values("f_score", ascending=False).reset_index(drop=True)
    anova_df["anova_rank"]     = anova_df.index + 1
    anova_df["is_significant"] = anova_df["p_value"] < 0.05
    anova_df.to_csv(os.path.join(ANALYSIS_DIR, "5_anova_ftest.csv"), index=False)
    print(f"  ✓ {os.path.join(ANALYSIS_DIR, '5_anova_ftest.csv')}  ({time.time()-t0:.1f}s)")
    return anova_df


def plot_feature_distributions(df_raw, X, y, le, mi_df):
    print("\n" + "─" * 62)
    print(f"[STEP 8]  Vẽ phân phối top {TOP_N_PLOT} features theo class...")
    top_features = mi_df["feature"].head(TOP_N_PLOT).tolist()
    labels = le.inverse_transform(y)
    plot_df = pd.DataFrame(X[top_features].values, columns=top_features)
    plot_df["Label"] = labels
    if len(plot_df) > 50_000:
        plot_df = plot_df.sample(50_000, random_state=RANDOM_SEED)
    ncols = 4
    nrows = (TOP_N_PLOT + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(top_features):
        ax = axes[i]
        classes = sorted(plot_df["Label"].unique())
        data = [plot_df.loc[plot_df["Label"] == cls, feat].values for cls in classes]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="red", linewidth=1.5))
        colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(feat, fontsize=8, fontweight="bold")
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=7); ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Feature Distribution by Class – Top {TOP_N_PLOT} (MI score)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "6_feature_distribution_top20.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.join(ANALYSIS_DIR, '6_feature_distribution_top20.png')}")


def build_final_ranking(feat_cols, var_df, mi_df, anova_df, corr_df):
    print("\n" + "─" * 62)
    print("[STEP 9]  Tổng hợp ranking cuối cùng...")
    ranking = pd.DataFrame({"feature": feat_cols})
    ranking = ranking.merge(mi_df[["feature", "mi_score", "mi_rank"]], on="feature", how="left")
    ranking = ranking.merge(anova_df[["feature", "f_score", "p_value", "anova_rank", "is_significant"]], on="feature", how="left")
    ranking = ranking.merge(var_df[["feature", "variance", "zero_ratio", "is_constant", "is_near_zero", "is_high_zero"]], on="feature", how="left")
    ranking["mi_rank_norm"]    = 1 - (ranking["mi_rank"]    - 1) / len(ranking)
    ranking["anova_rank_norm"] = 1 - (ranking["anova_rank"] - 1) / len(ranking)
    ranking["combined_score"]  = 0.60 * ranking["mi_rank_norm"] + 0.40 * ranking["anova_rank_norm"]
    ranking.sort_values("combined_score", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    ranking["final_rank"] = ranking.index + 1
    redundant_set = set()
    if len(corr_df) > 0:
        for _, row in corr_df.iterrows():
            a, b = row["feature_a"], row["feature_b"]
            score_a = ranking.loc[ranking["feature"] == a, "combined_score"]
            score_b = ranking.loc[ranking["feature"] == b, "combined_score"]
            if len(score_a) > 0 and len(score_b) > 0:
                if score_a.values[0] >= score_b.values[0]:
                    redundant_set.add(b)
                else:
                    redundant_set.add(a)
    ranking["is_redundant"]   = ranking["feature"].isin(redundant_set)
    ranking["recommend_drop"] = ranking["is_constant"] | ranking["is_near_zero"] | ranking["is_high_zero"] | ranking["is_redundant"]
    ranking["recommend_keep"] = ~ranking["recommend_drop"]
    ranking.to_csv(os.path.join(ANALYSIS_DIR, "7_feature_ranking_final.csv"), index=False)
    print(f"  ✓ {os.path.join(ANALYSIS_DIR, '7_feature_ranking_final.csv')}")
    return ranking


def write_summary(ranking, var_df, corr_df, mi_df):
    print("\n" + "─" * 62)
    print("[STEP 10]  Viết summary report...")
    keep_feats = ranking[ranking["recommend_keep"]]["feature"].tolist()
    drop_feats = ranking[ranking["recommend_drop"]]["feature"].tolist()
    lines = ["=" * 65, "  CIC-DDoS2019 – FEATURE ANALYSIS SUMMARY", "=" * 65,
             f"\n  Tổng features phân tích : {len(ranking)}",
             f"  Features nên GIỮ        : {len(keep_feats)}",
             f"  Features nên LOẠI       : {len(drop_feats)}"]
    lines.append("\n  SELECTED_FEATURES = [")
    for feat in keep_feats:
        lines.append(f'      "{feat}",')
    lines.append("  ]")
    summary = "\n".join(lines)
    path = os.path.join(ANALYSIS_DIR, "8_summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════
# BƯỚC 11: UMAP
# ══════════════════════════════════════════════
def analyze_umap(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder,
                  ranking: pd.DataFrame):
    """
    Giảm chiều bằng UMAP và vẽ 4 loại plot:
      9_umap_all_classes.png     – toàn bộ nhãn, mỗi lớp 1 màu
      9_umap_benign_vs_attack.png– BENIGN nổi bật trên nền attack
      9_umap_per_class.png       – grid subplot từng lớp / nền xám
      9_umap_3d.png              – UMAP 3 chiều (bonus)
    """
    try:
        import umap as umap_lib
    except ImportError:
        print("  ✗ Chưa cài umap-learn. Chạy: pip install umap-learn")
        return

    print("\n" + "─" * 62)
    print(f"[STEP 11]  UMAP  (sample={UMAP_SAMPLE:,}, "
          f"n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
    t0 = time.time()

    # ── 1. Chọn features & lấy mẫu stratified ────────────────────
    keep_feats = ranking[ranking["recommend_keep"]]["feature"].tolist()
    available  = [f for f in keep_feats if f in X.columns]

    X_sub  = X[available].copy()
    labels = le.inverse_transform(y)

    rng = np.random.default_rng(RANDOM_SEED)
    labels_unique = sorted(set(labels))
    n_cls         = len(labels_unique)
    n_per_class   = max(300, UMAP_SAMPLE // n_cls)

    idx_list = []
    for cls in labels_unique:
        idx_cls = np.where(labels == cls)[0]
        chosen  = rng.choice(idx_cls, size=min(len(idx_cls), n_per_class), replace=False)
        idx_list.append(chosen)
    idx_all = np.concatenate(idx_list)

    X_plot = X_sub.iloc[idx_all].values
    y_plot = labels[idx_all]

    print(f"  Features dùng   : {len(available)}")
    print(f"  Mẫu UMAP        : {len(X_plot):,}  ({n_cls} lớp, tối đa {n_per_class}/lớp)")

    # ── 2. Tiền xử lý ────────────────────────────────────────────
    from sklearn.preprocessing import RobustScaler
    X_plot = np.where(np.isfinite(X_plot), X_plot, 0).astype(np.float32)
    X_scaled = RobustScaler().fit_transform(X_plot)
    X_scaled = np.clip(X_scaled, -10, 10)

    # ── 3. Fit UMAP 2D ───────────────────────────────────────────
    print("  Fit UMAP 2D...", flush=True)
    reducer_2d = umap_lib.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_SEED,
        low_memory=True,
        verbose=False,
    )
    emb2d = reducer_2d.fit_transform(X_scaled)
    print(f"  ✓ UMAP 2D done  ({time.time()-t0:.1f}s)")

    # ── 4. Palette ───────────────────────────────────────────────
    PAL = [
        "#534AB7", "#1D9E75", "#D85A30", "#3266ad", "#BA7517",
        "#993556", "#639922", "#5DCAA5", "#185FA5", "#A32D2D",
        "#854F0B", "#0F6E56", "#D4537E", "#4A1B0C", "#27500A",
        "#E24B4A", "#F0997B", "#888780", "#2ecc71",
    ]
    label2c = {lbl: PAL[i % len(PAL)] for i, lbl in enumerate(labels_unique)}
    if "BENIGN" in label2c:
        label2c["BENIGN"] = "#E24B4A"   # BENIGN luôn đỏ để dễ nhận
    colors_all = [label2c[l] for l in y_plot]

    # ── 5. Plot A: toàn bộ nhãn ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(emb2d[:, 0], emb2d[:, 1],
               c=colors_all, s=8, alpha=0.55, linewidths=0)
    handles = [mpatches.Patch(color=label2c[l], label=l) for l in labels_unique]
    ax.legend(handles=handles, fontsize=7.5, ncol=2,
              bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(
        f"UMAP – CIC DDoS 2019  ({len(emb2d):,} mẫu, {len(available)} features)",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_aspect("equal")
    plt.tight_layout()
    p = os.path.join(ANALYSIS_DIR, "9_umap_all_classes.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {p}")

    # ── 6. Plot B: BENIGN vs Attack ───────────────────────────────
    is_benign = (y_plot == "BENIGN")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(emb2d[~is_benign, 0], emb2d[~is_benign, 1],
               c="#B5D4F4", s=7, alpha=0.4, linewidths=0,
               label=f"Attack  (n={(~is_benign).sum():,})")
    ax.scatter(emb2d[is_benign, 0], emb2d[is_benign, 1],
               c="#E24B4A", s=18, alpha=0.85, linewidths=0,
               label=f"BENIGN  (n={is_benign.sum():,})")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("UMAP – BENIGN vs Attack traffic", fontsize=12, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_aspect("equal")
    plt.tight_layout()
    p = os.path.join(ANALYSIS_DIR, "9_umap_benign_vs_attack.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {p}")

    # ── 7. Plot C: grid subplot mỗi lớp ──────────────────────────
    ncols = 4
    nrows = int(np.ceil(n_cls / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3))
    fig.suptitle("UMAP – phân bổ từng lớp", fontsize=13, fontweight="bold", y=1.01)
    for i, (lbl, ax) in enumerate(zip(labels_unique, axes.flat)):
        mask = (y_plot == lbl)
        ax.scatter(emb2d[~mask, 0], emb2d[~mask, 1],
                   c="#e0e0e0", s=3, alpha=0.3, linewidths=0)
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=label2c[lbl], s=7, alpha=0.75, linewidths=0)
        ax.set_title(f"{lbl}\n(n={mask.sum():,})", fontsize=8, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect("equal")
    for ax in axes.flat[n_cls:]:
        ax.set_visible(False)
    plt.tight_layout()
    p = os.path.join(ANALYSIS_DIR, "9_umap_per_class.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {p}")

    # ── 8. Plot D: UMAP 3D ────────────────────────────────────────
    print("  Fit UMAP 3D...", flush=True)
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    reducer_3d = umap_lib.UMAP(
        n_components=3,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_SEED,
        low_memory=True,
        verbose=False,
    )
    emb3d = reducer_3d.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(emb3d[:, 0], emb3d[:, 1], emb3d[:, 2],
                c=colors_all, s=5, alpha=0.5, linewidths=0)
    handles3 = [mpatches.Patch(color=label2c[l], label=l) for l in labels_unique]
    ax3.legend(handles=handles3, fontsize=7, ncol=2,
               bbox_to_anchor=(1.15, 1), loc="upper left")
    ax3.set_xlabel("UMAP-1"); ax3.set_ylabel("UMAP-2"); ax3.set_zlabel("UMAP-3")
    ax3.set_title("UMAP 3D – CIC DDoS 2019", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(ANALYSIS_DIR, "9_umap_3d.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  ✓ {p}")

    print(f"\n  [STEP 11 DONE]  UMAP tổng thời gian: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 62)
    print("  FEATURE ANALYSIS  –  CIC-DDoS2019")
    print("=" * 62)

    df_raw   = load_sample(TRAIN_PATH)
    X, y, feat_cols, le = prep_features(df_raw)
    basic_stats = analyze_basic_stats(X, y, feat_cols, le)
    var_df      = analyze_variance(X, feat_cols, basic_stats)
    corr_df     = analyze_correlation(X, feat_cols)
    mi_df       = analyze_mutual_information(X, y, feat_cols)
    anova_df    = analyze_anova(X, y, feat_cols)
    plot_feature_distributions(df_raw, X, y, le, mi_df)
    ranking     = build_final_ranking(feat_cols, var_df, mi_df, anova_df, corr_df)
    write_summary(ranking, var_df, corr_df, mi_df)

    # ── STEP 11: UMAP ──────────────────────────────────────────
    analyze_umap(X, y, le, ranking)

    print("\n" + "=" * 62)
    print(f"  [DONE]  Tổng thời gian: {(time.time()-t_start)/60:.1f} phút")
    print(f"  Output: {ANALYSIS_DIR}/")
    print("=" * 62)


if __name__ == "__main__":
    main()