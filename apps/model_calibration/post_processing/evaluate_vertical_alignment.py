import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from calibration_model_full_compare import (
    fit_similarity, apply_similarity,
    fit_rbf_residual, apply_rbf_to_points,
    ThinPlateSpline2D, PiecewiseAffine,
    fit_gpr_residual, apply_gpr,
    eval_stats
)

# ============================================================
#  把 vertical 数据每 n 个样本合并成一个平均 gaze + 平均 dot
# ============================================================
def average_every_n(df, n=4):
    """
    df: 包含 gaze_x, gaze_y, dot_x, dot_y, timestamp 的数据
    n:  每 n 条数据合并成 1 条（默认 4）
    """

    df = df.copy().reset_index(drop=True)

    groups = []
    total = len(df)

    for i in range(0, total, n):
        block = df.iloc[i:i+n]
        if len(block) < n:
            continue  # 不够 4 条就丢掉

        avg_gaze_x = block["gaze_x"].mean()
        avg_gaze_y = block["gaze_y"].mean()
        avg_dot_x  = block["dot_x"].mean()
        avg_dot_y  = block["dot_y"].mean()

        # line_idx 取这 4 个中出现最多的（通常是相同的）
        line_idx = block["line_idx"].mode()[0]

        # timestamp 取中间值
        timestamp = block["timestamp"].median() if "timestamp" in block.columns else i

        groups.append({
            "gaze_x": avg_gaze_x,
            "gaze_y": avg_gaze_y,
            "dot_x": avg_dot_x,
            "dot_y": avg_dot_y,
            "line_idx": line_idx,
            "timestamp": timestamp,
        })

    return pd.DataFrame(groups)

# ============================================================
#  主函数：输入 session 路径，自动找到 origin + vertical
# ============================================================
def evaluate_vertical_session(
    session_dir,
    model_type="sim+rbf",
    dist_threshold=500,
    edge_margin=100,
    rbf_kernel="multiquadric",
    rbf_smooth=1.0
):
    session_dir = Path(session_dir)
    origin_csv = session_dir / "origin" / "grid_gaze_log.csv"
    align_csv = session_dir / "vertical" / "reading_alignment.csv"
    print("\n=== Vertical Evaluation ===")
    print("Session path:", session_dir)

    # ---------------------------
    # 检查文件是否存在
    # ---------------------------
    if not origin_csv.exists():
        raise FileNotFoundError(f"origin file not found: {origin_csv}")
    if not align_csv.exists():
        raise FileNotFoundError(f"alignment file not found: {align_csv}")

    print("Using origin:   ", origin_csv)
    print("Using alignment:", align_csv)

    # ---------------------------------------------------------
    # 1. 加载 origin 数据 → 训练模型
    # ---------------------------------------------------------
    odf = pd.read_csv(origin_csv)
    origin_obs = odf[['original_gaze_x','original_gaze_y']].values
    origin_tgt = odf[['target_x','target_y']].values

    # 1.1 similarity
    s, R, t = fit_similarity(origin_obs, origin_tgt)
    origin_sim = apply_similarity(origin_obs, s, R, t)

    # 1.2 residual model
    rbf_x = rbf_y = None
    tps_obj = None
    pwa_obj = None
    gpr_x = gpr_y = None

    if model_type == "sim+rbf":
        rbf_x, rbf_y = fit_rbf_residual(
            origin_obs, origin_sim, origin_tgt,
            kernel=rbf_kernel, smooth=rbf_smooth
        )
    elif model_type == "sim+tps":
        tps_obj = ThinPlateSpline2D(origin_sim, origin_tgt)
    elif model_type == "sim+pwa":
        pwa_obj = PiecewiseAffine(origin_sim, origin_tgt)
    elif model_type == "sim+gpr":
        gpr_x, gpr_y = fit_gpr_residual(origin_obs, origin_sim, origin_tgt)
    print("Base model trained.\n")

    # ---------------------------------------------------------
    # 2. 加载 alignment（vertical）数据
    # ---------------------------------------------------------
    df = pd.read_csv(align_csv)
    required_cols = {"gaze_x","gaze_y","dot_x","dot_y","line_idx"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"alignment csv must contain columns: {required_cols}")
    # 按时间排序（如果有 timestamp）
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # ---------------------------------------------------------
    # 3. 过滤逻辑
    # ---------------------------------------------------------

    # (1) dot–gaze 距离过滤
    dist = np.sqrt(
        (df["dot_x"] - df["gaze_x"])**2 +
        (df["dot_y"] - df["gaze_y"])**2
    )
    df = df[dist <= dist_threshold].copy()

    # (2) 每行去掉 dot_x 两端 edge_margin px
    kept = []
    for line_id, sub in df.groupby("line_idx"):
        min_x = sub["dot_x"].min()
        max_x = sub["dot_x"].max()

        sub2 = sub[
            (sub["dot_x"] >= min_x + edge_margin) &
            (sub["dot_x"] <= max_x - edge_margin)
        ]
        kept.append(sub2)

    df = pd.concat(kept, ignore_index=True)
    print("Filtered samples:", len(df))

    if len(df) < 10:
        print("Too few samples after filtering.")
        return None

    # 提取观察点和真实点
    obs = df[["gaze_x","gaze_y"]].values
    tgt = df[["dot_x","dot_y"]].values

    # ---------------------------------------------------------
    # 4. 应用模型
    # ---------------------------------------------------------
    pred = apply_similarity(obs, s, R, t)

    if model_type == "sim+rbf":
        pred = apply_rbf_to_points(obs, pred, rbf_x, rbf_y)
    elif model_type == "sim+tps":
        pred = tps_obj.transform(pred)
    elif model_type == "sim+pwa":
        pred = pwa_obj.transform(pred)
    elif model_type == "sim+gpr":
        pred, _ = apply_gpr(obs, pred, gpr_x, gpr_y)

    # ---------------------------------------------------------
    # 5. 输出整体误差
    # ---------------------------------------------------------
    stats_all = eval_stats(pred, tgt)

    print("\n=== Overall Error ===")
    print(f"Mean   = {stats_all['mean']:.2f} px")
    print(f"Median = {stats_all['median']:.2f} px")
    print(f"P95    = {stats_all['p95']:.2f} px")

    # ---------------------------------------------------------
    # 6. 按行误差
    # ---------------------------------------------------------
    print("\n=== Error Per Line ===")
    per_line = {}

    for line_id, sub in df.groupby("line_idx"):
        idx = sub.index.to_numpy()
        pred_l = pred[idx]
        tgt_l = tgt[idx]

        st = eval_stats(pred_l, tgt_l)
        per_line[int(line_id)] = st

        print(f"Line {line_id}:  mean={st['mean']:.2f}, median={st['median']:.2f}, p95={st['p95']:.2f}")

    # 返回结果
    return {
        "overall": stats_all,
        "per_line": per_line,
        "pred": pred,
        "tgt": tgt
    }

def fit_poly2_model(origin_obs, origin_tgt, degree=1, alpha=0):

    # build features
    X = origin_obs
    Y = origin_tgt

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)

    model_x = Ridge(alpha=alpha).fit(Xp, Y[:,0])
    model_y = Ridge(alpha=alpha).fit(Xp, Y[:,1])

    return {
        "poly": poly,
        "model_x": model_x,
        "model_y": model_y
    }


def apply_poly2_model(obs, model):
    poly = model["poly"]
    Xp = poly.transform(obs)
    px = model["model_x"].predict(Xp)
    py = model["model_y"].predict(Xp)
    return np.column_stack([px, py])


# =========================================================
# 2. 单个 session：运行所有模型，并返回误差统计
# =========================================================
def evaluate_single_session(session_dir):

    session_dir = Path(session_dir)
    origin_csv = session_dir / "origin" / "grid_gaze_log.csv"
    align_csv  = session_dir / "vertical" / "reading_alignment.csv"

    if not origin_csv.exists() or not align_csv.exists():
        print(f"[Skip] Missing data in {session_dir}")
        return None

    # --------- load data ---------
    odf = pd.read_csv(origin_csv)
    vdf = pd.read_csv(align_csv)

    vdf = average_every_n(vdf, n=4)

    required = {"gaze_x","gaze_y","dot_x","dot_y","line_idx"}
    if not required.issubset(vdf.columns):
        print("Invalid vertical data format:", align_csv)
        return None

    # --------- filtering (same as before) ---------
    dist = np.sqrt((vdf["dot_x"]-vdf["gaze_x"])**2 + (vdf["dot_y"]-vdf["gaze_y"])**2)
    vdf = vdf[dist < 250].copy()

    kept = []
    for lid, sub in vdf.groupby("line_idx"):
        mn = sub["dot_x"].min()
        mx = sub["dot_x"].max()
        kept.append(sub[(sub["dot_x"]>=mn+100) & (sub["dot_x"]<=mx-100)])
    vdf = pd.concat(kept, ignore_index=True)

    # features
    origin_obs = odf[['original_gaze_x','original_gaze_y']].values
    origin_tgt = odf[['target_x','target_y']].values
    obs = vdf[['gaze_x','gaze_y']].values
    tgt = vdf[['dot_x','dot_y']].values

    # --------- Similarity ---------
    s, R, t = fit_similarity(origin_obs, origin_tgt)
    sim_pred = apply_similarity(obs, s, R, t)

    # --------- Polynomial2 ---------
    poly_model = fit_poly2_model(origin_obs, origin_tgt, degree=2, alpha=0)
    poly_pred = apply_poly2_model(obs, poly_model)

    # --------- RBF ---------
    origin_sim = apply_similarity(origin_obs, s, R, t)
    rbf_x, rbf_y = fit_rbf_residual(origin_obs, origin_sim, origin_tgt,
                                    kernel="multiquadric", smooth=1.0)
    rbf_pred = apply_rbf_to_points(obs, sim_pred, rbf_x, rbf_y)

    # --------- TPS ---------
    tps_obj = ThinPlateSpline2D(origin_sim, origin_tgt)
    tps_pred = tps_obj.transform(sim_pred)

    # --------- PWA ---------
    pwa_obj = PiecewiseAffine(origin_sim, origin_tgt)
    pwa_pred = pwa_obj.transform(sim_pred)

    # --------- GPR ---------
    gpr_x, gpr_y = fit_gpr_residual(origin_obs, origin_sim, origin_tgt)
    gpr_pred, _ = apply_gpr(obs, sim_pred, gpr_x, gpr_y)

    # --------- return evaluation ---------
    results = {
        "similarity": eval_stats(sim_pred, tgt),
        "poly2":      eval_stats(poly_pred, tgt),
        "sim+rbf":    eval_stats(rbf_pred, tgt),
        "sim+tps":    eval_stats(tps_pred, tgt),
        "sim+pwa":    eval_stats(pwa_pred, tgt),
        "sim+gpr":    eval_stats(gpr_pred, tgt),
    }

    return results



# =========================================================
# 3. 打印单个 session 的对比表
# =========================================================
def compare_models_for_session(session_dir):

    stats = evaluate_single_session(session_dir)
    if stats is None:
        return

    print(f"\n=== Model Comparison for {session_dir} ===")
    print("{:<12} {:>8} {:>8} {:>8}".format("Model","Mean","Median","P95"))
    print("-"*40)

    for m, st in stats.items():
        print(f"{m:<12} {st['mean']:8.2f} {st['median']:8.2f} {st['p95']:8.2f}")

    return stats



# =========================================================
# 4. 扫描所有 session 并输出全部模型对比
# =========================================================
def scan_and_evaluate_all_sessions(root_dir):

    root_dir = Path(root_dir)
    all_results = {}

    # 遍历两层目录
    for person in root_dir.glob("*"):
        if not person.is_dir():
            continue

        for session in person.glob("*"):
            if not session.is_dir():
                continue

            if (session/"origin").exists() and (session/"vertical").exists():
                print("\n------------------------------------------------------")
                print("Evaluating session:", session)

                stats = evaluate_single_session(session)
                if stats is not None:
                    all_results[str(session)] = stats

    return all_results

# ============================================================
#  计算所有 session 的整体平均
# ============================================================
def aggregate_results(all_results):

    if not all_results:
        print("No valid sessions found.")
        return None

    # models = keys from any session
    model_list = list(next(iter(all_results.values())).keys())

    df_list = []

    for session, stats in all_results.items():
        for model, st in stats.items():
            df_list.append({
                "session": session,
                "model": model,
                "mean": st["mean"],
                "median": st["median"],
                "p95": st["p95"],
            })

    df = pd.DataFrame(df_list)

    # 聚合平均
    summary = df.groupby("model")[["mean","median","p95"]].mean().sort_values("mean")
    print("len of summary:", len(all_results))

    print("\n\n======= Overall Summary Across All Sessions =======")
    print(summary)

    return summary


def find_best_rbf_improvements(all_results, top_k=5):
    """
    输入 scan_and_evaluate_all_sessions 的结果，
    输出 sim+rbf 相比 poly2 提升最大的 top_k 个 session
    """

    improvements = []

    for session, stats in all_results.items():

        if "poly2" not in stats or "sim+rbf" not in stats:
            continue

        poly_mean = stats["poly2"]["mean"]
        rbf_mean = stats["sim+rbf"]["mean"]
        improvement = poly_mean - rbf_mean

        improvements.append({
            "session": session,
            "poly_mean": poly_mean,
            "rbf_mean": rbf_mean,
            "improvement": improvement,
        })

    df = pd.DataFrame(improvements)
    df = df.sort_values("improvement", ascending=False)

    return df.head(top_k)

def print_model_comparison_for_sessions(all_results, session_list):
    """
    给定若干 session 名（列表），打印所有模型表现。
    """

    for session in session_list:
        print("\n===============================================")
        print("Session:", session)

        stats = all_results.get(session)
        if stats is None:
            print("(no data)")
            continue

        print("{:<12} {:>8} {:>8} {:>8}".format("Model","Mean","Median","P95"))
        print("-"*40)
        for model, st in stats.items():
            print(f"{model:<12} {st['mean']:8.2f} {st['median']:8.2f} {st['p95']:8.2f}")

# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    session = r"C:\\Users\\SCCN\\Desktop\\systematic_recalibration"
    all_results = scan_and_evaluate_all_sessions(session)
    # aggregate_results(results)
    # 找出 RBF 提升最大的 10 个 session
    best_df = find_best_rbf_improvements(all_results, top_k=15)

    # 输出这些 session 的各模型对比
    best_sessions = best_df["session"].tolist()
    print_model_comparison_for_sessions(all_results, best_sessions)
    print(best_df["session"])