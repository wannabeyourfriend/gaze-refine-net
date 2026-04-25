"""Generate publication-quality figures for the NeurIPS 2026 resubmission.

All output PDFs land in `research/figures/` with the same base names as
referenced from the LaTeX source.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT.parent.parent / "data" / "prepared"
COMBINED = ROOT / "data" / "all_trials_combined.csv"

# ----- style ----------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C = {
    "raw":        "#777777",
    "classical":  "#264653",
    "neural":     "#7A3293",
    "leaky":      "#D1495B",
    "ours":       "#2E933C",
    "anchor":     "#3454D1",
    "highlight":  "#F4A261",
}

ANCHOR = "pred_sim_rbf_multiquadric_s2.0"


# ============================================================================
# Figure 2 - leakage diagnosis bar chart
# ============================================================================

def fig02_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)

    # JuDo
    ax = axes[0]
    labels = ["raw\n(origin)", "best classical\n(similarity)", "honest paper\nmethod", "leaky paper\nmethod"]
    vals   = [21.96, 21.66, 30.46, 4.21]
    colors = [C["raw"], C["classical"], C["neural"], C["leaky"]]
    hatches = ["", "", "", "////"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6, hatch=None)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    ax.axhline(21.66, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0, 22.5, "best classical", fontsize=7, color="black", alpha=0.7)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylabel("test mean L2 (px)")
    ax.set_title("(a) JuDo1000")
    ax.set_ylim(0, 36)
    ax.annotate("matches published 5.82",
                xy=(3, 4.21), xytext=(2.0, 18),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                fontsize=7.5, ha="center", color=C["leaky"])

    # Self-collected
    ax = axes[1]
    labels = ["raw\n(origin)", "best classical\n(sim+RBF)", "honest paper\nmethod", "leaky paper\nmethod\n(train L2)"]
    vals   = [67.25, 45.53, 45.06, 12.0]
    colors = [C["raw"], C["classical"], C["neural"], C["leaky"]]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    bars[3].set_hatch("////")
    ax.axhline(45.53, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0, 47, "best classical", fontsize=7, color="black", alpha=0.7)
    for x, v in enumerate(vals):
        ax.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylabel("test mean L2 (px)")
    ax.set_title("(b) Self-collected (12 subj.)")
    ax.set_ylim(0, 80)
    ax.annotate("inflated training signal\n(test L2 ≈ 42 px)",
                xy=(3, 12), xytext=(2.0, 35),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                fontsize=7.5, ha="center", color=C["leaky"])

    plt.tight_layout()
    fig.savefig(OUT / "fig02_leakage.pdf")
    plt.close(fig)
    print("wrote fig02_leakage.pdf")


# ============================================================================
# Figure 3 - dataset visualization (4 panels)
# ============================================================================

def fig03_dataset():
    df = pd.read_csv(COMBINED)
    df["trial_id"] = df["subject"] + "|" + df["timestamp"]

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    # ---- panel (a): one-trial layout ---------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    # Pick a trial with good coverage
    trial_counts = df.groupby("trial_id").size()
    trial = trial_counts.sort_values(ascending=False).index[3]  # 4th biggest trial
    g = df[df["trial_id"] == trial]
    ax.scatter(g["origin_gaze_x"], g["origin_gaze_y"], s=18, c="#88aacc", alpha=0.6, label="raw mean (test)")
    ax.scatter(g["target_x"], g["target_y"], s=22, c="#264653", marker="s", label="target")
    for _, r in g.iterrows():
        ax.annotate("", xy=(r["target_x"], r["target_y"]), xytext=(r["origin_gaze_x"], r["origin_gaze_y"]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.4, alpha=0.4))
    ax.set_xlim(0, 1920); ax.set_ylim(1080, 0)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title(f"(a) one trial: 32 test fixations\n({trial[:20]}...)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")

    # ---- panel (b): per-subject anchor error box plot ----------------------
    ax = fig.add_subplot(gs[0, 1])
    err = np.linalg.norm(df[[f"{ANCHOR}_x", f"{ANCHOR}_y"]].values
                         - df[["target_x", "target_y"]].values, axis=1)
    df["anchor_l2"] = err
    subj_order = df.groupby("subject")["anchor_l2"].median().sort_values().index.tolist()
    box_data = [df[df["subject"] == s]["anchor_l2"].values for s in subj_order]
    bp = ax.boxplot(box_data, vert=True, widths=0.6, patch_artist=True,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cce5ff")
        patch.set_edgecolor("#264653")
    # Highlight the two dominant subjects
    dominant = {"Liu Jiaqi", "ZAnna"}
    for i, s in enumerate(subj_order):
        if s in dominant:
            bp["boxes"][i].set_edgecolor(C["highlight"])
            bp["boxes"][i].set_linewidth(1.6)
    ax.set_xticks(range(1, len(subj_order) + 1))
    ax.set_xticklabels(subj_order, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("per-fixation L2 (px)")
    ax.set_title("(b) per-subject baseline error\n(orange = dominant subjects)")
    ax.set_ylim(0, 200)

    # ---- panel (c): per-trial bias scatter ---------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bias = df.groupby(["subject", "timestamp"]).apply(
        lambda g: pd.Series({
            "bias_x": (g["target_x"] - g[f"{ANCHOR}_x"]).mean(),
            "bias_y": (g["target_y"] - g[f"{ANCHOR}_y"]).mean(),
            "n":      len(g),
        })
    ).reset_index()
    palette = plt.cm.tab20(np.linspace(0, 1, len(subj_order)))
    color_map = {s: palette[i] for i, s in enumerate(subj_order)}
    sizes = 8 + 5 * np.log1p(bias["n"])
    for s in subj_order:
        sub = bias[bias["subject"] == s]
        ax.scatter(sub["bias_x"], sub["bias_y"], s=sizes[sub.index], c=[color_map[s]], alpha=0.7, label=s, edgecolor="white", linewidth=0.4)
    for r in [25, 50, 75]:
        circle = plt.Circle((0, 0), r, fill=False, color="grey", linestyle="--", linewidth=0.5)
        ax.add_artist(circle)
        ax.text(r, 0, f" {r} px", fontsize=6, color="grey")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("per-trial bias x (px)"); ax.set_ylabel("per-trial bias y (px)")
    ax.set_title("(c) per-trial bias = mean(target − anchor)")
    ax.legend(loc="upper left", fontsize=5.5, ncol=2, frameon=False)

    # ---- panel (d): residual vector field for a single trial ---------------
    ax = fig.add_subplot(gs[1, 1])
    g = df[df["trial_id"] == trial]
    dx = g["target_x"] - g[f"{ANCHOR}_x"]
    dy = g["target_y"] - g[f"{ANCHOR}_y"]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    q = ax.quiver(g[f"{ANCHOR}_x"], g[f"{ANCHOR}_y"], dx, dy, mag,
                  cmap="magma_r", scale_units="xy", scale=1, width=0.005)
    ax.scatter(g["target_x"], g["target_y"], s=10, c="#264653", marker="s", label="target")
    cb = plt.colorbar(q, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("|residual| (px)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlim(0, 1920); ax.set_ylim(1080, 0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("(d) residual vector field after\nclassical calibration")

    fig.savefig(OUT / "fig03_dataset.pdf")
    plt.close(fig)
    print("wrote fig03_dataset.pdf")


# ============================================================================
# Figure 4 - shrinkage main result
# ============================================================================

def fig04_shrinkage():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # ---- panel (a): test L2 vs K -------------------------------------------
    ax = axes[0]
    Ks       = [0, 1, 2, 3, 5, 8, 12, 18]
    raw      = [44.69, 58.80, 48.98, 45.99, 43.27, 40.61, 37.95, 38.60]
    fixed    = [44.69, 43.26, 42.74, 41.97, 41.02, 39.37, 37.52, 38.18]
    learned  = [44.69, 63.88, 60.08, 61.29, 56.32, 54.13, 42.29, 40.20]

    ax.axhline(44.69, color=C["raw"], linestyle="--", linewidth=1.0, label="anchor (sim+RBF, K=0)")
    ax.plot(Ks, raw, color=C["highlight"], marker="o", linewidth=1.6, label="raw bias (λ=1)")
    ax.plot(Ks, fixed, color=C["leaky"], marker="s", linewidth=1.6, label="fixed-λ tuned per K")
    ax.plot(Ks, learned, color=C["ours"], marker="D", linewidth=2.0, label="learned shrinkage (LOSO, ours)")
    # callouts for K=12, K=18
    ax.annotate("−17 % vs anchor", xy=(12, 42.29), xytext=(13, 50),
                arrowprops=dict(arrowstyle="->", color=C["ours"], lw=0.7), fontsize=7.5, color=C["ours"])
    ax.annotate("−10 %", xy=(18, 40.20), xytext=(16.5, 33),
                arrowprops=dict(arrowstyle="->", color=C["ours"], lw=0.7), fontsize=7.5, color=C["ours"])
    ax.set_xlabel("context size K (extra fixations after standard 18-pt calibration)")
    ax.set_ylabel("test mean L2 (px)")
    ax.set_title("(a) honest pipeline: shrinkage vs raw mean")
    ax.set_xticks(Ks)
    ax.set_ylim(28, 70)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    # ---- panel (b): heatmap of fixed-lambda sweep --------------------------
    ax = axes[1]
    # Use the fixed-lambda sweep results
    with open(ROOT / "experiments" / "shrinkage_bias" / "results.json") as f:
        rows = json.load(f)
    df_lam = pd.DataFrame(rows)
    K_vals = sorted(df_lam["k"].unique())
    lam_vals = sorted(df_lam["lam"].unique())
    grid = np.zeros((len(K_vals), len(lam_vals)))
    for i, kk in enumerate(K_vals):
        for j, ll in enumerate(lam_vals):
            sub = df_lam[(df_lam["k"] == kk) & (df_lam["lam"] == ll)]
            if len(sub):
                grid[i, j] = sub["l2_mean"].iloc[0]
            else:
                grid[i, j] = np.nan
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r",
                   vmin=np.nanmin(grid), vmax=np.nanmin(grid) + 25)
    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([f"{l:.1f}" for l in lam_vals], fontsize=7)
    ax.set_yticks(range(len(K_vals)))
    ax.set_yticklabels(K_vals, fontsize=7)
    ax.set_xlabel("shrinkage λ")
    ax.set_ylabel("context size K")
    ax.set_title("(b) test L2 (px) by (K, λ)")
    # Annotate best λ per K with red star
    for i, kk in enumerate(K_vals):
        j = int(np.nanargmin(grid[i]))
        ax.scatter(j, i, marker="*", s=80, c="red", edgecolor="white", linewidth=0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("test L2 (px)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(OUT / "fig04_shrinkage.pdf")
    plt.close(fig)
    print("wrote fig04_shrinkage.pdf")


# ============================================================================
# Figure 7 - per-subject LOSO improvement
# ============================================================================

def fig07_loso():
    df_l = pd.read_csv(ROOT / "experiments" / "learned_shrinkage" / "results.csv")
    # Compute per-subject anchor (k=0 == anchor) L2 from the combined data
    df = pd.read_csv(COMBINED)
    df["anchor_l2"] = np.linalg.norm(
        df[[f"{ANCHOR}_x", f"{ANCHOR}_y"]].values - df[["target_x", "target_y"]].values, axis=1)
    anchor_per_subj = df.groupby("subject")["anchor_l2"].mean().to_dict()

    rows = []
    for _, r in df_l.iterrows():
        subj = r["held"]
        anchor = anchor_per_subj[subj]
        ours = r.get("learned_K12", np.nan)
        if not np.isnan(ours):
            rows.append({"subject": subj, "anchor": anchor, "ours": ours, "delta": ours - anchor})
        else:
            rows.append({"subject": subj, "anchor": anchor, "ours": np.nan, "delta": np.nan})

    rdf = pd.DataFrame(rows).sort_values("delta", ascending=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(rdf))
    height = 0.4
    ax.barh(y - height / 2, rdf["anchor"], height=height, color="#cccccc",
            edgecolor="black", linewidth=0.5, label="anchor (sim+RBF)")
    ax.barh(y + height / 2, rdf["ours"], height=height, color=C["ours"],
            edgecolor="black", linewidth=0.5, label="learned shrinkage K=12 (ours)")
    for i, (_, r) in enumerate(rdf.iterrows()):
        ax.text(r["anchor"] + 1, y[i] - height / 2, f"{r['anchor']:.1f}", va="center", fontsize=6.5)
        if not np.isnan(r["ours"]):
            ax.text(r["ours"] + 1, y[i] + height / 2, f"{r['ours']:.1f}", va="center", fontsize=6.5, color=C["ours"])
        else:
            ax.text(2, y[i] + height / 2, "(insufficient ctx)", va="center", fontsize=6, color="grey", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(rdf["subject"], fontsize=8)
    ax.set_xlabel("test mean L2 (px) — held-out subject")
    ax.set_xlim(0, 110)
    valid = rdf.dropna(subset=["delta"])
    txt = (f"# improved at K=12 : {(valid['delta'] < 0).sum()}/{len(valid)}\n"
           f"mean Δ : {valid['delta'].mean():+.2f} px\n"
           f"median Δ : {valid['delta'].median():+.2f} px")
    ax.text(0.97, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6))
    ax.legend(loc="lower right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.18))
    ax.set_title("Per-subject LOSO evaluation at K=12 context fixations")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT / "fig07_loso.pdf")
    plt.close(fig)
    print("wrote fig07_loso.pdf")


# ============================================================================
# Figure A5 - subject x baseline heatmap
# ============================================================================

def figA5_heatmap():
    df = pd.read_csv(COMBINED)
    methods = [
        ("origin_gaze", "raw"),
        ("pred_similarity", "similarity"),
        ("pred_poly", "poly"),
        ("pred_gpr", "gpr"),
        ("pred_tps", "tps"),
        ("pred_sim_pwa", "sim+PWA"),
        ("pred_sim_rbf_multiquadric_s2.0", "sim+RBF (anchor)"),
    ]
    # Subjects sorted by trial count desc
    subj_count = df.groupby("subject")["timestamp"].nunique().sort_values(ascending=False)
    subj_order = subj_count.index.tolist()

    grid = np.zeros((len(subj_order), len(methods) + 1))
    for i, s in enumerate(subj_order):
        sub = df[df["subject"] == s]
        for j, (col, _) in enumerate(methods):
            err = np.linalg.norm(sub[[f"{col}_x", f"{col}_y"]].values - sub[["target_x", "target_y"]].values, axis=1)
            grid[i, j] = err.mean()
        # Last column: shrinkage K=12 (use learned_shrinkage results if available)
        try:
            df_l = pd.read_csv(ROOT / "experiments" / "learned_shrinkage" / "results.csv")
            row = df_l[df_l["held"] == s]
            if len(row) and "learned_K12" in row.columns:
                v = row["learned_K12"].iloc[0]
                grid[i, -1] = v if not np.isnan(v) else grid[i, -2]
            else:
                grid[i, -1] = grid[i, -2]
        except Exception:
            grid[i, -1] = grid[i, -2]

    method_labels = [m[1] for m in methods] + ["shrinkage K=12 (ours)"]

    # Compute per-subject deltas vs anchor (sim+RBF) for color
    anchor_col = method_labels.index("sim+RBF (anchor)")
    deltas = grid - grid[:, anchor_col:anchor_col + 1]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    vmax = max(abs(deltas.min()), abs(deltas.max()), 20)
    im = ax.imshow(deltas, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(subj_order)))
    ax.set_yticklabels([f"{s} (n={subj_count[s]})" for s in subj_order], fontsize=7)
    for i in range(len(subj_order)):
        for j in range(len(method_labels)):
            ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(deltas[i,j]) < vmax * 0.6 else "white")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Δ vs anchor (px) — green = better", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Per-subject test L2 (px). Color encodes Δ vs sim+RBF anchor.")

    plt.tight_layout()
    fig.savefig(OUT / "figA5_heatmap.pdf")
    plt.close(fig)
    print("wrote figA5_heatmap.pdf")


if __name__ == "__main__":
    fig02_leakage()
    fig03_dataset()
    fig04_shrinkage()
    fig07_loso()
    figA5_heatmap()
