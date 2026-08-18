"""
generate_figures.py
===================
Generates all publication-quality figures for:
  "Lottery Ticket Compression of Genomic Transformers for
   Clinically Oriented Variant Pathogenicity Classification"

Run:  python generate_figures.py
Output: all figures saved as PDF (for LaTeX) + PNG 300 DPI (for preview)

Requirements:
    pip install matplotlib numpy seaborn scipy
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# GLOBAL STYLE  (IEEE-grade: clean, no chart-junk, 300 DPI)
# ----------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.5,
    "grid.color":         "#cccccc",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
})

# IBM colorblind-safe palette
IBM_BLUE   = "#648FFF"
IBM_ORANGE = "#FE6100"
IBM_PURPLE = "#785EF0"
IBM_GREEN  = "#009E73"
IBM_RED    = "#DC267F"
IBM_YELLOW = "#FFB000"
GRAY       = "#6B6B6B"

# ----------------------------------------------------------------
# DATA
# ----------------------------------------------------------------
SPARSITY        = [0, 10, 20, 30, 50, 60, 70, 80, 90]
MAG_AUROC       = [89.42, 88.50, 89.18, 88.93, 89.35, 88.08, 79.83, 73.04, 70.34]
FISHER_AUROC    = [89.42, 88.71, 88.63, 88.30, 84.36, 75.87, None, None, None]
EMA_AUROC       = [89.42, 87.84, 83.43, 74.00, 69.07, 68.95, None, None, None]
MOVE_AUROC      = [89.42, 86.57, 79.92, 75.78, 71.86, 71.90, None, None, None]
DRF_AUROC       = [89.42, 88.48, 88.64, 88.97, 89.09, 87.99, None, None, None]

MAG_ACC         = [81.24, 80.38, 80.75, 80.77, 81.13, 78.45, 72.05, 66.80, 64.82]
DENSE_AUROC     = 89.42
DENSE_ACC       = 81.24

SPARSITY_MAIN   = [10, 20, 30, 50, 60]
DELTA_AUROC     = [-0.92, -0.24, -0.49, -0.07, -1.34, -9.59, -16.38, -19.08]
SPARSITY_DELTA  = [10, 20, 30, 50, 60, 70, 80, 90]

HEATMAP_DATA = np.array([
    [88.50, 89.18, 88.93, 89.35, 88.08],  # Magnitude
    [88.71, 88.63, 88.30, 84.36, 75.87],  # Fisher
    [87.84, 83.43, 74.00, 69.07, 68.95],  # EMA Gradient
    [86.57, 79.92, 75.78, 71.86, 71.90],  # Weight Movement
    [88.48, 88.64, 88.97, 89.09, 87.99],  # Hybrid DRF
])
SCORER_LABELS   = ["Magnitude", "Fisher Info.", "EMA Gradient",
                   "Weight Movement", "Hybrid DRF"]
SPARSITY_LABELS = ["10%", "20%", "30%", "50%", "60%"]

METRICS_100M  = [89.35, 81.13, 62.3, 92.85, 85.46, 89.01, 89.11]
METRICS_50M   = [84.26, 75.98, 52.1, 87.87, 78.96, 85.33, 84.05]
METRIC_LABELS = ["Val\nAUROC", "Val\nAcc", "Val\nMCC×100",
                 "ClinVar\nAUROC", "dbSNP\nAUROC",
                 "cBio+gnomAD\nAUROC", "Mean Unseen\nAUROC"]

def _save(fig, name):
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ================================================================
# FIGURE 1 — Pruning Signal Comparison (enhanced)
# ================================================================
def fig_scorer_comparison():
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    sp = np.array(SPARSITY)

    def _plot(data, color, marker, label, zorder=2):
        x = [s for s, v in zip(sp, data) if v is not None]
        y = [v for v in data if v is not None]
        ax.plot(x, y, color=color, marker=marker, label=label,
                zorder=zorder, clip_on=False)

    # Dense baseline band
    ax.axhline(DENSE_AUROC, color=GRAY, lw=1.2, ls="--", zorder=1)
    ax.axhspan(DENSE_AUROC - 0.5, DENSE_AUROC + 0.5, alpha=0.08,
               color=GRAY, zorder=0)
    ax.text(91, DENSE_AUROC + 0.15, "Dense baseline", color=GRAY,
            fontsize=9, va="bottom", ha="right")

    _plot(MAG_AUROC,    IBM_BLUE,   "o", "Magnitude",            zorder=5)
    _plot(FISHER_AUROC, IBM_ORANGE, "s", "Fisher Information",   zorder=4)
    _plot(EMA_AUROC,    IBM_GREEN,  "^", "EMA Gradient",         zorder=3)
    _plot(MOVE_AUROC,   IBM_PURPLE, "D", "Weight Movement",      zorder=3)
    _plot(DRF_AUROC,    IBM_RED,    "P", "Dynamic Rank Fusion",  zorder=4)

    # Annotate 50%-optimal point
    ax.annotate("50% Optimal\n(89.35%)",
                xy=(50, 89.35), xytext=(38, 87.2),
                arrowprops=dict(arrowstyle="->", color=IBM_BLUE,
                                connectionstyle="arc3,rad=-0.2"),
                fontsize=9, color=IBM_BLUE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=IBM_BLUE, lw=0.8))

    # Compression cliff shading
    ax.axvspan(60, 90, alpha=0.06, color=IBM_RED, zorder=0)
    ax.text(74, 66.5, "Compression\ncliff", color=IBM_RED,
            fontsize=8.5, ha="center", va="bottom")

    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Validation AUROC (%)")
    ax.set_title("Pruning Signal Comparison Across Sparsity Levels")
    ax.set_xlim(-2, 93)
    ax.set_ylim(64, 91)
    ax.set_xticks([0, 10, 20, 30, 50, 60, 70, 80, 90])
    ax.legend(loc="lower left", ncol=1, frameon=True)
    fig.tight_layout()
    _save(fig, "fig1_scorer_comparison")


# ================================================================
# FIGURE 2 — Metric vs Sparsity (AUROC + Accuracy dual panel)
# ================================================================
def fig_metric_vs_sparsity():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=False)

    sp = np.array(SPARSITY)

    # --- (a) AUROC ---
    ax = axes[0]
    ax.plot(sp, MAG_AUROC, color=IBM_BLUE, marker="o", zorder=3)
    ax.axhline(DENSE_AUROC, color=GRAY, lw=1.2, ls="--")
    ax.text(2, DENSE_AUROC - 0.6, f"Dense = {DENSE_AUROC}%",
            color=GRAY, fontsize=8.5, style="italic")
    # stable zone
    ax.axvspan(0, 60, alpha=0.07, color=IBM_BLUE)
    # compression cliff arrow
    ax.annotate("Compression\ncliff", xy=(60, 88.08),
                xytext=(63, 85.5),
                arrowprops=dict(arrowstyle="->", color=IBM_RED,
                                lw=1.2),
                fontsize=8.5, color=IBM_RED)
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("AUROC (%)")
    ax.set_title("(a) AUROC vs Sparsity")
    ax.set_xlim(-2, 93)
    ax.set_ylim(68, 91.5)
    ax.set_xticks([0, 10, 20, 30, 50, 60, 70, 80, 90])

    # --- (b) Accuracy ---
    ax = axes[1]
    ax.plot(sp, MAG_ACC, color=IBM_ORANGE, marker="s", zorder=3)
    ax.axhline(DENSE_ACC, color=GRAY, lw=1.2, ls="--")
    ax.text(2, DENSE_ACC - 0.8, f"Dense = {DENSE_ACC}%",
            color=GRAY, fontsize=8.5, style="italic")
    ax.axvspan(0, 60, alpha=0.07, color=IBM_ORANGE)
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Accuracy vs Sparsity")
    ax.set_xlim(-2, 93)
    ax.set_ylim(62, 83.5)
    ax.set_xticks([0, 10, 20, 30, 50, 60, 70, 80, 90])

    fig.suptitle("Magnitude-Based Pruning: Performance vs.\ Sparsity",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig2_metric_vs_sparsity")


# ================================================================
# FIGURE 3 — ΔAUROCs bar chart (enhanced colors + labels)
# ================================================================
def fig_delta_auroc():
    sp     = np.array(SPARSITY_DELTA)
    deltas = np.array(DELTA_AUROC)

    colors = []
    for d in deltas:
        if abs(d) < 1.0:
            colors.append(IBM_BLUE)
        elif abs(d) < 5.0:
            colors.append(IBM_YELLOW)
        else:
            colors.append(IBM_RED)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(sp, deltas, width=7.0, color=colors, zorder=3,
                  edgecolor="white", linewidth=0.6)

    # Acceptable threshold line
    ax.axhline(-1.5, color=GRAY, lw=1.0, ls=":", zorder=2)
    ax.text(91, -1.35, "Acceptable\nthreshold (−1.5%)",
            color=GRAY, fontsize=8.5, va="bottom", ha="right")
    ax.axhline(0, color="black", lw=0.8, zorder=2)

    # Value labels
    for bar, d in zip(bars, deltas):
        ypos = d - 0.3 if d < 0 else d + 0.1
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{d:.2f}%", ha="center", va="top",
                fontsize=9, fontweight="bold",
                color="white" if abs(d) > 2 else IBM_BLUE)

    # Legend
    legend_elements = [
        mpatches.Patch(color=IBM_BLUE,   label="< 1 pp drop (operational)"),
        mpatches.Patch(color=IBM_YELLOW, label="1–5 pp drop (marginal)"),
        mpatches.Patch(color=IBM_RED,    label="> 5 pp drop (unacceptable)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              frameon=True, fontsize=9)

    ax.set_xlabel("Sparsity Level (%)")
    ax.set_ylabel("ΔAUROC vs Dense Baseline (pp)")
    ax.set_title("Impact of Sparsity on AUROC (Magnitude Pruning)")
    ax.set_xticks(sp)
    ax.set_xticklabels([f"{s}%" for s in sp])
    ax.set_xlim(5, 95)
    ax.set_ylim(-21, 1.5)
    fig.tight_layout()
    _save(fig, "fig3_delta_auroc")


# ================================================================
# FIGURE 4 — 100M@50% vs Native 50M (grouped horizontal bars)
# ================================================================
def fig_100m_vs_50m():
    n      = len(METRIC_LABELS)
    x      = np.arange(n)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    b1 = ax.bar(x - width/2, METRICS_100M, width, label="100M @ 50% Sparse",
                color=IBM_BLUE, zorder=3, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, METRICS_50M,  width, label="Native NT-v2-50M",
                color=IBM_ORANGE, zorder=3, edgecolor="white", linewidth=0.5)

    diffs = [a - b for a, b in zip(METRICS_100M, METRICS_50M)]
    for xi, (v1, v2, d) in enumerate(zip(METRICS_100M, METRICS_50M, diffs)):
        ymax = max(v1, v2)
        ax.text(xi - width/2, v1 + 0.4, f"{v1:.1f}", ha="center",
                va="bottom", fontsize=8.5, color=IBM_BLUE, fontweight="bold")
        ax.text(xi + width/2, v2 + 0.4, f"{v2:.1f}", ha="center",
                va="bottom", fontsize=8.5, color=IBM_ORANGE, fontweight="bold")
        # Difference annotation
        ax.text(xi, ymax + 2.8,
                f"+{d:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=IBM_RED, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=IBM_RED, lw=0.7, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9.5)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Compressed NT-v2-100M (50% Sparse) vs Native NT-v2-50M")
    ax.legend(loc="lower right", frameon=True)
    ax.text(0.02, 0.97,
            "Note: Val MCC values scaled ×100 for visual comparison",
            transform=ax.transAxes, fontsize=8, color=GRAY,
            va="top", style="italic")
    fig.tight_layout()
    _save(fig, "fig4_100m_vs_50m")


# ================================================================
# FIGURE 5 — AUROC Heatmap (diverging from dense baseline)
# ================================================================
def fig_heatmap():
    fig, ax = plt.subplots(figsize=(8.0, 4.0))

    # Diverging colormap centered at dense baseline
    center = DENSE_AUROC
    vmin, vmax = 68, 91

    cmap = sns.diverging_palette(10, 130, s=85, l=50, as_cmap=True)

    im = ax.imshow(HEATMAP_DATA, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax)

    # Cell annotations — font color based on value
    bold_threshold = 88.5
    for i in range(HEATMAP_DATA.shape[0]):
        for j in range(HEATMAP_DATA.shape[1]):
            val  = HEATMAP_DATA[i, j]
            bold = val >= bold_threshold
            fc   = "white" if val < 78 else "black"
            ax.text(j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if bold else "normal",
                    color=fc)

    ax.set_xticks(range(len(SPARSITY_LABELS)))
    ax.set_xticklabels(SPARSITY_LABELS)
    ax.set_yticks(range(len(SCORER_LABELS)))
    ax.set_yticklabels(SCORER_LABELS)
    ax.set_xlabel("Sparsity Level")
    ax.set_title("Validation AUROC (%) by Scorer and Sparsity Level")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("AUROC (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Mark dense baseline on colorbar
    cbar.ax.axhline((center - vmin) / (vmax - vmin),
                    color="black", lw=1.5, ls="--")
    cbar.ax.text(2.8, (center - vmin) / (vmax - vmin),
                 f"Dense\n{center}%", va="center", fontsize=7.5,
                 color=GRAY, transform=cbar.ax.transAxes)

    fig.tight_layout()
    _save(fig, "fig5_heatmap")


# ================================================================
# FIGURE 6 — Data Curation Funnel (NEW)
# ================================================================
def fig_data_funnel():
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9.5, bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor="white",
                                       linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color="white", zorder=4, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                   lw=1.4),
                    zorder=2)

    # Source boxes (row 1)
    sources = [
        ("ClinVar\n(clinical\nassertions)",   "#1A6DA5"),
        ("dbSNP\n(broad\ncatalogue)",          "#2E9E6E"),
        ("gnomAD\n(population\nfrequency)",    "#B5522C"),
        ("cBioPortal\n(somatic\noncology)",    "#7B3FA0"),
    ]
    xs = [0.3, 2.8, 5.3, 7.8]
    for (label, color), x in zip(sources, xs):
        box(x, 8.2, 1.8, 1.5, label, color, fontsize=8.5)

    # Arrows from sources to filter
    for x in xs:
        arrow(x + 0.9, 8.2, 5.0, 6.8)

    # Filter / Harmonize box
    box(3.0, 5.9, 4.0, 0.85,
        "SNV filter · chromosome restriction · annotation confidence",
        "#444444", fontsize=8.0)

    # Deduplication box
    arrow(5.0, 5.9, 5.0, 5.2)
    box(2.5, 4.35, 5.0, 0.8,
        "Deduplication by (chr, pos, ref, alt)   ⟶   217,026 variants",
        "#222222", fontsize=8.5, bold=True)

    # Leakage audit
    arrow(5.0, 4.35, 5.0, 3.65)
    box(3.2, 3.1, 3.6, 0.5,
        "Automated leakage audit (no coordinate overlap)",
        "#8B0000", fontsize=8.0)

    # Split
    arrow(5.0, 3.1, 5.0, 2.5)

    # Train / Val / Test boxes
    split = [
        ("Training\n100,000",   IBM_BLUE,   1.0),
        ("Validation\n24,914",  IBM_GREEN,  4.1),
        ("Test Holdouts\n92,112", IBM_RED,  7.2),
    ]
    for label, color, x in split:
        box(x, 1.1, 2.0, 1.2, label, color, fontsize=9.0, bold=True)
        arrow(5.0, 2.5, x + 1.0, 2.3)

    # Test sub-boxes
    test_sets = [
        ("ClinVar\n30,796",         "#1A6DA5", 5.9),
        ("dbSNP\n30,771",           "#2E9E6E", 7.1),
        ("cBio+\ngnomAD 30,545",    "#7B3FA0", 8.3),
    ]
    for label, color, x in test_sets:
        box(x, 0.05, 1.1, 0.95, label, color, fontsize=7.5)
        arrow(8.2, 1.1, x + 0.55, 1.0)

    ax.set_title("Multi-Source Variant Benchmark Construction Pipeline",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "fig6_data_funnel")


# ================================================================
# FIGURE 7 — Per-layer sparsity heatmap skeleton
#            (runs without GPU; to be filled with real mask data)
# ================================================================
def fig_layer_sparsity_template():
    """
    Template figure showing expected per-layer sparsity structure.
    Replace `sparsity_matrix` with real data from:

        import torch
        state = torch.load("sparse_50pct_magnitude_mask.pt")
        # state is a dict: layer_name -> binary mask tensor
        # Compute fraction pruned per (layer, weight_type):
        #   pruned = 1 - mask.float().mean()
    """

    # PLACEHOLDER DATA — replace with output of mask analysis
    np.random.seed(42)
    n_layers = 22
    weight_types = ["Q", "K", "V", "FFN1", "FFN2"]
    # Simulate: deeper layers and FFN weights pruned less aggressively
    base = np.random.uniform(0.40, 0.62, (n_layers, len(weight_types)))
    # FFN typically retains more magnitude mass
    base[:, 3] -= 0.08
    base[:, 4] -= 0.06
    base = np.clip(base, 0.35, 0.65)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    cmap = plt.cm.RdYlGn_r   # red = more pruned, green = less pruned
    im = ax.imshow(base.T, aspect="auto", cmap=cmap, vmin=0.3, vmax=0.7)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i+1}" for i in range(n_layers)], fontsize=8)
    ax.set_yticks(range(len(weight_types)))
    ax.set_yticklabels(weight_types)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Weight Type")
    ax.set_title(
        "Per-Layer Fraction Pruned at 50% Sparsity (Magnitude)\n"
        "[PLACEHOLDER — replace with real mask statistics]",
        fontsize=11, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction Pruned", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.axvline(10.5, color="white", lw=1.5, ls="--", alpha=0.6)
    ax.text(5.0, -0.9, "Earlier layers", ha="center", fontsize=8.5,
            color=GRAY, style="italic")
    ax.text(15.5, -0.9, "Later layers", ha="center", fontsize=8.5,
            color=GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig7_layer_sparsity_template")
    print("  NOTE: Replace placeholder data with real mask statistics.")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("Generating publication figures...")
    fig_scorer_comparison()
    fig_metric_vs_sparsity()
    fig_delta_auroc()
    fig_100m_vs_50m()
    fig_heatmap()
    fig_data_funnel()
    fig_layer_sparsity_template()
    print("\nDone. All figures saved as PDF + PNG (300 DPI).")
    print("Use the PDF versions for LaTeX inclusion (best quality).")
    print("Rename output files to match LaTeX \\includegraphics paths.")