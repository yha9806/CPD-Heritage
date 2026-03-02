#!/usr/bin/env python3
"""
Generate CPD database statistics figure (3-panel bar chart) for the CPD paper.

Panels:
  (a) Top:          Dynasty distribution — horizontal bars, descending order
  (b) Bottom-left:  Primary category percentages (Landscape / Flower-Bird / Figure)
  (c) Bottom-right: Three-layer ontology attribute counts (L1=8, L2=10, L3=6)

Output: experiments/figures/cpd_statistics.pdf  (300 dpi)
        experiments/figures/cpd_statistics.png  (150 dpi)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data (from paper_numbers.json + paper text) ────────────────────────────
DYNASTY_RAW = {
    "Qing":              899,
    "Ming":              490,
    "Song":              266,
    "Yuan":               55,
    "Ming-Qing":          30,
    "Tang":               27,
    "Five Dyn. (Hou Shu)": 12,
    "Five Dyn.":           2,
    "Qi":                  2,
}

CATEGORY = {
    "Landscape":    59.6,
    "Flower-Bird":  28.4,
    "Figure":       12.0,
}

# Layer order: L1 → L2 → L3 (bottom to top so invert_yaxis shows L1 on top)
ONTOLOGY = {
    "L1: Factual (8)":   8,
    "L2: Artistic (10)": 10,
    "L3: Technical (6)": 6,
}


def main() -> None:
    # Sort dynasty by count descending
    dyn_items = sorted(DYNASTY_RAW.items(), key=lambda x: x[1], reverse=True)
    dyn_names = [n for n, _ in dyn_items]
    dyn_vals  = [v for _, v in dyn_items]

    # ── Layout: 1 row × 3 cols ───────────────────────────────────────────────
    fig = plt.figure(figsize=(6.5, 2.8))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[4.5, 2.5, 2.0],
        wspace=0.80,
        left=0.16,
        right=0.97,
        top=0.90,
        bottom=0.18,
    )
    ax_top = fig.add_subplot(gs[0, 0])   # dynasty
    ax_bl  = fig.add_subplot(gs[0, 1])   # category
    ax_br  = fig.add_subplot(gs[0, 2])   # ontology

    # ── (a) Dynasty ─────────────────────────────────────────────────────────
    cmap   = plt.cm.Blues
    n      = len(dyn_vals)
    colors = [cmap(0.35 + 0.55 * (n - 1 - i) / (n - 1)) for i in range(n)]

    y = np.arange(n)
    bars = ax_top.barh(y, dyn_vals, color=colors,
                       edgecolor="white", linewidth=0.4, height=0.68)
    ax_top.set_yticks(y)
    ax_top.set_yticklabels(dyn_names, fontsize=7.5)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Number of Paintings", fontsize=8)
    ax_top.set_title("(a) Dynasty Distribution", fontsize=9,
                     fontweight="bold", pad=3)
    ax_top.tick_params(axis="x", labelsize=7.5)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    for bar, v in zip(bars, dyn_vals):
        ax_top.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=7.5,
        )
    ax_top.set_xlim(0, max(dyn_vals) * 1.13)

    # ── (b) Primary Category ─────────────────────────────────────────────────
    cat_names  = list(CATEGORY.keys())
    cat_vals   = list(CATEGORY.values())
    cat_colors = ["#4A90D9", "#E67E22", "#27AE60"]

    yc = np.arange(len(cat_names))
    bars_c = ax_bl.barh(yc, cat_vals, color=cat_colors,
                        edgecolor="white", linewidth=0.4, height=0.6)
    ax_bl.set_yticks(yc)
    ax_bl.set_yticklabels(cat_names, fontsize=9)
    ax_bl.invert_yaxis()
    ax_bl.set_xlabel("Percentage (%)", fontsize=8)
    ax_bl.set_title("(b) Primary Category", fontsize=9,
                    fontweight="bold", pad=3)
    ax_bl.tick_params(axis="x", labelsize=8)
    ax_bl.spines["top"].set_visible(False)
    ax_bl.spines["right"].set_visible(False)

    xlim_cat = max(cat_vals) * 1.18
    for bar, v in zip(bars_c, cat_vals):
        if bar.get_width() > xlim_cat * 0.35:   # long bar → label inside
            ax_bl.text(
                bar.get_width() - 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="right", fontsize=8, color="white",
            )
        else:                                    # short bar → label outside
            ax_bl.text(
                bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8,
            )
    ax_bl.set_xlim(0, xlim_cat)

    # ── (c) Three-Layer Ontology ──────────────────────────────────────────────
    ont_names  = list(ONTOLOGY.keys())
    ont_vals   = list(ONTOLOGY.values())
    ont_colors = ["#E74C3C", "#1ABC9C", "#8E44AD"]

    yo = np.arange(len(ont_names))
    bars_o = ax_br.barh(yo, ont_vals, color=ont_colors,
                        edgecolor="white", linewidth=0.4, height=0.6)
    ax_br.set_yticks(yo)
    ax_br.set_yticklabels(ont_names, fontsize=8.5)
    ax_br.invert_yaxis()
    ax_br.set_xlabel("# Attribute Types", fontsize=8)
    ax_br.set_title("(c) Ontology Layers", fontsize=9,
                    fontweight="bold", pad=3)
    ax_br.tick_params(axis="x", labelsize=8)
    ax_br.spines["top"].set_visible(False)
    ax_br.spines["right"].set_visible(False)

    for bar, v in zip(bars_o, ont_vals):
        ax_br.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=9,
        )
    ax_br.set_xlim(0, max(ont_vals) * 1.30)
    ax_br.set_xticks([0, 4, 8, 12])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = os.path.join(OUTPUT_DIR, "cpd_statistics.pdf")
    out_png = os.path.join(OUTPUT_DIR, "cpd_statistics.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
