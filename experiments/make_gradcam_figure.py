#!/usr/bin/env python3
"""
Generate a 3×4 Grad-CAM comparison figure for the CPD paper.
Rows: 3 representative paintings (one per category)
Cols: Original + 224 + 512 + 1024 heatmaps
Model: ConvNeXt V2-Tiny (best 3-class model, 96.7% at 512)

Layout improvements:
  - figsize=(14, 10): wider canvas with more room for row labels
  - Row labels via fig.text() centred vertically on each row
  - Shared colorbar on the right labelled "Attention"
  - Ground-truth category badge on the Original column
  - Cell border linewidth=0.8, color="#666"
  - No suptitle (caption carries the explanation)
"""
import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "figures", "gradcam")
DATA_ROOT   = os.environ.get("CPD_DATA_ROOT", "./data")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "figures")

# Representative paintings: one per primary category
PAINTINGS = [
    (
        "清_焦秉贞_历朝贤后故事图_页_工笔,人物_ _人物 _  _  _ _1da55811",
        "(a) Gongbi Figure (Qing)",
        "Figure",
    ),
    (
        "清_郑板桥_墨笔竹石图_轴_写意,花鸟_竹子,竹叶,叶子,文字,印章 _ _49cmX25cm_北京_479d17ec",
        "(b) Xieyi Flower-Bird (Qing)",
        "Flower-Bird",
    ),
    (
        "明_蒋乾_仿王蒙山水_轴_水墨,山水_山水_ _94cmX35cm_台北博物馆_79a10912",
        "(c) Ink Landscape (Ming)",
        "Landscape",
    ),
]

MODEL       = "convnextv2_tiny"
RESOLUTIONS = [224, 512, 1024]


def load_original(stem: str) -> Optional[Image.Image]:
    path = os.path.join(DATA_ROOT, stem + ".jpg")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


def load_gradcam(stem: str, res: int) -> Optional[Image.Image]:
    run_dir = os.path.join(GRADCAM_DIR, f"{MODEL}_{res}_3class")
    for suffix in ["_OK.jpg", "_WRONG.jpg"]:
        path = os.path.join(run_dir, stem + suffix)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    return None


def add_gt_badge(ax: "plt.Axes", label: str) -> None:
    """Overlay a semi-transparent GT category badge on the top-left of ax."""
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        fontsize=8, fontweight="bold", color="white",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, lw=0),
    )


def main() -> None:
    n_rows = len(PAINTINGS)
    n_cols = 4  # Original + 224 + 512 + 1024

    # Extra horizontal space on left (row labels) and right (colorbar)
    fig = plt.figure(figsize=(14, 10))

    # GridSpec: 3 rows × 4 cols for images + 1 col for colorbar
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[1, 1, 1, 1, 0.06],
        wspace=0.04,
        hspace=0.12,
        left=0.09,
        right=0.93,
        top=0.94,
        bottom=0.03,
    )

    col_titles = ["Original", "224×224", "512×512", "1024×1024"]
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    for row, (stem, row_label, cat) in enumerate(PAINTINGS):
        orig = load_original(stem)
        images = [orig] + [load_gradcam(stem, r) for r in RESOLUTIONS]

        for col, img in enumerate(images):
            ax = axes[row][col]
            if img is not None:
                ax.imshow(np.array(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#666666")

            # Column header (top row only)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11,
                             fontweight="bold", pad=5)

            # GT badge on Original column
            if col == 0:
                add_gt_badge(ax, cat)

        # Row label — centred vertically on the row, left of all images
        # Compute the vertical centre in figure coordinates
        row_axes = axes[row]
        # Use the bbox of first and last ax in the row to find midpoint
        fig.canvas.draw()  # needed for get_position
        pos0 = row_axes[0].get_position()
        y_mid = pos0.y0 + pos0.height / 2
        fig.text(
            0.005, y_mid, row_label,
            va="center", ha="left",
            fontsize=9.5, fontweight="bold",
            rotation=90,
        )

    # ── Shared colorbar (jet: cold=low, hot=high attention) ──────────────
    cbar_ax = fig.add_subplot(gs[:, n_cols])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm   = mcm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention", fontsize=10, labelpad=6)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.ax.tick_params(labelsize=8)

    out_pdf = os.path.join(OUTPUT_DIR, "gradcam_comparison.pdf")
    out_png = os.path.join(OUTPUT_DIR, "gradcam_comparison.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
