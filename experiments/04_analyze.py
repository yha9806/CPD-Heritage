#!/usr/bin/env python3
"""
04_analyze.py — Generate analysis tables, figures, and paper_numbers.json
from completed experiment results.

Usage:
  python 04_analyze.py           # generate all outputs
  python 04_analyze.py --latex   # also output LaTeX table code
"""
import os, sys, json, argparse
from collections import defaultdict
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ════════════════════════════════════════════════════════
# Load results
# ════════════════════════════════════════════════════════

def load_all_results() -> dict[str, dict[str, Any]]:
    """Load all completed result JSONs."""
    results = {}
    for f in sorted(os.listdir(RESULTS_DIR)):
        if not f.endswith(".json") or f == "paper_numbers.json":
            continue
        with open(os.path.join(RESULTS_DIR, f)) as fh:
            d = json.load(fh)
        if d.get("status") == "OOM_FAILED":
            continue
        results[d["run_id"]] = d
    return results


def build_tables(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict]]:
    """Build accuracy and F1 matrices."""
    models = MODEL_PRIORITY
    resolutions = RESOLUTION_PRIORITY
    tasks = ["3class", "fine"]

    tables = {}
    for task in tasks:
        acc_matrix = {}
        f1_matrix = {}
        for m in models:
            acc_row = {}
            f1_row = {}
            for r in resolutions:
                rid = f"{m}_{r}_{task}"
                if rid in results:
                    acc_row[r] = results[rid]["test_accuracy"]
                    f1_row[r] = results[rid]["macro_f1"]
                else:
                    acc_row[r] = None
                    f1_row[r] = None
            acc_matrix[m] = acc_row
            f1_matrix[m] = f1_row
        tables[task] = {"accuracy": acc_matrix, "f1": f1_matrix}
    return tables


# ════════════════════════════════════════════════════════
# Pretty-print tables
# ════════════════════════════════════════════════════════

MODEL_DISPLAY = {
    "efficientnet_b0": "EfficientNet-B0",
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "convnextv2_tiny": "ConvNeXt V2-Tiny",
    "resnet50": "ResNet-50",
    "vit_base_16": "ViT-B/16",
    "swinv2_tiny": "Swin V2-Tiny",
}

PARADIGM_DISPLAY = {
    "sup_cnn": "Supervised CNN",
    "modern_cnn": "Modern CNN + MAE",
    "sup_vit": "Supervised ViT",
    "hier_vit": "Hierarchical ViT",
    "self_supervised": "Self-Supervised",
}

PARADIGM_GROUP = {
    "sup_cnn": "CNN",
    "modern_cnn": "CNN",
    "sup_vit": "ViT",
    "hier_vit": "ViT",
    "self_supervised": "SSL",
}


def print_table(tables: dict[str, dict[str, dict]], task: str, metric: str = "accuracy") -> tuple[float, str, int]:
    """Print a formatted accuracy/F1 table."""
    mat = tables[task][metric]
    label = "Accuracy" if metric == "accuracy" else "Macro-F1"
    nc = "3" if task == "3class" else "7"
    print(f"\n{'='*70}")
    print(f"  {nc}-Class {label}")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'224':>8} {'512':>8} {'1024':>8}  {'Best':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

    best_overall = 0
    best_model = ""
    best_res = 0
    for m in MODEL_PRIORITY:
        row = mat[m]
        vals = [row[r] for r in [224, 512, 1024] if row[r] is not None]
        best = max(vals) if vals else None
        if best and best > best_overall:
            best_overall = best
            best_model = m
            best_res = max(r for r in [224, 512, 1024] if row[r] == best)

        cells = []
        for r in [224, 512, 1024]:
            v = row[r]
            if v is None:
                cells.append(f"{'--':>8}")
            else:
                cells.append(f"{v*100:>7.1f}%")

        bv = f"{best*100:.1f}%" if best else "--"
        print(f"  {MODEL_DISPLAY[m]:<22} {cells[0]} {cells[1]} {cells[2]}  {bv:>8}")

    print(f"\n  Best: {MODEL_DISPLAY.get(best_model, best_model)} @ {best_res}px = {best_overall*100:.1f}%")
    return best_overall, best_model, best_res


def print_per_class(results: dict[str, dict[str, Any]], model: str, resolution: int, task: str) -> None:
    """Print per-class metrics for a specific run."""
    rid = f"{model}_{resolution}_{task}"
    if rid not in results:
        print(f"  [!] {rid} not found")
        return
    d = results[rid]
    print(f"\n{'='*70}")
    print(f"  Per-Class Metrics: {MODEL_DISPLAY[model]} @ {resolution}px ({task})")
    print(f"{'='*70}")
    print(f"  {'Class':<25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cls, m in d["per_class"].items():
        print(f"  {cls:<25} {m['precision']*100:>7.1f}% {m['recall']*100:>7.1f}% "
              f"{m['f1']*100:>7.1f}% {m['support']:>8.0f}")


# ════════════════════════════════════════════════════════
# Resolution analysis
# ════════════════════════════════════════════════════════

def resolution_analysis(tables: dict[str, dict[str, dict]], task: str) -> None:
    """Analyze resolution impact."""
    mat = tables[task]["accuracy"]
    print(f"\n{'='*70}")
    nc = "3" if task == "3class" else "7"
    print(f"  Resolution Impact ({nc}-class)")
    print(f"{'='*70}")
    for m in MODEL_PRIORITY:
        row = mat[m]
        v224 = row.get(224)
        v512 = row.get(512)
        v1024 = row.get(1024)
        if v224 and v512:
            d1 = (v512 - v224) * 100
            print(f"  {MODEL_DISPLAY[m]:<22} 224→512: {d1:+.1f}pp", end="")
            if v1024:
                d2 = (v1024 - v512) * 100
                d3 = (v1024 - v224) * 100
                print(f"  512→1024: {d2:+.1f}pp  224→1024: {d3:+.1f}pp")
            else:
                print()


# ════════════════════════════════════════════════════════
# Paradigm analysis
# ════════════════════════════════════════════════════════

def paradigm_analysis(results: dict[str, dict[str, Any]], task: str) -> None:
    """Compare paradigms: CNN vs ViT vs SSL."""
    groups = defaultdict(list)
    for m in MODEL_PRIORITY:
        paradigm = MODELS[m]["paradigm"]
        group = PARADIGM_GROUP[paradigm]
        for r in RESOLUTION_PRIORITY:
            rid = f"{m}_{r}_{task}"
            if rid in results:
                groups[group].append(results[rid]["test_accuracy"])

    print(f"\n{'='*70}")
    nc = "3" if task == "3class" else "7"
    print(f"  Paradigm Comparison ({nc}-class)")
    print(f"{'='*70}")
    for g in ["CNN", "ViT", "SSL"]:
        if g in groups:
            vals = groups[g]
            print(f"  {g:<6} mean={np.mean(vals)*100:.1f}%  "
                  f"max={np.max(vals)*100:.1f}%  min={np.min(vals)*100:.1f}%  "
                  f"n={len(vals)}")


# ════════════════════════════════════════════════════════
# Generate paper_numbers.json
# ════════════════════════════════════════════════════════

def generate_paper_numbers(results: dict[str, dict[str, Any]], tables: dict[str, dict[str, dict]], stats: dict[str, Any]) -> dict[str, Any]:
    """Generate all numbers needed for the paper."""
    pn = {}

    # Dataset numbers
    pn["total_files"] = stats["total_files"]
    pn["usable_images"] = stats["unique_paintings"]
    pn["n_classes_3"] = stats["n_classes_3"]
    pn["n_classes_fine"] = stats["n_classes_fine"]
    pn["split_train"] = stats["split_train"]
    pn["split_val"] = stats["split_val"]
    pn["split_test"] = stats["split_test"]

    # Distribution percentages
    pn["landscape_pct"] = stats["class_3_percentages"]["Landscape"]
    pn["flowerbird_pct"] = stats["class_3_percentages"]["FlowerBird"]
    pn["figure_pct"] = stats["class_3_percentages"]["Figure"]

    # Dynasty distribution (denominator = total paintings, not just those with dynasty info)
    pn["dynasty_distribution"] = stats.get("dynasty_distribution", {})
    total_paintings = stats["unique_paintings"]
    if total_paintings > 0:
        song_qing = sum(v for k, v in stats.get("dynasty_distribution", {}).items()
                        if k in ["宋", "元", "明", "清", "明清"])
        pn["song_qing_pct"] = round(song_qing / total_paintings * 100, 1)

    # Best models for each task
    for task in ["3class", "fine"]:
        mat = tables[task]["accuracy"]
        f1m = tables[task]["f1"]
        best_acc = 0
        best_m = ""
        best_r = 0
        for m in MODEL_PRIORITY:
            for r in RESOLUTION_PRIORITY:
                v = mat[m].get(r)
                if v and v > best_acc:
                    best_acc = v
                    best_m = m
                    best_r = r

        nc = "3" if task == "3class" else "7"
        pn[f"best_{task}_acc"] = round(best_acc * 100, 1)
        pn[f"best_{task}_model"] = MODEL_DISPLAY.get(best_m, best_m)
        pn[f"best_{task}_resolution"] = best_r
        pn[f"best_{task}_f1"] = round(f1m[best_m][best_r] * 100, 1) if f1m[best_m].get(best_r) else None

        # Per-class for best model
        rid = f"{best_m}_{best_r}_{task}"
        if rid in results:
            pn[f"best_{task}_per_class"] = results[rid]["per_class"]

    # Resolution improvement (best model, 3-class)
    for task in ["3class", "fine"]:
        mat = tables[task]["accuracy"]
        improvements = []
        for m in MODEL_PRIORITY:
            v224 = mat[m].get(224)
            v512 = mat[m].get(512)
            v1024 = mat[m].get(1024)
            if v224 and v512:
                improvements.append({
                    "model": m,
                    "224_to_512": round((v512 - v224) * 100, 1),
                    "512_to_1024": round((v1024 - v512) * 100, 1) if v1024 else None,
                    "224_to_1024": round((v1024 - v224) * 100, 1) if v1024 else None,
                })
        if improvements:
            avg_224_512 = np.mean([x["224_to_512"] for x in improvements])
            valid_512_1024 = [x["512_to_1024"] for x in improvements if x["512_to_1024"] is not None]
            avg_512_1024 = np.mean(valid_512_1024) if valid_512_1024 else None
            pn[f"res_improvement_{task}"] = {
                "per_model": improvements,
                "avg_224_to_512_pp": round(avg_224_512, 1),
                "avg_512_to_1024_pp": round(avg_512_1024, 1) if avg_512_1024 else None,
            }

    # Paradigm comparison
    for task in ["3class", "fine"]:
        groups = defaultdict(list)
        for m in MODEL_PRIORITY:
            group = PARADIGM_GROUP[MODELS[m]["paradigm"]]
            for r in RESOLUTION_PRIORITY:
                rid = f"{m}_{r}_{task}"
                if rid in results:
                    groups[group].append(results[rid]["test_accuracy"])
        pn[f"paradigm_{task}"] = {
            g: {
                "mean": round(np.mean(vals) * 100, 1),
                "max": round(np.max(vals) * 100, 1),
                "min": round(np.min(vals) * 100, 1),
            } for g, vals in groups.items()
        }

    # Full accuracy matrix
    for task in ["3class", "fine"]:
        mat = tables[task]["accuracy"]
        pn[f"acc_matrix_{task}"] = {
            m: {str(r): round(v * 100, 1) if v else None
                for r, v in mat[m].items()}
            for m in MODEL_PRIORITY
        }

    # Training stats
    total_time = sum(r["train_time_sec"] for r in results.values()
                     if isinstance(r.get("train_time_sec"), (int, float)))
    pn["total_train_time_hours"] = round(total_time / 3600, 1)
    pn["num_completed_runs"] = len(results)
    pn["num_total_runs"] = len(MODEL_PRIORITY) * len(RESOLUTION_PRIORITY) * 2

    return pn


# ════════════════════════════════════════════════════════
# LaTeX table generation
# ════════════════════════════════════════════════════════

def generate_latex_table(tables: dict[str, dict[str, dict]], task: str, metric: str = "accuracy") -> str:
    """Generate LaTeX table code for the paper."""
    mat = tables[task][metric]
    nc = "3" if task == "3class" else "7"
    label = "Accuracy (\\%)" if metric == "accuracy" else "Macro-F1"
    caption = f"{nc}-class classification {label.lower()} across models and resolutions"

    lines = []
    lines.append(f"% {nc}-class {metric} table")
    lines.append("\\begin{table}[tb]")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{tab:{task}_{metric}}}")
    lines.append("  \\begin{tabular}{llccc}")
    lines.append("    \\toprule")
    lines.append(f"    Paradigm & Model & 224 & 512 & 1024 \\\\")
    lines.append("    \\midrule")

    prev_paradigm = ""
    for m in MODEL_PRIORITY:
        paradigm = PARADIGM_DISPLAY[MODELS[m]["paradigm"]]
        p_label = paradigm if paradigm != prev_paradigm else ""
        prev_paradigm = paradigm

        cells = []
        row_vals = []
        for r in [224, 512, 1024]:
            v = mat[m].get(r)
            if v is None:
                cells.append("--")
                row_vals.append(-1)
            else:
                cells.append(f"{v*100:.1f}")
                row_vals.append(v)

        # Bold the best in this row
        best_v = max(row_vals)
        formatted = []
        for c, v in zip(cells, row_vals):
            if v == best_v and c != "--":
                formatted.append(f"\\textbf{{{c}}}")
            else:
                formatted.append(c)

        lines.append(f"    {p_label} & {MODEL_DISPLAY[m]} & "
                      f"{formatted[0]} & {formatted[1]} & {formatted[2]} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════
# Figures (matplotlib)
# ════════════════════════════════════════════════════════

def generate_figures(tables: dict[str, dict[str, dict]], results: dict[str, dict[str, Any]]) -> None:
    """Generate matplotlib figures for the paper."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not available, skipping figures")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Figure 1: Resolution-Accuracy Curves (3-class) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {
        "efficientnet_b0": "#2196F3",
        "resnet50": "#4CAF50",
        "convnextv2_tiny": "#FF9800",
        "vit_base_16": "#E91E63",
        "swinv2_tiny": "#9C27B0",
        "dinov2_vits14": "#795548",
    }
    markers = {
        "efficientnet_b0": "o",
        "resnet50": "s",
        "convnextv2_tiny": "D",
        "vit_base_16": "^",
        "swinv2_tiny": "v",
        "dinov2_vits14": "*",
    }

    for task_idx, (task, ax) in enumerate(zip(["3class", "fine"], [ax1, ax2])):
        mat = tables[task]["accuracy"]
        for m in MODEL_PRIORITY:
            xs, ys = [], []
            for r in RESOLUTION_PRIORITY:
                v = mat[m].get(r)
                if v is not None:
                    xs.append(r)
                    ys.append(v * 100)
            if xs:
                ax.plot(xs, ys, color=colors[m], marker=markers[m],
                        label=MODEL_DISPLAY[m], linewidth=2, markersize=8)

        nc = "3" if task == "3class" else "7"
        ax.set_title(f"{nc}-Class Classification", fontsize=14)
        ax.set_xlabel("Resolution (px)", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_xticks(RESOLUTION_PRIORITY)
        ax.set_xticklabels(["224", "512", "1024"])
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        if task == "3class":
            ax.set_ylim(75, 100)
        else:
            ax.set_ylim(50, 90)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "resolution_accuracy.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "resolution_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] resolution_accuracy.pdf/png")

    # ── Figure 2: Per-Model Paradigm Comparison Bar Chart ──
    # Model order grouped by paradigm: CNN → ViT → SSL
    PLOT_ORDER = [
        "efficientnet_b0",   # CNN
        "resnet50",           # CNN
        "convnextv2_tiny",    # CNN (modern)
        "vit_base_16",        # ViT
        "swinv2_tiny",        # ViT (hierarchical)
        "dinov2_vits14",      # SSL
    ]
    SHORT_NAMES = {
        "efficientnet_b0":  "Effic.\nB0",
        "resnet50":         "ResNet\n50",
        "convnextv2_tiny":  "ConvNeXt\nV2",
        "vit_base_16":      "ViT-B\n/16",
        "swinv2_tiny":      "Swin V2\nTiny",
        "dinov2_vits14":    "DINOv2\nViT-S",
    }
    RES_COLORS  = ["#4A90D9", "#E67E22", "#27AE60"]   # 224 / 512 / 1024
    RES_LABELS  = ["224", "512", "1024"]
    BAR_W       = 0.22
    PARADIGM_SPANS = [
        ("CNN",  -0.5,  2.5, "#E3F2FD"),   # light blue
        ("ViT",   2.5,  4.5, "#FCE4EC"),   # light pink
        ("SSL",   4.5,  5.5, "#F3E5F5"),   # light purple
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for task, ax in zip(["3class", "fine"], [ax1, ax2]):
        # Draw paradigm background spans
        for pname, xlo, xhi, pcolor in PARADIGM_SPANS:
            ax.axvspan(xlo, xhi, color=pcolor, alpha=0.55, zorder=0)
            ax.text((xlo + xhi) / 2, 101.5 if task == "3class" else 85,
                    pname, ha="center", va="bottom", fontsize=9,
                    color="#555555", fontstyle="italic")

        best_val = -1
        best_pos = None

        for i, m in enumerate(PLOT_ORDER):
            for j, (res, rc) in enumerate(zip(RESOLUTION_PRIORITY, RES_COLORS)):
                rid = f"{m}_{res}_{task}"
                if rid not in results:
                    continue
                val = results[rid]["test_accuracy"] * 100
                x   = i + (j - 1) * BAR_W
                bar = ax.bar(x, val, width=BAR_W * 0.92, color=rc,
                             edgecolor="white", linewidth=0.4, zorder=3)
                if val > best_val:
                    best_val = val
                    best_pos = (x, val)

        # Annotate global best
        if best_pos is not None:
            ax.text(best_pos[0], best_pos[1] + 0.6,
                    f"{best_val:.1f}%", ha="center", fontsize=8.5,
                    fontweight="bold", color="#C0392B", zorder=5)

        nc = "3" if task == "3class" else "7"
        ax.set_title(f"({('a' if task == '3class' else 'b')}) {nc}-Class Accuracy",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_xticks(range(len(PLOT_ORDER)))
        ax.set_xticklabels([SHORT_NAMES[m] for m in PLOT_ORDER], fontsize=9)
        ax.set_xlim(-0.55, len(PLOT_ORDER) - 0.45)
        if task == "3class":
            ax.set_ylim(60, 103)
        else:
            ax.set_ylim(40, 88)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    # Shared legend (resolution bars)
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=c, label=f"{r}px")
        for c, r in zip(RES_COLORS, RES_LABELS)
    ]
    ax1.legend(handles=legend_patches, title="Resolution",
               fontsize=9, title_fontsize=9, loc="upper right",
               framealpha=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "paradigm_comparison.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "paradigm_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] paradigm_comparison.pdf/png")

    # ── Figure 3: Model-wise accuracy heatmap ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for task_idx, (task, ax) in enumerate(zip(["3class", "fine"], [ax1, ax2])):
        mat = tables[task]["accuracy"]
        data = []
        ylabels = []
        for m in MODEL_PRIORITY:
            row = []
            for r in RESOLUTION_PRIORITY:
                v = mat[m].get(r)
                row.append(v * 100 if v else 0)
            data.append(row)
            ylabels.append(MODEL_DISPLAY[m])

        data = np.array(data)
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                       vmin=data[data > 0].min() - 5 if data[data > 0].size > 0 else 50,
                       vmax=data.max())

        ax.set_xticks(range(3))
        ax.set_xticklabels(["224", "512", "1024"])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=9)

        for i in range(len(ylabels)):
            for j in range(3):
                v = data[i, j]
                if v > 0:
                    color = "white" if v > (data.max() + data[data>0].min()) / 2 else "black"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=10, color=color, fontweight="bold")

        nc = "3" if task == "3class" else "7"
        ax.set_title(f"{nc}-Class Accuracy (%)", fontsize=13)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_heatmap.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] accuracy_heatmap.pdf/png")


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    print("Loading results...")
    results = load_all_results()
    print(f"  Loaded {len(results)} completed runs")

    with open(os.path.join(DATA_DIR, "stats.json")) as f:
        stats = json.load(f)

    tables = build_tables(results)

    # Print tables
    for task in ["3class", "fine"]:
        print_table(tables, task, "accuracy")
        print_table(tables, task, "f1")
        resolution_analysis(tables, task)
        paradigm_analysis(results, task)

    # Per-class for best models
    for task in ["3class", "fine"]:
        mat = tables[task]["accuracy"]
        best_acc, best_m, best_r = 0, "", 0
        for m in MODEL_PRIORITY:
            for r in RESOLUTION_PRIORITY:
                v = mat[m].get(r)
                if v and v > best_acc:
                    best_acc, best_m, best_r = v, m, r
        print_per_class(results, best_m, best_r, task)

    # Generate paper_numbers.json
    pn = generate_paper_numbers(results, tables, stats)
    pn_path = os.path.join(RESULTS_DIR, "paper_numbers.json")
    with open(pn_path, "w") as f:
        json.dump(pn, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] paper_numbers.json ({len(pn)} keys)")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(tables, results)

    # LaTeX tables
    if args.latex:
        print("\n" + "="*70)
        print("  LaTeX Tables")
        print("="*70)
        for task in ["3class", "fine"]:
            print(generate_latex_table(tables, task, "accuracy"))
            print()

    # Summary
    print(f"\n{'='*70}")
    print(f"  Analysis complete!")
    print(f"  {len(results)}/{len(MODEL_PRIORITY) * len(RESOLUTION_PRIORITY) * 2} runs")
    missing = []
    for m in MODEL_PRIORITY:
        for r in RESOLUTION_PRIORITY:
            for t in ["3class", "fine"]:
                rid = f"{m}_{r}_{t}"
                if rid not in results:
                    missing.append(rid)
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
