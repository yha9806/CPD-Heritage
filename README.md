# CPD-Heritage

**Chinese Painting Dataset — Heritage Classification Benchmark**

Experiment code for **CPSR (Chinese Painting Semantic Resource)**, published in *ACM Journal on Computing and Cultural Heritage* (JOCCH), Visual Heritage 2026 Special Issue.

Paper: [Chinese Painting Semantic Resource for Resolution-Aware Digitisation and Structured Heritage Access](https://doi.org/10.1145/3842674) (DOI: 10.1145/3842674)

> **Note:** This repository is named `CPD-Heritage` for historical reasons; the resource is called CPSR in the published paper.

---

## Overview

CPD-Heritage evaluates six vision models (CNN and ViT families) on:
- **3-class task**: Gongbi / Xieyi / Ink (technique family)
- **7-class fine-grained task**: 7 technique × subject combinations

Models evaluated at three resolutions: 224 × 224, 512 × 512, 1024 × 1024 px.

| Model | Paradigm | 3-class Best | 7-class Best |
|-------|----------|-------------|-------------|
| EfficientNet-B0 | Sup-CNN | 96.1% (512) | 76.7% (512) |
| ResNet-50 | Sup-CNN | 94.4% (512) | 75.0% (512) |
| ConvNeXt V2-Tiny | Modern-CNN | **96.7% (512)** | 79.4% (1024) |
| DINOv2 ViT-S/14 | Self-Supervised | 95.6% (224) | 69.4% (1024) |
| ViT-B/16 | Sup-ViT | 94.4% (224) | 78.9% (512) |
| Swin V2-Tiny | Hier-ViT | 96.1% (512) | **81.7% (1024)** |

---

## Repository Structure

```
CPD-Heritage/
├── experiments/
│   ├── config.py              # Model/training configuration
│   ├── 00_preprocess.py       # Image preprocessing pipeline
│   ├── 01_prepare_data.py     # Train/val/test split generation
│   ├── 02_train.py            # Main training loop
│   ├── 03_visualize.py        # Grad-CAM visualisation
│   ├── 04_analyze.py          # Result aggregation & tables
│   ├── 05_eval_checkpoint.py  # Checkpoint evaluation
│   ├── make_cpd_statistics_figure.py
│   ├── make_gradcam_figure.py
│   ├── run_all.sh             # Full experiment runner
│   ├── data/                  # Split CSVs + class weights
│   └── results/               # Per-model JSON evaluation logs
└── requirements.txt
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Image Data

Set the `CPD_DATA_ROOT` environment variable to point to your image directory:

```bash
export CPD_DATA_ROOT=/path/to/chinese_paintings
```

The image directory should contain subdirectories organised as:
```
chinese_paintings/
├── 工笔/
├── 写意/
├── 水墨/
...
```

### 3. Reproduce Experiments

```bash
cd experiments/

# Step 1: Preprocess images
python 00_preprocess.py

# Step 2: Generate train/val/test splits
python 01_prepare_data.py

# Step 3: Train all models (36 runs: 6 models × 3 resolutions × 2 tasks)
# Requires GPU; ~45 GPU-hours on RTX 2070
bash run_all.sh

# Step 4: Aggregate results
python 04_analyze.py
```

Pre-computed results are in `experiments/results/` (36 JSON files, 152 KB total).
Checkpoints (~8.6 GB) are not included.

---

## Data

`experiments/data/` contains:
- `train.csv`, `val.csv`, `test.csv` — 80/10/10 split indices
- `class_weights.json` — per-class weighting for imbalanced training
- `stats.json` — dataset statistics

Dataset images are not redistributed here due to copyright. See paper for details.

---

## Citation

```bibtex
@article{yu2026cpsr,
  title   = {Chinese Painting Semantic Resource for Resolution-Aware Digitisation and Structured Heritage Access},
  author  = {Yu, Haorui and Xu, Jiao and Yang, Tingting and Yang, Ziyue and Yi, Qiufeng},
  journal = {Journal on Computing and Cultural Heritage},
  publisher = {Association for Computing Machinery},
  year    = {2026},
  month   = aug,
  issn    = {1556-4711},
  doi     = {10.1145/3842674},
  url     = {https://doi.org/10.1145/3842674}
}
```

---

## License

- Code: MIT License (see `LICENSE`)
- Data splits: CC BY-NC-SA 4.0
