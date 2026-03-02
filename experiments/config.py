"""
CPD Experiment Configuration
All constants and model definitions for the 6-model × 3-resolution benchmark.
"""
import os

# ──── Paths ────
# Set CPD_DATA_ROOT env var to your image directory (e.g. export CPD_DATA_ROOT=/path/to/images)
DATA_ROOT = os.environ.get("CPD_DATA_ROOT", "./data")
EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_ROOT, "data")
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
FIGURES_DIR = os.path.join(EXP_ROOT, "figures")
CKPT_DIR = os.path.join(EXP_ROOT, "checkpoints")

# ──── Reproducibility ────
SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train / val / test

# ──── Category Mappings ────
# Raw technique labels from filenames
TECHNIQUE_MAP = {
    "工笔": "Gongbi",
    "写意": "Xieyi",
    "水墨": "Ink",
    "浅绛": "LightOchre",
    "白描": "Baimiao",
    "没骨": "Mogu",
    "重彩": "HeavyColor",
    "兼工带写": "Mixed",
}

# Raw subject labels
SUBJECT_MAP = {
    "山水": "Landscape",
    "人物": "Figure",
    "花鸟": "FlowerBird",
}

# Fine-grained merging rules (small classes → nearest larger class)
# Applied AFTER combining technique × subject
MERGE_RULES = {
    "Baimiao-Figure": "Gongbi-Figure",        # 白描-人物 → 工笔-人物
    "Baimiao-Landscape": "Ink-Landscape",      # 白描-山水 → 水墨-山水
    "Baimiao-FlowerBird": "Ink-FlowerBird",    # 白描-花鸟 → 水墨-花鸟
    "Mixed-Landscape": "Ink-Landscape",        # 兼工带写 → 水墨-山水
    "Mixed-Figure": "Gongbi-Figure",           # 兼工带写-人物 → 工笔-人物
    "Mixed-FlowerBird": "Ink-FlowerBird",      # 兼工带写-花鸟 → 水墨-花鸟
    "HeavyColor-Landscape": "Gongbi-Landscape", # 重彩-山水 → 工笔-山水 (if tiny)
    "HeavyColor-Figure": "Gongbi-Figure",     # 重彩-人物 → 工笔-人物 (if tiny)
    "HeavyColor-FlowerBird": "Gongbi-FlowerBird",
    "Xieyi-Landscape": "Ink-Landscape",        # 写意-山水 → 水墨-山水
}
MERGE_THRESHOLD = 30  # merge if class has fewer than this many samples

# ──── Exclusion Filters ────
EXCLUDE_KEYWORDS = ["敦煌", "唐卡", "藏传", "壁画"]  # non-Chinese-painting forms
EXCLUDE_PREFIX = "ch_"

# ──── Training ────
RESOLUTIONS = [224, 512, 1024]
EPOCHS_FINETUNE = 50
EPOCHS_LINEAR = 100
PATIENCE = 10
LR_FINETUNE = 1e-4
LR_HEAD = 1e-3
LR_LINEAR = 0.01
WEIGHT_DECAY = 1e-4

# ──── Models ────
MODELS = {
    "resnet50": {
        "source": "timm",
        "timm_name": "resnet50.a1_in1k",
        "paradigm": "sup_cnn",
        "year": 2015,
        "params_m": 25.0,
        "train_mode": "finetune",
        "batch_sizes": {224: 32, 512: 8, 1024: 2},
    },
    "efficientnet_b0": {
        "source": "timm",
        "timm_name": "efficientnet_b0.ra_in1k",
        "paradigm": "sup_cnn",
        "year": 2019,
        "params_m": 5.3,
        "train_mode": "finetune",
        "batch_sizes": {224: 48, 512: 12, 1024: 2},
    },
    "convnextv2_tiny": {
        "source": "timm",
        "timm_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "paradigm": "modern_cnn",
        "year": 2023,
        "params_m": 28.6,
        "train_mode": "finetune",
        "lr_backbone": 5e-6,   # FCMAE pretrained needs very low LR
        "lr_head": 1e-4,
        "batch_sizes": {224: 24, 512: 6, 1024: 1},
    },
    "vit_base_16": {
        "source": "timm",
        "timm_name": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "paradigm": "sup_vit",
        "year": 2020,
        "params_m": 86.0,
        "train_mode": "finetune",
        "lr_backbone": 2e-5,   # ViT needs lower LR than CNNs
        "lr_head": 5e-4,
        "batch_sizes": {224: 24, 512: 4, 1024: 1},
    },
    "swinv2_tiny": {
        "source": "timm",
        "timm_name": "swinv2_tiny_window8_256.ms_in1k",
        "paradigm": "hier_vit",
        "year": 2022,
        "params_m": 28.0,
        "train_mode": "finetune",
        "lr_backbone": 5e-6,   # Swin V2 needs very low LR (like ConvNeXt V2)
        "lr_head": 1e-4,
        "batch_sizes": {224: 24, 512: 6, 1024: 2},
    },
    "dinov2_vits14": {
        "source": "transformers",
        "hf_name": "facebook/dinov2-small",
        "paradigm": "self_supervised",
        "year": 2023,
        "params_m": 22.0,
        "train_mode": "linear",
        "batch_sizes": {224: 64, 512: 16, 1024: 4},
    },
}

# Model execution priority (most important first)
MODEL_PRIORITY = [
    "efficientnet_b0",  # paper's original baseline
    "dinov2_vits14",    # most important new paradigm
    "convnextv2_tiny",  # modern CNN
    "resnet50",         # classic baseline
    "vit_base_16",      # ViT milestone
    "swinv2_tiny",      # hierarchical ViT
]

# Resolution priority (fast → slow)
RESOLUTION_PRIORITY = [224, 512, 1024]

# Task definitions
TASKS = {
    "3class": {"num_classes": 3, "label_col": "category_3"},
    "fine": {"num_classes": None, "label_col": "category_fine"},  # set after data prep
}

# ──── ImageNet normalization ────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
