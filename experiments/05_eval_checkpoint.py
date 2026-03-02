#!/usr/bin/env python3
"""
05_eval_checkpoint.py — Evaluate a saved checkpoint on the test set.
Used to recover results from crashed training runs where the best checkpoint was saved.

Usage:
  python 05_eval_checkpoint.py --model swinv2_tiny --res 1024 --task fine
"""
import os, sys, json, time, argparse
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Use DATA_ROOT from config (set CPD_DATA_ROOT env var or it defaults to ./data)
ORIG_DATA_ROOT = DATA_ROOT

def log(msg: str) -> None:
    print(msg, flush=True)


class CPDEvalDataset(Dataset):
    """Load images directly from disk (no pre-computed tensors needed)."""
    def __init__(self, csv_path: str, label_col: str, label2idx: dict[str, int], resolution: int, pad_to: int | None = None) -> None:
        from torchvision import transforms
        from PIL import Image
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col
        self.label2idx = label2idx
        self.resolution = resolution
        self.Image = Image
        self.pad_to = pad_to
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = os.path.join(ORIG_DATA_ROOT, row["filename"])
        img = self.Image.open(img_path).convert("RGB")
        # Pad to square
        w, h = img.size
        mx = max(w, h)
        padded = self.Image.new("RGB", (mx, mx), (255, 255, 255))
        padded.paste(img, ((mx - w) // 2, (mx - h) // 2))
        tensor = self.transform(padded)
        # Swin V2 @224: pad tensor to 256 (matching 02_train.py)
        if self.pad_to and tensor.shape[-1] < self.pad_to:
            diff = self.pad_to - tensor.shape[-1]
            pad_l = diff // 2
            pad_r = diff - pad_l
            tensor = torch.nn.functional.pad(tensor, (pad_l, pad_r, pad_l, pad_r), value=0)
        return tensor, self.label2idx[row[self.label_col]]


def create_model(model_name: str, num_classes: int, resolution: int) -> nn.Module:
    cfg = MODELS[model_name]
    if cfg["source"] == "timm":
        import timm
        kw = {"pretrained": False, "num_classes": num_classes}
        if "vit" in cfg["timm_name"] and resolution != 224:
            kw["img_size"] = resolution
        if "swin" in cfg["timm_name"]:
            swin_res = 256 if resolution == 224 else resolution
            if swin_res != 256:
                kw["img_size"] = swin_res
        return timm.create_model(cfg["timm_name"], **kw)
    elif cfg["source"] == "transformers":
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(cfg["hf_name"])
        hd = backbone.config.hidden_size
        for p in backbone.parameters():
            p.requires_grad = False
        class LP(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(hd, num_classes)
            def forward(self, x):
                return self.classifier(self.backbone(pixel_values=x).last_hidden_state[:, 0])
        return LP()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--res", type=int, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    model_name = args.model
    resolution = args.res
    task = args.task
    cfg = MODELS[model_name]

    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    label_col = "category_3" if task == "3class" else "category_fine"
    label2idx = {c: i for i, c in enumerate(sorted(meta[label_col].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}
    nc = len(label2idx)

    run_id = f"{model_name}_{resolution}_{task}"
    ckpt_path = os.path.join(CKPT_DIR, f"{run_id}.pt")
    result_path = os.path.join(RESULTS_DIR, f"{run_id}.json")

    log(f"Evaluating {run_id} from checkpoint...")
    log(f"  Classes: {nc}, Checkpoint: {ckpt_path}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = cfg["batch_sizes"].get(resolution, 2)

    # Load model
    model = create_model(model_name, nc, resolution)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    model.to(dev)
    model.eval()
    log(f"  Model loaded on {dev}")

    # Test dataset (reads from original images, no pre-computed tensors needed)
    pad_to = 256 if "swin" in model_name and resolution == 224 else None
    test_ds = CPDEvalDataset(os.path.join(DATA_DIR, "test.csv"), label_col, label2idx, resolution, pad_to)
    test_ld = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    # Evaluate
    t0 = time.time()
    ap, al = [], []
    with torch.no_grad():
        for imgs, labs in test_ld:
            imgs = imgs.to(dev, non_blocking=True)
            with autocast("cuda"):
                out = model(imgs)
            ap.extend(out.argmax(1).cpu().numpy())
            al.extend(labs.numpy())

    ap, al = np.array(ap), np.array(al)
    acc = accuracy_score(al, ap)
    mf1 = f1_score(al, ap, average="macro")
    wf1 = f1_score(al, ap, average="weighted")
    cm = confusion_matrix(al, ap).tolist()
    cn = [idx2label[i] for i in range(nc)]
    rpt = classification_report(al, ap, target_names=cn, output_dict=True)
    eval_time = time.time() - t0

    result = {
        "run_id": run_id, "model": model_name, "resolution": resolution,
        "task": task, "num_classes": nc, "paradigm": cfg["paradigm"],
        "train_mode": cfg["train_mode"], "batch_size": bs,
        "epochs_run": "recovered_from_checkpoint",
        "best_val_acc": None,
        "test_accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
        "weighted_f1": round(wf1, 4), "confusion_matrix": cm,
        "per_class": {c: {
            "precision": round(rpt[c]["precision"], 4),
            "recall": round(rpt[c]["recall"], 4),
            "f1": round(rpt[c]["f1-score"], 4),
            "support": rpt[c]["support"],
        } for c in cn},
        "train_time_sec": None,
        "eval_time_sec": round(eval_time, 1),
        "timestamp": datetime.now().isoformat(),
        "note": "Recovered from best checkpoint (val_acc unavailable)",
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    log(f"\n  >>> {run_id}: acc={acc:.1%}  macro_F1={mf1:.4f}  eval={eval_time:.1f}s")
    log(f"  Saved to {result_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
