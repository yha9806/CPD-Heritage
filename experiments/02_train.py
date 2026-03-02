#!/usr/bin/env python3
"""
02_train.py — CPD classification benchmark.
Reads pre-processed .pt tensors (from 00_preprocess.py) + AMP.

Usage:
  python 02_train.py                          # all 36 runs
  python 02_train.py --model efficientnet_b0  # single model
  python 02_train.py --list                   # status
"""
import os, sys, json, time, argparse, random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

TENSOR_ROOT = "/tmp/cpd_tensors"

def log(msg):
    print(msg, flush=True)


# ════════════════════════════════════════════════════════
# Dataset — reads pre-computed .pt tensors
# ════════════════════════════════════════════════════════

class CPDDataset(Dataset):
    def __init__(self, csv_path, label_col, label2idx, resolution, augment=False, pad_to=None):
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col
        self.label2idx = label2idx
        self.resolution = resolution
        self.augment = augment
        self.tensor_dir = os.path.join(TENSOR_ROOT, str(resolution))
        self.pad_to = pad_to  # e.g. 256 for Swin V2 when resolution=224

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pt_name = row["filename"].replace(".jpg", ".pt")
        tensor = torch.load(os.path.join(self.tensor_dir, pt_name),
                            weights_only=True)

        # Pad to target size if needed (e.g. Swin V2 at 224→256)
        if self.pad_to and tensor.shape[-1] < self.pad_to:
            diff = self.pad_to - tensor.shape[-1]
            pad_l = diff // 2
            pad_r = diff - pad_l
            tensor = torch.nn.functional.pad(tensor, (pad_l, pad_r, pad_l, pad_r), value=0)

        if self.augment:
            # Fast tensor augmentations (no PIL round-trip)
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, [2])  # horizontal flip
            if torch.rand(1).item() < 0.8:
                b = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                tensor = tensor * b
                c = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                m = tensor.mean()
                tensor = (tensor - m) * c + m

        return tensor, self.label2idx[row[self.label_col]]


# ════════════════════════════════════════════════════════
# Model Factory
# ════════════════════════════════════════════════════════

def create_model(model_name, num_classes, resolution):
    cfg = MODELS[model_name]

    if cfg["source"] == "timm":
        import timm
        kw = {"pretrained": True, "num_classes": num_classes}
        if "vit" in cfg["timm_name"] and resolution != 224:
            kw["img_size"] = resolution
        if "swin" in cfg["timm_name"]:
            # Swin V2 (window_size=8) needs resolution divisible by 32
            # At 224, pad input to 256 (native size); at 512/1024, use as-is
            swin_res = 256 if resolution == 224 else resolution
            if swin_res != 256:
                kw["img_size"] = swin_res
        return timm.create_model(cfg["timm_name"], **kw), False

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
                return self.classifier(
                    self.backbone(pixel_values=x).last_hidden_state[:, 0])

        return LP(), True


# ════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════

def train_one_run(model_name, resolution, task, train_path, val_path, test_path,
                  label2idx, class_weights):
    cfg = MODELS[model_name]
    nc = len(label2idx)
    label_col = TASKS[task]["label_col"]
    run_id = f"{model_name}_{resolution}_{task}"
    result_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    ckpt_path = os.path.join(CKPT_DIR, f"{run_id}.pt")

    if os.path.exists(result_path):
        with open(result_path) as f:
            d = json.load(f)
        if d.get("status") != "OOM_FAILED":
            log(f"  [SKIP] {run_id}")
            return d

    log(f"\n{'='*60}")
    log(f"  {run_id} | {nc}cls")
    log(f"{'='*60}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = dev.type == "cuda"
    bs = cfg["batch_sizes"].get(resolution, 8)

    # Swin V2 (window_size=8) needs resolution divisible by 32; pad 224→256
    pad_to = None
    if "swin" in model_name and resolution == 224:
        pad_to = 256

    train_ds = CPDDataset(train_path, label_col, label2idx, resolution, True, pad_to=pad_to)
    val_ds = CPDDataset(val_path, label_col, label2idx, resolution, False, pad_to=pad_to)
    test_ds = CPDDataset(test_path, label_col, label2idx, resolution, False, pad_to=pad_to)

    for oom_try in range(3):
        try:
            torch.cuda.empty_cache()
            # .pt loading is fast, use many workers
            nw = 4
            kw = dict(num_workers=nw, pin_memory=True, persistent_workers=True)
            train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, **kw)
            val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, **kw)
            test_ld = DataLoader(test_ds, batch_size=bs, shuffle=False, **kw)

            model, frozen = create_model(model_name, nc, resolution)
            model.to(dev)

            if resolution >= 1024 and hasattr(model, 'set_grad_checkpointing'):
                # Skip for Swin V2: its checkpoint impl has namespace conflicts
                if "swin" not in model_name:
                    try:
                        model.set_grad_checkpointing(True)
                    except Exception as e:
                        log(f"  [WARN] grad checkpointing failed: {e}")

            idx2label = {v: k for k, v in label2idx.items()}

            if frozen:
                opt = torch.optim.SGD(model.classifier.parameters(), lr=LR_LINEAR,
                                      momentum=0.9, weight_decay=WEIGHT_DECAY)
                epochs = EPOCHS_LINEAR
            else:
                # Identify classifier head via timm API (avoids substring false positives)
                classifier_module = model.get_classifier()
                classifier_param_ids = set(id(p) for p in classifier_module.parameters())
                hp, bp = [], []
                for n, p in model.named_parameters():
                    if id(p) in classifier_param_ids:
                        hp.append(p)
                    else:
                        bp.append(p)
                # Model-specific LR (ConvNeXt V2, ViT need lower)
                lr_bb = cfg.get("lr_backbone", LR_FINETUNE)
                lr_hd = cfg.get("lr_head", LR_HEAD)
                opt = torch.optim.AdamW([
                    {"params": bp, "lr": lr_bb},
                    {"params": hp, "lr": lr_hd},
                ], weight_decay=WEIGHT_DECAY)
                epochs = EPOCHS_FINETUNE
                log(f"  LR: backbone={lr_bb}, head={lr_hd}")

            wt = torch.zeros(nc)
            for i in range(nc):
                wt[i] = class_weights.get(idx2label[i], 1.0)
            crit = nn.CrossEntropyLoss(weight=wt.to(dev))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            scaler = GradScaler("cuda") if amp_on else None

            best_va = 0.0
            pat = 0
            t0 = time.time()

            for ep in range(epochs):
                model.train()
                if frozen:
                    model.backbone.eval()
                tl = tc = tt = 0

                for imgs, labs in train_ld:
                    imgs, labs = imgs.to(dev, non_blocking=True), labs.to(dev, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    if amp_on:
                        with autocast("cuda"):
                            out = model(imgs)
                            loss = crit(out, labs)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        out = model(imgs)
                        loss = crit(out, labs)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        opt.step()
                    tl += loss.item() * imgs.size(0)
                    tc += (out.argmax(1) == labs).sum().item()
                    tt += imgs.size(0)

                sched.step()

                model.eval()
                vc = vt = 0
                with torch.no_grad():
                    for imgs, labs in val_ld:
                        imgs, labs = imgs.to(dev, non_blocking=True), labs.to(dev, non_blocking=True)
                        if amp_on:
                            with autocast("cuda"):
                                out = model(imgs)
                        else:
                            out = model(imgs)
                        vc += (out.argmax(1) == labs).sum().item()
                        vt += imgs.size(0)

                va = vc / vt
                if (ep + 1) % 5 == 0 or ep == 0:
                    log(f"  Ep {ep+1:3d}/{epochs} loss={tl/tt:.4f} "
                        f"trn={tc/tt:.3f} val={va:.3f} [{time.time()-t0:.0f}s]")

                if va > best_va:
                    best_va = va
                    pat = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    pat += 1
                    if pat >= PATIENCE:
                        log(f"  Early stop @ ep {ep+1}")
                        break

            train_time = time.time() - t0

            # Test
            model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
            model.eval()
            ap, al = [], []
            with torch.no_grad():
                for imgs, labs in test_ld:
                    imgs = imgs.to(dev, non_blocking=True)
                    if amp_on:
                        with autocast("cuda"):
                            out = model(imgs)
                    else:
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

            result = {
                "run_id": run_id, "model": model_name, "resolution": resolution,
                "task": task, "num_classes": nc, "paradigm": cfg["paradigm"],
                "train_mode": cfg["train_mode"], "batch_size": bs,
                "epochs_run": ep + 1, "best_val_acc": round(best_va, 4),
                "test_accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                "weighted_f1": round(wf1, 4), "confusion_matrix": cm,
                "per_class": {c: {
                    "precision": round(rpt[c]["precision"], 4),
                    "recall": round(rpt[c]["recall"], 4),
                    "f1": round(rpt[c]["f1-score"], 4),
                    "support": rpt[c]["support"],
                } for c in cn},
                "train_time_sec": round(train_time, 1),
                "timestamp": datetime.now().isoformat(),
            }
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            log(f"\n  >>> {run_id}: acc={acc:.3f} F1={mf1:.3f} {train_time:.0f}s")

            del model, opt, crit, scaler
            torch.cuda.empty_cache()
            return result

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try: del model
            except: pass
            torch.cuda.empty_cache()
            if oom_try < 2:
                old = bs
                bs = max(1, bs // 2)
                log(f"  [OOM] BS {old}->{bs}")
            else:
                log(f"  [FAIL] {run_id}: OOM")
                result = {"run_id": run_id, "model": model_name,
                          "resolution": resolution, "task": task,
                          "status": "OOM_FAILED", "timestamp": datetime.now().isoformat()}
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)
                return result


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def get_label_mappings():
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    return {
        "3class": {c: i for i, c in enumerate(sorted(meta["category_3"].unique()))},
        "fine": {c: i for i, c in enumerate(sorted(meta["category_fine"].unique()))},
    }

def list_runs():
    log(f"{'Run ID':<45} {'Status':<10} {'Acc':<8} {'F1':<8}")
    log("-" * 73)
    for m in MODEL_PRIORITY:
        for r in RESOLUTION_PRIORITY:
            for t in TASKS:
                rid = f"{m}_{r}_{t}"
                rp = os.path.join(RESULTS_DIR, f"{rid}.json")
                if os.path.exists(rp):
                    with open(rp) as f: d = json.load(f)
                    s = d.get("status", "DONE")
                    a = f"{d['test_accuracy']:.3f}" if "test_accuracy" in d else ""
                    fm = f"{d['macro_f1']:.3f}" if "macro_f1" in d else ""
                    log(f"  {rid:<43} {s:<10} {a:<8} {fm:<8}")
                else:
                    log(f"  {rid:<43} {'PENDING':<10}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--res", type=int)
    parser.add_argument("--task", type=str)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ──── Reproducibility ────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.list: return list_runs()

    train_p = os.path.join(DATA_DIR, "train.csv")
    val_p = os.path.join(DATA_DIR, "val.csv")
    test_p = os.path.join(DATA_DIR, "test.csv")

    lm = get_label_mappings()
    with open(os.path.join(DATA_DIR, "class_weights.json")) as f:
        aw = json.load(f)

    models = [args.model] if args.model else MODEL_PRIORITY
    ress = [args.res] if args.res else RESOLUTION_PRIORITY
    tasks = [args.task] if args.task else list(TASKS.keys())

    total = len(models) * len(ress) * len(tasks)
    ok = fail = 0

    log(f"\n{'#'*60}")
    log(f"  CPD Benchmark: {total} runs")
    log(f"  AMP: ON | pre-computed .pt tensors | ext4")
    log(f"{'#'*60}")

    t0 = time.time()
    for m in models:
        if m not in MODELS: continue
        for r in ress:
            for t in tasks:
                wk = "category_3" if t == "3class" else "category_fine"
                res = train_one_run(m, r, t, train_p, val_p, test_p, lm[t], aw[wk])
                if res and res.get("status") != "OOM_FAILED":
                    ok += 1
                else:
                    fail += 1

    log(f"\n{'='*60}")
    log(f"  Done: {ok} ok, {fail} fail, {(time.time()-t0)/3600:.1f}h")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
