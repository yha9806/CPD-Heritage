#!/usr/bin/env python3
"""
00_preprocess.py — Pre-resize images to 224/512/1024 and save as .pt tensors.
One-time cost, eliminates per-epoch resize bottleneck.

Output: /tmp/cpd_tensors/{resolution}/{filename}.pt
"""
import os, sys, time
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_ROOT, DATA_DIR, RESOLUTIONS, IMAGENET_MEAN, IMAGENET_STD

TENSOR_ROOT = "/tmp/cpd_tensors"

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    m = max(w, h)
    p = Image.new("RGB", (m, m), (255, 255, 255))
    p.paste(img, ((m - w) // 2, (m - h) // 2))
    return p

def main() -> None:
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    filenames = sorted(meta["filename"].tolist())
    print(f"Images: {len(filenames)}", flush=True)

    for res in RESOLUTIONS:
        out_dir = os.path.join(TENSOR_ROOT, str(res))
        os.makedirs(out_dir, exist_ok=True)

        # Check existing
        existing = set(os.listdir(out_dir))
        todo = [f for f in filenames if f.replace(".jpg", ".pt") not in existing]

        if not todo:
            print(f"  {res}px: all {len(filenames)} already done", flush=True)
            continue

        print(f"  {res}px: {len(todo)} to process...", flush=True)
        t0 = time.time()

        for i, fname in enumerate(todo):
            path = os.path.join(DATA_ROOT, fname)
            try:
                img = Image.open(path).convert("RGB")
                img = pad_to_square(img)
                img = img.resize((res, res), Image.LANCZOS)
                tensor = normalize(img)
                out_path = os.path.join(out_dir, fname.replace(".jpg", ".pt"))
                torch.save(tensor, out_path)
            except Exception as e:
                print(f"    [WARN] {fname}: {e}", flush=True)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(todo) - i - 1) / rate
                print(f"    {i+1}/{len(todo)} [{elapsed:.0f}s, ~{eta:.0f}s left]", flush=True)

        elapsed = time.time() - t0
        print(f"  {res}px: done in {elapsed:.0f}s", flush=True)

    print("All resolutions preprocessed.", flush=True)

if __name__ == "__main__":
    main()
