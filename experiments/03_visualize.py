#!/usr/bin/env python3
"""
03_visualize.py — Grad-CAM / Attention visualization for CPD benchmark.
Generates heatmap overlays for representative paintings across models and resolutions.

Usage:
  python 03_visualize.py                       # all models, default images
  python 03_visualize.py --model convnextv2_tiny --res 512 --task 3class
  python 03_visualize.py --select              # interactively select images
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

TENSOR_ROOT = "/tmp/cpd_tensors"


def log(msg):
    print(msg, flush=True)


# ════════════════════════════════════════════════════════
# Model loading (reuse from 02_train.py)
# ════════════════════════════════════════════════════════

def load_model(model_name, num_classes, resolution, ckpt_path):
    """Load a trained model from checkpoint."""
    cfg = MODELS[model_name]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["source"] == "timm":
        import timm
        kw = {"pretrained": False, "num_classes": num_classes}
        if "vit" in cfg["timm_name"] and resolution != 224:
            kw["img_size"] = resolution
        if "swin" in cfg["timm_name"]:
            swin_res = 256 if resolution == 224 else resolution
            if swin_res != 256:
                kw["img_size"] = swin_res
        model = timm.create_model(cfg["timm_name"], **kw)

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
        model = LP()

    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    model.to(dev)
    model.eval()
    return model, dev


# ════════════════════════════════════════════════════════
# Target layer selection
# ════════════════════════════════════════════════════════

def get_target_layer(model, model_name):
    """Get the appropriate target layer for Grad-CAM."""
    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "efficientnet_b0":
        return model.conv_head  # last conv before pooling
    elif model_name == "convnextv2_tiny":
        return model.stages[-1].blocks[-1]
    elif model_name == "vit_base_16":
        return model.blocks[-1].norm1
    elif model_name == "swinv2_tiny":
        return model.layers[-1].blocks[-1].norm1
    else:
        raise ValueError(f"No target layer defined for {model_name}")


def reshape_transform_vit(tensor, height=14, width=14):
    """Reshape ViT attention output for Grad-CAM."""
    # tensor: [B, N, D] -> [B, D, H, W]  (skip CLS token)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def reshape_transform_swin(tensor, height, width):
    """Reshape Swin V2 output for Grad-CAM (no CLS token to skip)."""
    # tensor: [B, H*W, D] -> [B, D, H, W]
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


# ════════════════════════════════════════════════════════
# Grad-CAM implementation (simple hook-based)
# ════════════════════════════════════════════════════════

class SimpleGradCAM:
    """Lightweight Grad-CAM without pytorch_grad_cam dependency."""

    def __init__(self, model, target_layer, reshape_fn=None):
        self.model = model
        self.target_layer = target_layer
        self.reshape_fn = reshape_fn
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # One-hot for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        act = self.activations
        grad = self.gradients

        if self.reshape_fn:
            act = self.reshape_fn(act)
            grad = self.reshape_fn(grad)

        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)

        # Normalize
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), output


# ════════════════════════════════════════════════════════
# DINOv2 attention visualization
# ════════════════════════════════════════════════════════

def get_dino_attention(model, input_tensor, dev):
    """Extract attention map from DINOv2's last self-attention layer."""
    model.eval()
    # Hook into the last attention layer
    attn_map = {}

    def hook_fn(module, input, output):
        # For DINOv2, get attention weights
        attn_map["attn"] = output

    backbone = model.backbone
    last_layer = backbone.encoder.layer[-1].attention.attention

    handle = last_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        out = model(input_tensor)

    handle.remove()

    if "attn" not in attn_map:
        return None, out

    # Average across heads, take CLS token attention to patches
    attn = attn_map["attn"]
    # attn shape depends on model implementation
    # For DINOv2: we need to compute attention manually
    return None, out  # Fallback; DINOv2 doesn't expose attn weights easily


def get_dino_attention_v2(model, input_tensor, dev):
    """Alternative: use last_hidden_state patch embeddings as pseudo-attention."""
    model.eval()
    with torch.no_grad():
        outputs = model.backbone(pixel_values=input_tensor, output_attentions=True)
        # outputs.attentions is a tuple of [B, num_heads, N, N]
        if hasattr(outputs, 'attentions') and outputs.attentions:
            attn = outputs.attentions[-1]  # last layer
            # Average over heads, CLS token row
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
            h = w = int(cls_attn.shape[-1] ** 0.5)
            attn_map = cls_attn.reshape(1, h, w).cpu().numpy()[0]
            if attn_map.max() > 0:
                attn_map = attn_map / attn_map.max()
            logits = model(input_tensor)
            return attn_map, logits
    return None, model(input_tensor)


# ════════════════════════════════════════════════════════
# Image preparation
# ════════════════════════════════════════════════════════

def prepare_image(img_path, resolution, dev):
    """Load and preprocess an image for model input."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Pad to square (maintain aspect ratio)
    max_dim = max(orig_w, orig_h)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(img, ((max_dim - orig_w) // 2, (max_dim - orig_h) // 2))

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(padded).unsqueeze(0).to(dev)
    return tensor, img, padded


def overlay_heatmap(original_img, cam, alpha=0.5, colormap="jet"):
    """Overlay Grad-CAM heatmap on original image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Resize CAM to original image size
    cam_resized = np.array(
        Image.fromarray(cam).resize(original_img.size, Image.BILINEAR)
    )

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    orig_array = np.array(original_img)
    blended = (alpha * heatmap + (1 - alpha) * orig_array).astype(np.uint8)

    return Image.fromarray(blended)


# ════════════════════════════════════════════════════════
# Select representative images
# ════════════════════════════════════════════════════════

def select_representative_images(test_csv, n_per_class=2, task="3class"):
    """Select representative images from test set."""
    import pandas as pd
    df = pd.read_csv(test_csv)
    label_col = "category_3" if task == "3class" else "category_fine"

    selected = []
    for cls in sorted(df[label_col].unique()):
        cls_df = df[df[label_col] == cls]
        # Select first n images from each class
        for _, row in cls_df.head(n_per_class).iterrows():
            selected.append({
                "filename": row["filename"],
                "class": cls,
                "path": os.path.join(DATA_ROOT, row["filename"]),
            })
    return selected


# ════════════════════════════════════════════════════════
# Main visualization pipeline
# ════════════════════════════════════════════════════════

def visualize_model(model_name, resolution, task, images, output_dir):
    """Generate Grad-CAM/attention visualizations for one model configuration."""
    cfg = MODELS[model_name]
    nc = 3 if task == "3class" else 7
    run_id = f"{model_name}_{resolution}_{task}"
    ckpt_path = os.path.join(CKPT_DIR, f"{run_id}.pt")

    if not os.path.exists(ckpt_path):
        log(f"  [SKIP] {run_id}: no checkpoint")
        return

    result_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(result_path):
        log(f"  [SKIP] {run_id}: no result")
        return

    with open(result_path) as f:
        result = json.load(f)

    log(f"\n  Visualizing: {run_id} (acc={result['test_accuracy']:.1%})")

    model, dev = load_model(model_name, nc, resolution, ckpt_path)

    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    is_dino = cfg["source"] == "transformers"
    is_vit = "vit" in model_name and not is_dino

    if not is_dino:
        target_layer = get_target_layer(model, model_name)
        reshape_fn = None
        if is_vit or "swin" in model_name:
            # Compute grid size
            if "swin" in model_name:
                # Swin V2: pad 224->256, so effective res is 256 at minimum
                swin_res = 256 if resolution == 224 else resolution
                gs = swin_res // 32
                reshape_fn = lambda t, h=gs, w=gs: reshape_transform_swin(t, h, w)
            elif "vit" in model_name:
                gs = resolution // 16
                reshape_fn = lambda t, h=gs, w=gs: reshape_transform_vit(t, h, w)
            else:
                gs = 7
                reshape_fn = lambda t, h=gs, w=gs: reshape_transform_vit(t, h, w)

        cam_gen = SimpleGradCAM(model, target_layer, reshape_fn)

    # Load label mapping
    import pandas as pd
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    label_col = "category_3" if task == "3class" else "category_fine"
    labels = sorted(meta[label_col].unique())
    idx2label = {i: l for i, l in enumerate(labels)}

    for img_info in images:
        img_path = img_info["path"]
        if not os.path.exists(img_path):
            log(f"    [!] Missing: {img_path}")
            continue

        tensor, orig_img, padded_img = prepare_image(img_path, resolution, dev)

        # Swin V2 at 224 needs padding to 256 (matching 02_train.py)
        if "swin" in model_name and resolution == 224:
            diff = 256 - tensor.shape[-1]
            pad_l = diff // 2
            pad_r = diff - pad_l
            tensor = torch.nn.functional.pad(tensor, (pad_l, pad_r, pad_l, pad_r), value=0)

        if is_dino:
            attn_map, logits = get_dino_attention_v2(model, tensor, dev)
            pred = logits.argmax(1).item()
        else:
            cam_map, logits = cam_gen(tensor)
            pred = logits.argmax(1).item()
            attn_map = cam_map

        pred_label = idx2label[pred]
        true_label = img_info["class"]
        correct = "OK" if pred_label == true_label else "WRONG"

        if attn_map is not None:
            # Convert to uint8 for PIL
            cam_uint8 = (attn_map * 255).astype(np.uint8)
            blended = overlay_heatmap(padded_img, cam_uint8, alpha=0.4)

            fname = img_info["filename"].replace(".jpg", "")
            out_path = os.path.join(run_dir, f"{fname}_{correct}.jpg")
            blended.save(out_path, quality=95)
            log(f"    {fname}: pred={pred_label} true={true_label} [{correct}]")
        else:
            log(f"    {img_info['filename']}: attention extraction failed")

    # Cleanup
    del model
    torch.cuda.empty_cache()


def generate_comparison_figure(images, models_resolutions, task, output_dir):
    """Generate a comparison figure across resolutions for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # For each selected image, show Grad-CAM across resolutions
    for img_info in images[:3]:  # Top 3 images for paper
        fname = img_info["filename"].replace(".jpg", "")
        fig_paths = []

        for model_name, resolutions in models_resolutions:
            for res in resolutions:
                run_id = f"{model_name}_{res}_{task}"
                img_path = os.path.join(output_dir, run_id,
                                        f"{fname}_OK.jpg")
                if os.path.exists(img_path):
                    fig_paths.append((model_name, res, img_path))

        if not fig_paths:
            continue

        n = len(fig_paths)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (m, r, p) in zip(axes, fig_paths):
            img = Image.open(p)
            ax.imshow(img)
            ax.set_title(f"{MODEL_DISPLAY[m]}\n{r}px", fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        out = os.path.join(output_dir, f"comparison_{fname}.pdf")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  [OK] comparison_{fname}.pdf")


MODEL_DISPLAY = {
    "efficientnet_b0": "EfficientNet-B0",
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "convnextv2_tiny": "ConvNeXt V2-Tiny",
    "resnet50": "ResNet-50",
    "vit_base_16": "ViT-B/16",
    "swinv2_tiny": "Swin V2-Tiny",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Single model to visualize")
    parser.add_argument("--res", type=int, help="Single resolution")
    parser.add_argument("--task", type=str, default="3class")
    parser.add_argument("--n-images", type=int, default=3,
                        help="Number of images per class")
    args = parser.parse_args()

    output_dir = os.path.join(FIGURES_DIR, "gradcam")
    os.makedirs(output_dir, exist_ok=True)

    test_csv = os.path.join(DATA_DIR, "test.csv")
    images = select_representative_images(test_csv, args.n_images, args.task)
    log(f"Selected {len(images)} representative images")

    models = [args.model] if args.model else MODEL_PRIORITY
    resolutions = [args.res] if args.res else RESOLUTION_PRIORITY

    for m in models:
        if m not in MODELS:
            continue
        for r in resolutions:
            visualize_model(m, r, args.task, images, output_dir)

    # Generate comparison figure for paper
    log("\nGenerating comparison figures...")
    model_res_pairs = [(m, resolutions) for m in models if m in MODELS]
    generate_comparison_figure(images, model_res_pairs, args.task, output_dir)

    log("\nVisualization complete!")


if __name__ == "__main__":
    main()
