"""
Run_enhance.py
=========================================================
Script to enhance low-light images using trained model
=========================================================

Usage:
    python Run_enhance.py --input_dir "D:/project/input"
    python Run_enhance.py --input_dir "path/to/low" --output_dir "path/to/enhanced"
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Image IO helpers
# ----------------------------
def pil_to_tensor_rgb01(p: Image.Image) -> torch.Tensor:
    if p.mode != "RGB":
        p = p.convert("RGB")
    arr = np.asarray(p).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def tensor_to_pil_rgb01(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ----------------------------
# Color space: RGB <-> HSV
# ----------------------------
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    max_rgb, _ = torch.max(rgb, dim=1, keepdim=True)
    min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)
    diff = max_rgb - min_rgb + 1e-8
    V = max_rgb
    S = torch.where(max_rgb > 1e-8, diff / (max_rgb + 1e-8), torch.zeros_like(max_rgb))
    H = torch.zeros_like(max_rgb)
    mask_r = (max_rgb == r) & (diff > 1e-8)
    H = torch.where(mask_r, (g - b) / diff, H)
    mask_g = (max_rgb == g) & (diff > 1e-8)
    H = torch.where(mask_g, 2.0 + (b - r) / diff, H)
    mask_b = (max_rgb == b) & (diff > 1e-8)
    H = torch.where(mask_b, 4.0 + (r - g) / diff, H)
    H = H / 6.0
    H = torch.where(H < 0, H + 1.0, H)
    return torch.cat([H, S, V], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    H = H * 6.0
    C = V * S
    X = C * (1 - torch.abs(torch.fmod(H, 2.0) - 1))
    m = V - C
    zeros = torch.zeros_like(C)
    r0, g0, b0 = C, X, zeros
    r1, g1, b1 = X, C, zeros
    r2, g2, b2 = zeros, C, X
    r3, g3, b3 = zeros, X, C
    r4, g4, b4 = X, zeros, C
    r5, g5, b5 = C, zeros, X
    H_float = H / 6.0 * 6.0
    r = torch.where(H_float < 1, r0, torch.where(H_float < 2, r1, torch.where(H_float < 3, r2, torch.where(H_float < 4, r3, torch.where(H_float < 5, r4, r5)))))
    g = torch.where(H_float < 1, g0, torch.where(H_float < 2, g1, torch.where(H_float < 3, g2, torch.where(H_float < 4, g3, torch.where(H_float < 5, g4, g5)))))
    b = torch.where(H_float < 1, b0, torch.where(H_float < 2, b1, torch.where(H_float < 3, b2, torch.where(H_float < 4, b3, torch.where(H_float < 5, b4, b5)))))
    R = (r + m).clamp(0, 1)
    G = (g + m).clamp(0, 1)
    B = (b + m).clamp(0, 1)
    return torch.cat([R, G, B], dim=1)


# ----------------------------
# Fuzzy components
# ----------------------------
def sigmoid_uv(x, u, v, eps=1e-8):
    x = x.clamp(eps, 1.0 - eps)
    ratio = ((1.0 - u) * (1.0 - x)) / (u * x)
    return 1.0 / (1.0 + (ratio + eps) ** v)


def inv_sigmoid_uv(mu, u, v, eps=1e-6):
    mu = mu.clamp(eps, 1.0 - eps)
    inv_ratio = torch.pow((1.0 - mu) / (mu + eps), 1.0 / v)
    return 1.0 / (1.0 + (u / (1.0 - u)) * inv_ratio)


def int_operator(mu, r=1):
    out = mu
    for _ in range(r):
        out = torch.where(out <= 0.5, 2.0 * out * out, 1.0 - 2.0 * (1.0 - out) * (1.0 - out))
    return out.clamp(0, 1)


@torch.no_grad()
def fuzzy_candidate_hsv(ilow_rgb, u, v, r_int, eps):
    hsv = rgb_to_hsv(ilow_rgb)
    H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    mu = sigmoid_uv(V, u, v, eps)
    mu2 = int_operator(mu, r=r_int)
    V2 = inv_sigmoid_uv(mu2, u, v, eps).clamp(0, 1)
    out_hsv = torch.cat([H, S, V2], dim=1)
    return hsv_to_rgb(out_hsv).clamp(0, 1)


def _apply_v_only(ilow_rgb, v_fn):
    hsv = rgb_to_hsv(ilow_rgb)
    H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    V2 = v_fn(V).clamp(0, 1)
    out_hsv = torch.cat([H, S, V2], dim=1)
    return hsv_to_rgb(out_hsv).clamp(0, 1)


def _generate_candidates(ilow, bank_u, bank_v, gamma_list, scale_list, bias_list, include_identity, y_only, r_int):
    cand = []
    if include_identity:
        cand.append(ilow)
    x = ilow.clamp(0, 1)
    for gamma in gamma_list:
        if y_only:
            cand.append(_apply_v_only(x, lambda V: torch.pow(V.clamp(1e-6, 1), float(gamma))))
        else:
            cand.append(torch.pow(x, gamma).clamp(0, 1))
    for s in scale_list:
        if y_only:
            cand.append(_apply_v_only(x, lambda V: V * float(s)))
        else:
            cand.append((x * float(s)).clamp(0, 1))
    for b in bias_list:
        if y_only:
            cand.append(_apply_v_only(x, lambda V: V + float(b)))
        else:
            cand.append((x + float(b)).clamp(0, 1))
    for u, v in zip(bank_u, bank_v):
        cand.append(fuzzy_candidate_hsv(ilow, u, v, r_int=r_int, eps=1e-6))
    return cand


# ----------------------------
# Neural Network Models
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class QSelectorCNN(nn.Module):
    def __init__(self, in_ch=9, use_attention=True, base_ch=48):
        super().__init__()
        self.use_attention = use_attention
        ch1, ch2, ch3, ch4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 4
        self.conv1 = nn.Conv2d(in_ch, ch1, 3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, ch1)
        if use_attention:
            self.ca1 = ChannelAttention(ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, ch2)
        if use_attention:
            self.ca2 = ChannelAttention(ch2)
        self.conv3 = nn.Conv2d(ch2, ch3, 3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, ch3)
        if use_attention:
            self.ca3 = ChannelAttention(ch3)
        self.conv4 = nn.Conv2d(ch3, ch4, 3, stride=2, padding=1)
        self.gn4 = nn.GroupNorm(8, ch4)
        self.fc1 = nn.Linear(ch4, ch2)
        self.fc2 = nn.Linear(ch2, 1)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        if self.use_attention:
            x = self.ca1(x)
        x = F.relu(self.gn2(self.conv2(x)))
        if self.use_attention:
            x = self.ca2(x)
        x = F.relu(self.gn3(self.conv3(x)))
        if self.use_attention:
            x = self.ca3(x)
        x = F.relu(self.gn4(self.conv4(x)))
        x = x.mean(dim=(2, 3))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RefineNet(nn.Module):
    def __init__(self, in_ch=9, base_ch=32):
        super().__init__()
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.e1 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.b = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.c1 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x):
        x0 = F.relu(self.init_conv(x))
        e1 = self.e1(x0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.b(e3)
        u2 = self._up_to(self.up2(b), e2)
        d2 = self.c2(torch.cat([u2, e2], dim=1))
        u1 = self._up_to(self.up1(d2), e1)
        d1 = self.c1(torch.cat([u1, e1], dim=1))
        return torch.tanh(self.out(d1)) * 0.7

    @staticmethod
    def _up_to(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class RefineNetLegacy(nn.Module):
    def __init__(self, in_ch=9, base_ch=48):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.b = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.c1 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.b(e3)
        u2 = self._up_to(self.up2(b), e2)
        d2 = self.c2(torch.cat([u2, e2], dim=1))
        u1 = self._up_to(self.up1(d2), e1)
        d1 = self.c1(torch.cat([u1, e1], dim=1))
        return torch.tanh(self.out(d1)) * 0.7

    @staticmethod
    def _up_to(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def detect_model_architecture(state_dict):
    has_attention = "ca1.fc.0.weight" in state_dict
    conv1_shape = state_dict.get("conv1.weight", None)
    base_ch = conv1_shape.shape[0] if conv1_shape is not None else 48
    return has_attention, base_ch


def detect_refiner_architecture(state_dict):
    has_init_conv = "init_conv.weight" in state_dict
    e1_shape = state_dict.get("e1.0.weight", None)
    base_ch = e1_shape.shape[0] if e1_shape is not None else 32
    return has_init_conv, base_ch



# ----------------------------
# Path helper
# ----------------------------
def resolve_weights_path(weights_path="best.pt"):
    """Resolve model weights.

    Priority:
    1) The user-provided path.
    2) best.pt located in the same folder as this script.
    """
    if weights_path is None or str(weights_path).strip() == "":
        weights_path = "best.pt"

    weights_path = os.path.expanduser(str(weights_path))

    if os.path.isfile(weights_path):
        return weights_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, os.path.basename(weights_path))

    if os.path.isfile(local_path):
        return local_path

    raise FileNotFoundError(
        f"Model weights not found: {weights_path}\n"
        f"Please place best.pt in the same folder as Run_enhance.py or pass --weights PATH."
    )

# ----------------------------
# Main Enhancement Function
# ----------------------------
@torch.no_grad()
def enhance_images(weights_path="best.pt", input_dir="input", output_dir="output", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("  🌟 LOW-LIGHT IMAGE ENHANCEMENT")
    print(f"{'='*60}")
    print(f"  📁 Weights:  {weights_path}")
    print(f"  📂 Input:    {input_dir}")
    print(f"  📂 Output:   {output_dir}")
    print(f"  💻 Device:   {device}")
    print(f"{'='*60}\n")
    
    # Load checkpoint from best.pt by default
    weights_path = resolve_weights_path(weights_path)
    print(f"  ✅ Resolved weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    
    bank_u = ckpt.get("bank_u", [0.5] * 9)
    bank_v = ckpt.get("bank_v", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    gamma_list = ckpt.get("gamma_list", [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80])
    scale_list = ckpt.get("scale_list", [1.05, 1.10, 1.15, 1.25, 1.40])
    bias_list = ckpt.get("bias_list", [0.02, 0.04, 0.06, 0.10])
    include_identity = ckpt.get("include_identity", True)
    y_only = ckpt.get("y_only", True)
    r_int = ckpt.get("r_int", 1)
    T = ckpt.get("T", 0.7)
    
    has_attention, base_ch = detect_model_architecture(ckpt["model_state"])
    model = QSelectorCNN(in_ch=9, use_attention=has_attention, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    use_refiner = ckpt.get("use_refiner", False)
    refiner = None
    if use_refiner and ckpt.get("refiner_state") is not None:
        has_init_conv, refiner_base_ch = detect_refiner_architecture(ckpt["refiner_state"])
        if has_init_conv:
            refiner = RefineNet(in_ch=9, base_ch=refiner_base_ch).to(device)
        else:
            refiner = RefineNetLegacy(in_ch=9, base_ch=refiner_base_ch).to(device)
        refiner.load_state_dict(ckpt["refiner_state"])
        refiner.eval()
        print("  ✅ RefineNet loaded")
    
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in exts]
    files.sort()
    
    print(f"  📷 Found {len(files)} images to process\n")
    
    pbar = tqdm(files, desc="[ENHANCING]", leave=True)
    
    for fn in pbar:
        p = Image.open(os.path.join(input_dir, fn)).convert("RGB")
        ilow = pil_to_tensor_rgb01(p).unsqueeze(0).to(device)
        
        cand = _generate_candidates(ilow, bank_u, bank_v, gamma_list, scale_list, bias_list, include_identity, y_only, r_int)
        
        scores = []
        for ck in cand:
            xk = torch.cat([ilow, ck, (ck - ilow).abs()], dim=1)
            scores.append(model(xk))
        scores = torch.cat(scores, dim=1)
        
        w = F.softmax(scores / T, dim=1)
        
        out_fused = 0.0
        for k in range(len(cand)):
            wk = w[:, k:k+1].unsqueeze(-1).unsqueeze(-1)
            out_fused = out_fused + wk * cand[k]
        out_fused = out_fused.clamp(0, 1)
        
        if refiner is not None:
            xref = torch.cat([ilow, out_fused, (out_fused - ilow).abs()], dim=1)
            out = (out_fused + refiner(xref)).clamp(0, 1)
        else:
            out = out_fused
        
        out_img = tensor_to_pil_rgb01(out.squeeze(0))
        out_img.save(os.path.join(output_dir, fn))
    
    print(f"\n{'='*60}")
    print(f"  ✅ Enhancement complete!")
    print(f"  📷 Processed: {len(files)} images")
    print(f"  📂 Saved to:  {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Enhance low-light images")
    parser.add_argument("--input_dir", type=str, default="input", help="Directory containing low-light images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save enhanced images")
    parser.add_argument("--weights", type=str, default="best.pt", help="Model weights file. Default: best.pt in the same folder as this script")
    args = parser.parse_args()

    enhance_images(weights_path=args.weights, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
