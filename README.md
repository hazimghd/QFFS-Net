# ✨ QFFS-Net
## Quality-Guided Fuzzy Sigmoid Selection Network for Low-Light Image Enhancement

### Authors
- Shahad J. Shahbaz
- Hazim G. Daway
- Ahlam M. Kadhim

### Submitted to
Journal of Electrical and Computer Engineering

---

# Overview

QFFS-Net is a hybrid fuzzy deep-learning framework for low-light image enhancement using quality-guided candidate fusion and perceptual CNN selection.

The framework combines:
- Fuzzy sigmoid enhancement
- CNN-based quality selection
- Candidate fusion
- RefineNet enhancement
- Lab / YCbCr luminance processing

---

# Features

- Low-light image enhancement
- Fuzzy logic enhancement
- CNN quality selector
- Softmax fusion
- Lightweight architecture
- SSIM-oriented optimization
- PyTorch implementation

---

# Datasets

- LOL-v1
- LOLv2-Synthetic
- LIME
- DICM

---

# Metrics

- PSNR
- SSIM
- NIQE
- PIQE

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python fuzzy_qssnet.py --mode train
```

---

# Inference

```bash
python Run_enhance.py
```

---

# Model

Pretrained model:
- best.pt

---

# Suggested Repository Visibility

Before paper acceptance:
- Private Repository

After acceptance:
- Public Repository

