# 🎨 Fashion Image Editing with Segmentation & Inpainting

A collection of notebooks and a Gradio GUI for AI-powered fashion image editing. Select any part of an image — clothing, hair, background — and replace it using text prompts via diffusion models.

---

## 📁 Project Structure

```
├── SAM-SD2.ipynb          # SAM + Stable Diffusion 2 inpainting
├── SAM2-SDXL.ipynb        # SAM 2 + Stable Diffusion XL inpainting
├── SegFormer-SDXL.ipynb   # SegFormer (clothing-specific) + SDXL inpainting
└── app.py                 # Gradio web interface (multi-mask support)
```

---

## 🔍 Approach Overview

Each notebook follows the same two-stage pipeline:

```
Input Image → Segmentation Model → Mask(s) → Diffusion Model → Edited Image
```

**Stage 1 — Segmentation** identifies and isolates the region to edit (shirt, hair, background, etc.)

**Stage 2 — Inpainting** fills the masked region with new content guided by a text prompt, while preserving everything outside the mask.

The GUI supports **multiple edits in a single run** — each edit is applied sequentially, with the output of one step feeding into the next:

```
Original Image
    ↓
Edit 1: Hair mask + "blonde hair"        → Intermediate Image
    ↓
Edit 2: Upper-clothes mask + "blue shirt" → Final Image
```

---

## 📓 Notebooks

### SAM-SD2.ipynb

Uses Meta's **Segment Anything Model (ViT-H)** for general-purpose segmentation and **Stable Diffusion 2** for inpainting.

- SAM generates multiple masks automatically across the whole image
- Masks are visualized with numbered labels so you can pick the right region
- SD2 inpaints at 512×512

**Best for:** General editing, non-clothing regions, quick experimentation

---

### SAM2-SDXL.ipynb

Upgrades both models — uses **SAM 2** (Meta, 2024) for segmentation and **Stable Diffusion XL** for higher quality inpainting.

- SAM 2 produces sharper, more accurate mask boundaries than SAM v1
- SDXL runs at 1024×1024 for significantly better output quality
- Lower `pred_iou_thresh` (0.25) to generate more mask candidates

**Best for:** Higher quality outputs, complex scenes, detailed clothing edits

---

### SegFormer-SDXL.ipynb

Replaces SAM with **SegFormer** (`mattmdjaga/segformer_b2_clothes`), a transformer model specifically trained on fashion datasets, combined with **SDXL** inpainting.

- Segments 18 specific clothing categories (shirt, pants, hair, shoes, etc.)
- No manual mask index guessing — select by semantic label (e.g. "Upper-clothes")
- More reliable clothing boundaries than general-purpose SAM

**Best for:** Clothing-specific edits, hair color changes, background replacement

**Segmentation Labels:**
| ID | Label | ID | Label |
|----|-------|----|-------|
| 0 | Background | 9 | Left-shoe |
| 1 | Hat | 10 | Right-shoe |
| 2 | Hair | 11 | Face |
| 3 | Sunglasses | 12 | Left-leg |
| 4 | Upper-clothes | 13 | Right-leg |
| 5 | Skirt | 14 | Left-arm |
| 6 | Pants | 15 | Right-arm |
| 7 | Dress | 16 | Bag |
| 8 | Belt | 17 | Scarf |

---

## 🖥️ GUI (app.py)

A Gradio web interface that wraps the SegFormer + SDXL pipeline into a 3-step UI with support for editing multiple regions in a single run.

### Running the GUI

```bash
python app.py
```

Or in Colab:

```python
!python app.py
```

### Steps

**Step 1 — Upload & Segment**
Upload an image and click "Segment Image". The app displays a color-coded segmentation map and populates a dropdown with only the labels detected in your image.

**Step 2 — Select Items & Add Edits**
Pick a label from the dropdown (e.g. "Upper-clothes", "Hair"). A mask preview updates automatically so you can verify the correct region is selected. Enter a prompt for that region and click **+ Add to Queue**. Repeat for as many regions as you want to edit. The edit queue shows a live summary of all pending edits. Use **🗑 Clear Queue** to start over.

**Step 3 — Generate**
Click **🎨 Generate All Edits**. Each edit is applied sequentially in queue order. A generation log shows ✅/❌ status for each edit so you can see exactly what succeeded.

### Notes on Edit Order

Since edits are applied sequentially, **order matters** — the output of each step feeds into the next. Add edits in the order you want them applied. If regions overlap (e.g. hair overlapping a shirt collar), apply the more important edit last so it has the final say.

---

## ⚙️ Installation

### Common Dependencies

```bash
pip install diffusers transformers accelerate scipy
pip install xformers
pip install gradio
pip install opencv-python
```

### For SAM (SAM-SD2.ipynb)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pycocotools onnxruntime onnx
```

Download the SAM checkpoint:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### For SAM 2 (SAM2-SDXL.ipynb)

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[demo]"
```

Download the SAM 2 checkpoint:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### For SegFormer + GUI

```bash
pip install transformers
```

The SegFormer model downloads automatically from HuggingFace on first run.

---

## 💡 Prompt Tips

- **Lead with the base color** — `"black shirt with red flowers"` works better than `"shirt with red flowers on black"`
- **Specify texture/material** — `"velvet"`, `"denim"`, `"silk"` help the model render realistic fabric
- **Include lighting terms** — `"soft natural lighting"`, `"golden hour"` for better background blending
- **Add a negative prompt** to avoid artifacts:
  ```
  blurry edges, inconsistent lighting, artifacts, distorted, watermark
  ```

---

## 🔄 Model Comparison

| Notebook       | Segmentation | Inpainting  | Quality         | Speed  |
| -------------- | ------------ | ----------- | --------------- | ------ |
| SAM-SD2        | SAM ViT-H    | SD2 512px   | Good            | Fast   |
| SAM2-SDXL      | SAM 2        | SDXL 1024px | Very Good       | Medium |
| SegFormer-SDXL | SegFormer B2 | SDXL 1024px | Best (clothing) | Medium |

---

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
