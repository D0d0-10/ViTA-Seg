# ViTA-Seg: Vision Transformer for Amodal Segmentation in Robotics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the **ViTA-Seg architecture** for amodal segmentation.

---

## 🚀 Key Features

- **Hybrid Architecture**: Shared early decoder for common features + task-specific late decoders for specialized features
- **Fusion Mechanism**: Cross-task collaboration between amodal and occluded predictions
- **Multi-Dataset Support**: ViTASim, COCOA, and KINS datasets
- **Full Resolution Evaluation**: Proper alignment for fair comparison with C2F-Seg baseline
- **Vision Transformer Backbone**: Pretrained ViT-Base from timm

---

## 🏗️ Architecture

```
Input (RGB + Visible Mask)
    ↓
ViT Encoder (Pretrained)
    ↓
Shared Early Decoder (768→512→256)
    ↓
    ├──────────────────────┬
    ↓                      ↓                      
Occluded Late          Amodal Late            
Decoder (256→64)       Decoder (256→64)       
    ↓                      ↓                      
    ├──────────────────→ FUSION
    ↓                      ↓
Occluded Mask       Amodal Mask (Fused)
```

**Key Innovations:**
1. **Task-Specific Branches**: Specialized learning paths for amodal and occluded predictions
2. **Cross-Task Fusion**: Occluded features enhance amodal prediction
3. **Best of Both Worlds**: Combines specialization with collaboration

---

## 📥 Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/D0d0-10/vita-seg-hybrid-fused.git
cd vita-seg-hybrid-fused

# Create conda environment
conda env create -f environment.yml
conda activate vita-seg
```

### Using Pip

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA >= 11.8 (for GPU support)

---

## 📁 Dataset Preparation

### Expected Directory Structure

```
data/
├── ViTASim/
│   ├── train2014/
│   ├── val2014/
│   ├── ViTASim_amodal_train2014_with_classes.json
│   ├── ViTASim_amodal_val2014_with_classes.json
│   ├── fusion_train.pkl
│   └── fusion_test.pkl
├── COCOA/
│   ├── train2014/
│   ├── val2014/
│   ├── COCO_amodal_train2014_with_classes.json
│   └── COCO_amodal_val2014_with_classes.json
└── KINS/
    ├── training/image_2/
    ├── testing/image_2/
    ├── update_train_2020.json
    └── update_test_2020.json
```

Update dataset paths in `config_github.yaml` if needed.

---

## 🏋️ Training

### Basic Training

```bash
python training_dual_head_hybrid_fused.py --config config_github.yaml
```

### Training with Custom Occluded Weight

```bash
python training_dual_head_hybrid_fused.py \
    --config config_github.yaml \
    --occluded-weight 0.50
```

### Resume Training

```bash
python training_dual_head_hybrid_fused.py \
    --config config_github.yaml \
    --resume path/to/checkpoint.pth
```

### Key Training Arguments

- `--config`: Path to configuration file
- `--occluded-weight`: Override occluded loss weight (0.25-0.50)
- `--resume`: Path to checkpoint to resume from
- `--reset-epoch`: Reset epoch counter when resuming

---

## 📊 Evaluation

### Full Resolution Evaluation

```bash
python test_cocoa_fullres_aligned.py \
    --model your_model_run_name \
    --model-type dual-hybrid-fused
```

### Single Sample Testing

```bash
python test_single_sample_fullres_aligned.py \
    --model your_model_run_name \
    --model-type dual-hybrid-fused \
    --index 0
```

### Compare Two Models

```bash
python compare_two_models.py \
    --model1 model_1 \
    --model2 model_2 \
    --diff-threshold 0.05
```

Or with difference range:

```bash
python compare_two_models.py \
    --model1 model_1 \
    --model2 model_2 \
    --diff-range 0.02 0.05
```

---

## 🎨 Generate Publication Figures

```bash
python fig_paper_IFAC.py \
    --model your_model_run_name \
    --model-type dual-hybrid-fused \
    --index 0 \
    --min-occluded-pixels 100
```

Generates 6 figures with different visualizations (RGB + masks, predictions, GT comparisons).

---

## 📦 Model Files

### Core Architecture
- `model_dual_head_hybrid_fused.py` - Main model implementation
- `model.py` - Base ViT encoder with pretrained support
- `dataset.py` - Data loaders for ViTASim, COCOA, KINS

### Training & Evaluation
- `training_dual_head_hybrid_fused.py` - Training script with WandB integration
- `test_cocoa_fullres_aligned.py` - Full resolution evaluation
- `test_single_sample_fullres_aligned.py` - Single sample visualization
- `evaluation_metrics.py` - Comprehensive metrics computation

### Utilities
- `fig_paper_IFAC.py` - Publication figure generation
- `compare_two_models.py` - Model comparison tool

---

## 🛠️ Configuration

Edit `config_github.yaml` to customize:
- Dataset selection (`vitasim`, `cocoa`, `kins`, `all`)
- Model architecture (embed_dim, depth, num_heads)
- Training hyperparameters (lr, batch_size, epochs)
- Loss weights (amodal, occluded, dice, bce)
- WandB project settings

---

## 📈 Training Tips

1. **Occluded Weight**: Start with 0.25, increase to 0.50 if occluded IoU is low
2. **Learning Rate**: 5e-6 works well for ViT-Base with MAE pretraining
3. **Batch Size**: 8 recommended (adjust based on GPU memory)
4. **Warmup**: 1000 iterations for stable training
5. **Gradient Accumulation**: Use 2 steps for effective batch size of 16

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Vision Transformer implementation from [timm](https://github.com/huggingface/pytorch-image-models)
- Amodal segmentation benchmarks: [COCOA](https://arxiv.org/abs/1509.01329), [KINS](https://arxiv.org/abs/1811.11397)
- Inspired by [C2F-Seg](https://github.com/yihui-he/C2F-Seg)

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## ⭐ Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025vitaseg,
  title={ViTA-Seg: Hybrid+Fused Dual-Head Vision Transformer for Amodal Instance Segmentation},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2025}
}
```
