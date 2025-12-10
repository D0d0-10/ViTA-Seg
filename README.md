# ViTA-Seg: Vision Transformer for Amodal Segmentation in Robotics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the **ViTA-Seg architecture** for amodal segmentation.

<p align="center">
  <img src="images/architecture.png" alt="Architecture Diagram" width="800"/>
  <br>
  <em>Figure 1: The proposed ViTA-Seg Dual Head architectures.</em>
</p>

<p align="center">
  <img src="images/ViTA-SimData.png" alt="ViTA-SimData Dataset" width="800"/>
  <br>
  <em>Figure 2: The qualitative results estimated by C2F-Seg, ViTA-Seg Single Head and Dual Head on ViTA-SimData.</em>
</p>

<p align="center">
  <img src="images/COCOA_KINS.png" alt="COCOA and KINS Datasets" width="800"/>
  <br>
  <em>Figure 3: The qualitative results estimated by C2F-Seg, ViTA-Seg Single Head and Dual Head on COCOA and KINS.</em>
</p>

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

## 📊 Performance

Performance comparison of ViTA-Seg and C2F-Seg models on the validation datasets:

| Model | KINS | | | COCOA | | | ViTA-SimData | | | Inference Time |
|:------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| | mIoU<sub>A</sub> | mIoU<sub>V</sub> | mIoU<sub>O</sub> | mIoU<sub>A</sub> | mIoU<sub>V</sub> | mIoU<sub>O</sub> | mIoU<sub>A</sub> | mIoU<sub>V</sub> | mIoU<sub>O</sub> | t<sub>inf</sub> (ms) |
| C2F-Seg | 87.89 | 72.12 | 57.30 | 87.15 | 92.52 | 36.69 | 85.17 | 90.67 | 53.28 | 113.77 |
| ViTA-Seg Single-Head | 89.94 | **98.84** | 60.05 | 91.57 | **99.63** | 27.65 | 88.31 | 91.01 | 49.33 | **8.54** |
| ViTA-Seg Dual-Head | **91.12** | 98.65 | **63.48** | **93.70** | 99.38 | **49.88** | **91.09** | **91.48** | **58.65** | 8.96 |

**Metrics:**
- mIoU<sub>A</sub>: Mean IoU for Amodal masks
- mIoU<sub>V</sub>: Mean IoU for Visible masks
- mIoU<sub>O</sub>: Mean IoU for Occluded regions
- t<sub>inf</sub>: Inference time per image (ms)

**Key Results:**
- ✅ ViTA-Seg Dual-Head achieves **best overall performance** across all datasets
- ✅ **13x faster** inference than C2F-Seg (8.96ms vs 113.77ms)
- ✅ Superior amodal and occluded segmentation on COCOA and KINS
- ✅ Single-Head variant optimal for visible mask prediction

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

Update dataset paths in `config_dual_head_hybrid_fused.yaml` if needed.

---

## 🏋️ Training

### Basic Training

```bash
python training_dual_head_hybrid_fused.py --config config_dual_head_hybrid_fused.yaml
```

### Training with Custom Occluded Weight

```bash
python training_dual_head_hybrid_fused.py \
    --config config_dual_head_hybrid_fused.yaml \
    --occluded-weight 0.50
```

### Resume Training

```bash
python training_dual_head_hybrid_fused.py \
    --config config_dual_head_hybrid_fused.yaml \
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

## 🛠️ Configuration

Edit `config_dual_head_hybrid_fused.yaml` to customize:
- Dataset selection (`vitasim`, `cocoa`, `kins`, `all`)
- Model architecture (embed_dim, depth, num_heads)
- Training hyperparameters (lr, batch_size, epochs)
- Loss weights (amodal, occluded, dice, bce)
- WandB project settings

---

## 📈 Training Tips

1. **Occluded Weight**: Set to 0.25, increase to 0.50 if occluded IoU is low
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
