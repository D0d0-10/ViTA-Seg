"""
Test a specific sample by index from the validation dataset at FULL RESOLUTION
This script computes metrics at the original image resolution using the same
align_raw_size approach as c2f-seg (properly handling bbox cropping and padding).

This enables fair comparison with c2f-seg baseline at the same resolution.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
from pathlib import Path
import time
from PIL import Image
from torchvision import transforms

from model import AmodalViT
from model_dual_head import DualHeadAmodalViT
from model_dual_head_fused import DualHeadFusedAmodalViT
from model_dual_head_hybrid import HybridDualHeadAmodalViT
from model_dual_head_hybrid_fused import HybridFusedDualHeadAmodalViT
from dataset import COCOADataset, ViTASimDataset


def compute_metrics(pred, target):
    """Compute IoU, Dice, Precision, Recall - handles empty masks correctly"""
    pred_binary = (pred > 0.5)  # Keep as boolean for bitwise operations
    gt_binary = (target > 0.5)  # Keep as boolean for bitwise operations
    
    intersection = (pred_binary & gt_binary).sum().float()  # Logical AND for intersection
    union = (pred_binary | gt_binary).sum().float()  # Logical OR for union
    
    # Handle empty union case
    if union.item() == 0:
        iou = 1.0  # Both empty = perfect match
    else:
        iou = (intersection / union).item()
    
    pred_count = pred_binary.sum().float()
    gt_count = gt_binary.sum().float()
    
    if (pred_count + gt_count).item() == 0:
        dice = 1.0
    else:
        dice = (2 * intersection / (pred_count + gt_count)).item()
    
    tp = intersection
    fp = (pred_binary & ~gt_binary).sum().float()  # False Positive = pred AND NOT gt
    fn = (~pred_binary & gt_binary).sum().float()  # False Negative = NOT pred AND gt
    
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    
    return iou, dice, precision, recall


def align_raw_size(pred_crop, bbox_original, image_size, pad_factor=0.2, target_size=(224, 224)):
    """
    Align cropped prediction to original full image size (properly accounting for padding)
    
    This follows the same logic as dataset.py crop_and_resize:
    1. Bbox is padded by pad_factor (20% by default)
    2. Prediction is resized from target_size to the actual cropped size (with padding)
    3. Prediction is placed back at the correct position in full image
    
    Args:
        pred_crop: Predicted mask at model output size (e.g., 1, 1, 224, 224)
        bbox_original: Original bounding box [x, y, width, height] in COCO format
        image_size: Original image size (H, W)
        pad_factor: Padding factor used in crop_and_resize (default: 0.2 = 20%)
        target_size: Model input size (H, W)
    
    Returns:
        Aligned mask at original full image size
    """
    device = pred_crop.device
    H_orig, W_orig = image_size
    
    # Initialize full-size mask with zeros
    full_mask = torch.zeros((1, 1, H_orig, W_orig), dtype=torch.float32, device=device)
    
    # Unpack bbox
    x, y, w, h = bbox_original
    
    # Calculate padded bbox (same as in dataset.py crop_and_resize)
    pad_w = int(w * pad_factor)
    pad_h = int(h * pad_factor)
    
    x1 = int(x) - pad_w
    y1 = int(y) - pad_h
    x2 = int(x + w) + pad_w
    y2 = int(y + h) + pad_h
    
    # Clip to image boundaries
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)
    x2_clipped = min(W_orig, x2)
    y2_clipped = min(H_orig, y2)
    
    # Get actual cropped size (after boundary clipping)
    crop_h = y2_clipped - y1_clipped
    crop_w = x2_clipped - x1_clipped
    
    # Resize prediction from target_size to actual crop size
    if crop_h > 0 and crop_w > 0:
        resize = transforms.Resize([crop_h, crop_w], interpolation=transforms.InterpolationMode.BILINEAR)
        pred_resized = resize(pred_crop)
        
        # Place resized prediction at the correct position in full image
        full_mask[:, :, y1_clipped:y2_clipped, x1_clipped:x2_clipped] = pred_resized
    
    return full_mask


def find_best_checkpoint(run_name, model_type='single'):
    """
    Find the best checkpoint file in the specified run directory
    
    Args:
        run_name: Name of the wandb run (e.g., 'cosmic-galaxy-42')
        model_type: 'single' or 'dual' to determine base directory
        
    Returns:
        Tuple of (checkpoint_path, config_path)
    """
    if model_type == 'dual':
        best_models_dir = Path('best-models-dual-head')
    elif model_type == 'dual-fused':
        best_models_dir = Path('best-models-dual-head-fused')
    elif model_type == 'dual-hybrid':
        best_models_dir = Path('best-models-dual-head-hybrid')
    elif model_type == 'dual-hybrid-fused':
        best_models_dir = Path('best-models-dual-head-hybrid-fused')
    else:
        best_models_dir = Path('best-models')
    
    run_dir = best_models_dir / run_name
    
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Available runs in {best_models_dir}/:\n" +
            "\n".join([f"  - {d.name}" for d in best_models_dir.iterdir() if d.is_dir()])
        )
    
    # Find the checkpoint file (should be only one)
    checkpoint_files = list(run_dir.glob("best_model_epoch_*.pth"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    
    if len(checkpoint_files) > 1:
        print(f"⚠️  Warning: Multiple checkpoints found in {run_dir}. Using the latest one.")
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    checkpoint_path = checkpoint_files[0]
    config_path = run_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return str(checkpoint_path), str(config_path), run_dir


def test_single_sample_fullres_aligned(run_name=None, checkpoint_path=None, config_path=None, 
                                        sample_idx=0, model_type='single'):
    """
    Test a single sample from validation dataset at FULL RESOLUTION
    using align_raw_size approach (like c2f-seg)
    
    Args:
        run_name: Name of the wandb run to load checkpoint from
        checkpoint_path: Direct path to checkpoint file (alternative to run_name)
        config_path: Direct path to config file (alternative to run_name)
        sample_idx: Index of the sample to test
        model_type: 'single' or 'dual' head model
    """
    # Load checkpoint and config
    if run_name:
        print(f"🔍 Looking for {model_type}-head model: {run_name}")
        checkpoint_path, config_path, save_dir = find_best_checkpoint(run_name, model_type)
        print(f"✅ Found checkpoint: {checkpoint_path}")
        print(f"✅ Found config: {config_path}")
    else:
        if not checkpoint_path or not config_path:
            raise ValueError("Must provide either run_name or both checkpoint_path and config_path")
        print(f"Using provided paths:")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Config: {config_path}")
        save_dir = Path(checkpoint_path).parent
    
    print(f"📍 Testing sample index: {sample_idx} at FULL RESOLUTION (aligned like c2f-seg)")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\n🔄 Loading {model_type}-head model...")
    if model_type == 'dual':
        model = DualHeadAmodalViT(**config['model']).to(device)
    elif model_type == 'dual-fused':
        model = DualHeadFusedAmodalViT(**config['model']).to(device)
    elif model_type == 'dual-hybrid':
        model = HybridDualHeadAmodalViT(**config['model']).to(device)
    elif model_type == 'dual-hybrid-fused':
        model = HybridFusedDualHeadAmodalViT(**config['model']).to(device)
    else:
        model = AmodalViT(**config['model']).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create validation dataset (choose dataset based on config)
    print("\n📂 Loading validation dataset...")
    dataset_choice = config['data'].get('datasets', 'cocoa')

    if dataset_choice in ['vitasim', 'both', 'all']:
        print("Using ViTASimDataset (vitasim)")
        val_dataset = ViTASimDataset(
            root_dir=config['data']['vitasim']['root_dir'],
            annotation_file=config['data']['vitasim']['val_annotations'],
            split='val',
            fusion_file=config['data']['vitasim'].get('val_fusion', None),
            target_size=tuple(config['data']['target_size'])
        )
    else:
        print("Using COCOADataset (cocoa)")
        val_dataset = COCOADataset(
            root_dir=config['data']['cocoa']['root_dir'],
            annotation_file=config['data']['cocoa']['val_annotations'],
            split='val',
            target_size=tuple(config['data']['target_size'])
        )
    
    print(f"   Validation dataset: {len(val_dataset)} samples")
    
    if sample_idx >= len(val_dataset):
        print(f"❌ Error: Index {sample_idx} is out of range. Dataset has {len(val_dataset)} samples.")
        return
    
    # Get annotation info
    ann = val_dataset.annotations[sample_idx]
    image_info = val_dataset.images[ann['image_id']]
    filename = image_info['file_name']
    bbox = ann['bbox']  # [x, y, width, height] in COCO format
    
    print(f"\n{'='*60}")
    print(f"Processing Sample {sample_idx} at FULL RESOLUTION (ALIGNED)")
    print(f"{'='*60}")
    
    # Load FULL resolution image (use correct dataset root)
    if dataset_choice in ['vitasim', 'both', 'all']:
        img_path = Path(config['data']['vitasim']['root_dir']) / 'val2014' / filename
        if not img_path.exists():
            img_path = Path(config['data']['vitasim']['root_dir']) / filename
    else:
        img_path = Path(config['data']['cocoa']['root_dir']) / 'val2014' / filename
        if not img_path.exists():
            img_path = Path(config['data']['cocoa']['root_dir']) / filename
    
    pil_img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = pil_img.size
    
    print(f"Original image size: {W_orig}x{H_orig}")
    print(f"Bbox: {bbox} (x, y, w, h)")
    
    # Decode FULL resolution GT masks (use correct method based on dataset type)
    if dataset_choice in ['vitasim', 'both', 'all']:
        # ViTASimDataset uses decode_rle_mask
        gt_amodal_full = val_dataset.decode_rle_mask(ann['segmentation'])
        if 'visible_mask' in ann:
            gt_visible_full = val_dataset.decode_rle_mask(ann['visible_mask'])
        else:
            gt_visible_full = gt_amodal_full.copy()
    else:
        # COCOADataset uses decode_segmentation
        gt_amodal_full = val_dataset.decode_segmentation(ann['segmentation'])
        if 'visible_mask' in ann:
            gt_visible_full = val_dataset.decode_segmentation(ann['visible_mask'])
        elif 'inmodal_seg' in ann:
            gt_visible_full = val_dataset.decode_segmentation(ann['inmodal_seg'])
        else:
            gt_visible_full = gt_amodal_full.copy()
    
    # Get the cropped sample (this includes bbox cropping and padding)
    sample = val_dataset[sample_idx]
    
    # Run inference on cropped input
    image = sample['image'].unsqueeze(0).to(device)
    visible_mask = sample['visible_mask'].unsqueeze(0).to(device)
    
    # Forward pass with timing
    print("🧠 Running inference on cropped input...")
    
    # Warm-up run
    if device.type == 'cuda':
        with torch.no_grad():
            if model_type in ['dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused']:
                _ = model(image, visible_mask)
            else:
                _ = model(image, visible_mask)
        torch.cuda.synchronize()
    
    # Timed inference
    start_time = time.time()
    with torch.no_grad():
        if model_type in ['dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused']:
            preds = model(image, visible_mask)
            pred_amodal_crop = preds['amodal']  # (1, 1, target_h, target_w)
            pred_occluded_crop = preds['occluded']
        else:
            pred_amodal_crop = model(image, visible_mask)  # (1, 1, target_h, target_w)
            pred_occluded_crop = None
    
    # Ensure GPU operations are complete before measuring time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    inference_time = time.time() - start_time
    
    # Align predictions to original full resolution using the same padding as dataset.py
    print(f"\n📐 Aligning predictions to full resolution (accounting for 20% padding)...")
    pred_amodal_full = align_raw_size(
        pred_amodal_crop, 
        bbox,  # Original bbox [x, y, w, h]
        (H_orig, W_orig),
        pad_factor=0.2,  # Same as in dataset.py
        target_size=config['data']['target_size']
    )
    pred_amodal_full = pred_amodal_full.squeeze(0).squeeze(0).cpu()
    
    if pred_occluded_crop is not None:
        pred_occluded_full = align_raw_size(
            pred_occluded_crop,
            bbox,  # Original bbox [x, y, w, h]
            (H_orig, W_orig),
            pad_factor=0.2,
            target_size=config['data']['target_size']
        )
        pred_occluded_full = pred_occluded_full.squeeze(0).squeeze(0).cpu()
    else:
        # Derive occluded from amodal - visible
        pred_occluded_full = (pred_amodal_full > 0.5).float() * (1.0 - torch.from_numpy(gt_visible_full).float())
    
    # Convert GT masks to tensors
    gt_amodal_tensor = torch.from_numpy(gt_amodal_full).float()
    gt_visible_tensor = torch.from_numpy(gt_visible_full).float()
    
    # Compute GT occluded mask
    gt_occluded_tensor = ((gt_amodal_tensor > 0.5) & ~(gt_visible_tensor > 0.5)).float()
    
    # Visualize full resolution binary masks for debugging (after alignment)
    print(f"\n🔍 Visualizing FULL RESOLUTION predictions after alignment ({H_orig}x{W_orig})...")
    pred_occluded_binary_full = (pred_occluded_full > 0.5)
    gt_occluded_binary_full = (gt_occluded_tensor > 0.5)
    # pred_occluded_binary_full = (pred_occluded_full)
    # gt_occluded_binary_full = (gt_occluded_tensor)
    
    pred_pixels = pred_occluded_binary_full.sum().item()
    gt_pixels = gt_occluded_binary_full.sum().item()
    intersection_pixels = (pred_occluded_binary_full & gt_occluded_binary_full).sum().item()
    union_pixels = (pred_occluded_binary_full | gt_occluded_binary_full).sum().item()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Full Resolution Occluded Masks (After Alignment) - Size: {H_orig}x{W_orig}', 
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(pred_occluded_binary_full.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Prediction Binary (>0.5)\nPixels: {pred_pixels:.0f}', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gt_occluded_binary_full.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Ground Truth Binary (>0.5)\nPixels: {gt_pixels:.0f}', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow((pred_occluded_binary_full & gt_occluded_binary_full).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Intersection\nPixels: {intersection_pixels:.0f}', fontsize=10, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow((pred_occluded_binary_full | gt_occluded_binary_full).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'Union\nPixels: {union_pixels:.0f}', fontsize=10, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compute metrics at FULL resolution
    print("\n📊 Computing metrics at FULL RESOLUTION (ALIGNED)...")
    amodal_iou, amodal_dice, amodal_prec, amodal_rec = compute_metrics(pred_amodal_full, gt_amodal_tensor)
    occluded_iou, occluded_dice, occluded_prec, occluded_rec = compute_metrics(pred_occluded_full, gt_occluded_tensor)
    
    # Compute visible region IoU
    visible_binary = (gt_visible_tensor > 0.5)
    pred_amodal_binary = (pred_amodal_full > 0.5)
    gt_amodal_binary = (gt_amodal_tensor > 0.5)
    
    visible_intersection = (pred_amodal_binary & visible_binary & gt_amodal_binary).sum().float()
    visible_union = ((pred_amodal_binary & visible_binary) | (gt_amodal_binary & visible_binary)).sum().float()
    
    if visible_union.item() == 0:
        visible_iou = 1.0
    else:
        visible_iou = (visible_intersection / visible_union).item()
    
    # Print metrics
    print("\n" + "="*60)
    print("FULL RESOLUTION METRICS (ALIGNED LIKE C2F-SEG)")
    print("="*60)
    print(f"\n📊 Amodal Prediction Metrics:")
    print(f"   IoU:       {amodal_iou:.4f}")
    print(f"   Dice:      {amodal_dice:.4f}")
    print(f"   Precision: {amodal_prec:.4f}")
    print(f"   Recall:    {amodal_rec:.4f}")
    
    print(f"\n📊 Occluded Region Metrics:")
    print(f"   IoU:       {occluded_iou:.4f}")
    print(f"   Dice:      {occluded_dice:.4f}")
    print(f"   Precision: {occluded_prec:.4f}")
    print(f"   Recall:    {occluded_rec:.4f}")
    
    print(f"\n📊 Region-Specific IoU:")
    print(f"   Visible IoU:   {visible_iou:.4f}")
    print(f"   Occluded IoU:  {occluded_iou:.4f}")
    print(f"   Gap:           {abs(visible_iou - occluded_iou):.4f}")
    
    print(f"\n⏱️  Inference Time:")
    print(f"   Time:      {inference_time*1000:.2f} ms ({inference_time:.4f} s)")
    
    # Compute occlusion rate
    gt_occluded_area = gt_occluded_tensor.sum().item()
    gt_amodal_area = gt_amodal_binary.sum().item()
    occlusion_rate = (gt_occluded_area / (gt_amodal_area + 1e-8)) * 100
    
    # Calculate the actual padded bbox for display
    x, y, w, h = bbox
    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)
    x1_padded = max(0, int(x) - pad_w)
    y1_padded = max(0, int(y) - pad_h)
    x2_padded = min(W_orig, int(x + w) + pad_w)
    y2_padded = min(H_orig, int(y + h) + pad_h)
    
    # Print sample info
    print("\n📋 Object Information:")
    print(f"   Image ID:      {ann['image_id']}")
    print(f"   Annotation ID: {ann['id']}")
    print(f"   Category ID:   {ann['category_id']}")
    print(f"   Filename:      {filename}")
    print(f"   Resolution:    {W_orig}x{H_orig}")
    print(f"   BBox (orig):   {bbox} (x, y, w, h)")
    print(f"   BBox (padded): [{x1_padded}, {y1_padded}, {x2_padded-x1_padded}, {y2_padded-y1_padded}]")
    print(f"   Crop size:     {x2_padded-x1_padded}x{y2_padded-y1_padded}")
    print(f"   Occlusion:     {occlusion_rate:.1f}%")
    print("="*60)
    
    # Visualize - 5 columns with full resolution data
    print(f"\n🎨 Creating full-resolution visualization (ALIGNED)...")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Prepare data for visualization
    img_np = np.array(pil_img) / 255.0
    visible_np = gt_visible_full
    gt_amodal_np = gt_amodal_full
    pred_amodal_np = pred_amodal_full.numpy()
    pred_amodal_bin = (pred_amodal_np > 0.5).astype(np.float32)
    
    # Column 1: Original Image (full res)
    axes[0].imshow(img_np)
    axes[0].set_title(f'Image (Full Res)\n{W_orig}x{H_orig}', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Draw both original bbox (red) and padded bbox (blue) on image
    from matplotlib.patches import Rectangle
    # Original bbox
    rect_orig = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                          linewidth=2, edgecolor='red', facecolor='none', label='Original BBox')
    axes[0].add_patch(rect_orig)
    
    # Padded bbox
    x, y, w, h = bbox
    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)
    x1_padded = max(0, int(x) - pad_w)
    y1_padded = max(0, int(y) - pad_h)
    x2_padded = min(W_orig, int(x + w) + pad_w)
    y2_padded = min(H_orig, int(y + h) + pad_h)
    
    rect_padded = Rectangle((x1_padded, y1_padded), x2_padded - x1_padded, y2_padded - y1_padded,
                            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--', label='Padded BBox')
    axes[0].add_patch(rect_padded)
    
    # Column 2: Image + Visible Mask Overlay
    img_with_visible = img_np.copy()
    mask_overlay = np.zeros_like(img_np)
    mask_overlay[:, :, 1] = visible_np  # Green for visible
    axes[1].imshow(img_with_visible)
    axes[1].imshow(mask_overlay, alpha=0.5)
    axes[1].set_title('Image + Visible', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    # Column 3: Ground Truth Amodal
    axes[2].imshow(gt_amodal_np, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('GT Amodal', fontsize=10, fontweight='bold')
    axes[2].axis('off')
    
    # Column 4: Predicted Amodal
    axes[3].imshow(pred_amodal_np, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title(f'Pred Amodal (Aligned)\nIoU: {amodal_iou:.3f}', fontsize=10, fontweight='bold')
    axes[3].axis('off')
    
    # Column 5: Comparison Overlay
    overlay = np.zeros((*gt_amodal_np.shape, 3))
    overlay[:, :, 1] = gt_amodal_np  # Green = Ground Truth
    overlay[:, :, 0] = pred_amodal_bin  # Red = Prediction
    
    axes[4].imshow(overlay)
    axes[4].set_title('Amodal Comparison\n(GT:Green | Pred:Red)', fontsize=10, fontweight='bold')
    axes[4].axis('off')
    
    # Add object info on the left
    info_text = (f"Method: align_raw_size\n"
                f"Cat: {ann['category_id']}\n"
                f"Ann: {ann['id']}\n"
                f"Occ Rate: {occlusion_rate:.1f}%\n"
                f"Amodal IoU: {amodal_iou:.3f}\n"
                f"Occ IoU: {occluded_iou:.3f}\n"
                f"Time: {inference_time*1000:.1f}ms\n"
                f"Full Res: {W_orig}x{H_orig}")
    axes[0].text(-0.15, 0.5, info_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    save_filename = f'sample_{sample_idx}_fullres_aligned_visualization.png'
    save_path = save_dir / save_filename
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()
    
    return {
        'amodal_iou': amodal_iou,
        'occluded_iou': occluded_iou,
        'visible_iou': visible_iou,
        'inference_time': inference_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a sample at FULL RESOLUTION using align_raw_size (like c2f-seg)')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the wandb run (e.g., peachy-frost-55)')
    parser.add_argument('--model-type', choices=['single', 'dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused'], default='single',
                        help='Model type: single-head or dual-head')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Direct path to checkpoint file (alternative to --model)')
    parser.add_argument('--config', type=str, default=None,
                        help='Direct path to configuration file (alternative to --model)')
    parser.add_argument('--index', type=int, required=True,
                        help='Sample index to test')
    
    args = parser.parse_args()
    
    # Test the sample
    try:
        test_single_sample_fullres_aligned(
            run_name=args.model,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            sample_idx=args.index,
            model_type=args.model_type
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
