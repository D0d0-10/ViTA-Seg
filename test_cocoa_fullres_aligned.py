"""
Test the validation dataset at FULL RESOLUTION
This script computes metrics at the original image resolution using the same
align_raw_size approach as c2f-seg (properly handling bbox cropping and padding).

The dataset is automatically selected based on the model's config file.
This enables fair comparison with c2f-seg baseline at the same resolution.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import argparse
from pathlib import Path
import time
from PIL import Image
from torchvision import transforms
import csv
from tqdm import tqdm

from model import AmodalViT
from model_dual_head import DualHeadAmodalViT
from model_dual_head_fused import DualHeadFusedAmodalViT
from model_dual_head_hybrid import HybridDualHeadAmodalViT
from model_dual_head_hybrid_fused import HybridFusedDualHeadAmodalViT
from dataset import COCOADataset, KINSDataset, ViTASimDataset


def create_val_dataset(config):
    """
    Create validation dataset based on config file settings
    
    Args:
        config: Configuration dictionary from the model's config.yaml
        
    Returns:
        Tuple of (dataset, dataset_name, pad_factor, enlarge_coef)
    """
    target_size = tuple(config['data']['target_size'])
    datasets_to_use = config['data']['datasets']
    
    print(f"📂 Dataset mode from config: '{datasets_to_use}'")
    
    # Determine which dataset to use for validation
    if datasets_to_use == 'kins':
        # Use KINS validation set
        print("   Loading KINS validation dataset...")
        kins_config = config['data']['kins']
        
        
        val_ann_path = os.path.join(kins_config['root_dir'], kins_config['val_annotations'])
        
        if not os.path.exists(val_ann_path):
            raise FileNotFoundError(f"KINS validation annotations not found: {val_ann_path}")
        
        dataset = KINSDataset(
            root_dir=kins_config['root_dir'],
            annotation_file=val_ann_path,
            split='test',  # KINS uses 'test' for validation
            target_size=target_size,
            enlarge_coef=kins_config.get('enlarge_coef', 2.0)
        )
        
        return dataset, 'KINS', None, kins_config.get('enlarge_coef', 2.0)
    
    elif datasets_to_use == 'cocoa' or datasets_to_use == 'both':
        # Use COCOA validation set
        print("   Loading COCOA validation dataset...")
        cocoa_config = config['data']['cocoa']
        
        if not os.path.exists(cocoa_config['val_annotations']):
            raise FileNotFoundError(f"COCOA validation annotations not found: {cocoa_config['val_annotations']}")
        
        dataset = COCOADataset(
            root_dir=cocoa_config['root_dir'],
            annotation_file=cocoa_config['val_annotations'],
            split='val',
            target_size=target_size
        )
        
        return dataset, 'COCOA', 0.2, None
    
    elif datasets_to_use in ['vitasim', 'all']:
        # Use ViTASim validation set
        print("   Loading ViTASim validation dataset...")
        vitasim_config = config['data']['vitasim']
        
        dataset = ViTASimDataset(
            root_dir=vitasim_config['root_dir'],
            annotation_file=vitasim_config['val_annotations'],
            split='val',
            fusion_file=vitasim_config.get('val_fusion', None),
            target_size=target_size
        )
        
        return dataset, 'ViTASim', 0.2, None
    
    else:
        raise ValueError(f"Unknown dataset mode: {datasets_to_use}")


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


def align_raw_size(pred_crop, bbox_original, image_size, pad_factor=None, enlarge_coef=None, target_size=(224, 224)):
    """
    Align cropped prediction to original full image size (properly accounting for padding)
    
    This follows the same logic as dataset.py crop_and_resize:
    - For COCOA/ViTASim: Bbox is padded by pad_factor (20% by default)
    - For KINS: Bbox is enlarged by enlarge_coef (2.0 by default)
    
    Args:
        pred_crop: Predicted mask at model output size (e.g., 1, 1, 224, 224)
        bbox_original: Original bounding box [x, y, width, height] in COCO format
        image_size: Original image size (H, W)
        pad_factor: Padding factor for COCOA/ViTASim (default: 0.2 = 20%)
        enlarge_coef: Enlargement coefficient for KINS (default: 2.0)
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
    
    # Calculate padded/enlarged bbox based on dataset type
    if enlarge_coef is not None:
        # KINS: use enlarge_coef
        pad_w = int(w * (enlarge_coef - 1) / 2)
        pad_h = int(h * (enlarge_coef - 1) / 2)
    else:
        # COCOA/ViTASim: use pad_factor
        if pad_factor is None:
            pad_factor = 0.2
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
        model_type: 'single', 'dual', 'dual-fused', 'dual-hybrid', or 'dual-hybrid-fused' to determine base directory
        
    Returns:
        Tuple of (checkpoint_path, config_path, run_dir)
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


def test_cocoa_fullres_aligned(run_name=None, checkpoint_path=None, config_path=None, 
                                model_type='single', save_per_sample=True, override_dataset=None):
    """
    Test the validation dataset at FULL RESOLUTION
    using align_raw_size approach (like c2f-seg)
    
    The dataset is automatically selected from the model's config file.
    
    Args:
        run_name: Name of the wandb run to load checkpoint from
        checkpoint_path: Direct path to checkpoint file (alternative to run_name)
        config_path: Direct path to config file (alternative to run_name)
        model_type: 'single' or 'dual' head model
        save_per_sample: Whether to save per-sample results to CSV
        override_dataset: Override dataset selection (e.g., 'cocoa', 'kins', 'vitasim')
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
    
    print(f"📍 Testing validation set at FULL RESOLUTION (aligned like c2f-seg)")
    
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
    
    # Create validation dataset based on config (or override)
    print(f"\n📂 Loading validation dataset from config...")
    
    if override_dataset:
        # Temporarily override the dataset setting
        original_datasets = config['data']['datasets']
        config['data']['datasets'] = override_dataset
        print(f"   ⚠️  Override: Using '{override_dataset}' instead of '{original_datasets}'")
    
    val_dataset, dataset_name, pad_factor, enlarge_coef = create_val_dataset(config)
    
    print(f"   ✅ Loaded {dataset_name} validation dataset: {len(val_dataset)} samples")
    if pad_factor is not None:
        print(f"   📏 Using pad_factor={pad_factor} for bbox alignment")
    if enlarge_coef is not None:
        print(f"   📏 Using enlarge_coef={enlarge_coef} for bbox alignment")
    print(f"   Starting evaluation...\n")
    
    # Initialize metrics storage
    results = []
    
    # Aggregate metrics
    total_amodal_iou = 0.0
    total_amodal_dice = 0.0
    total_occluded_iou = 0.0
    total_occluded_dice = 0.0
    total_visible_iou = 0.0
    total_inference_time = 0.0
    
    # Track occluded samples separately
    total_occluded_iou_only_occluded = 0.0
    total_occluded_dice_only_occluded = 0.0
    num_occluded_samples = 0
    
    # Process each sample
    num_samples = len(val_dataset)
    
    # Warmup: Run a few inference passes to warm up GPU/CUDA kernels
    print(f"\n🔥 Warming up GPU with 10 inference passes...")
    with torch.no_grad():
        for warmup_idx in range(min(10, num_samples)):
            sample_warmup = val_dataset[warmup_idx]
            image_warmup = sample_warmup['image'].unsqueeze(0).to(device)
            visible_warmup = sample_warmup['visible_mask'].unsqueeze(0).to(device)
            
            if model_type in ['dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused']:
                _ = model(image_warmup, visible_warmup)
            else:
                _ = model(image_warmup, visible_warmup)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    print(f"✅ Warmup complete. Starting timed evaluation...\n")
    
    with torch.no_grad():
        for sample_idx in tqdm(range(num_samples), desc="Processing samples", unit="sample"):
            try:
                # Get annotation info
                ann = val_dataset.annotations[sample_idx]
                image_info = val_dataset.images[ann['image_id']]
                filename = image_info['file_name']
                
                # Get bbox based on dataset type
                if dataset_name == 'KINS':
                    bbox = ann['i_bbox']  # KINS uses i_bbox (inmodal bbox)
                else:
                    bbox = ann['bbox']  # COCOA/ViTASim use bbox
                
                # Load FULL resolution image (dataset-specific paths)
                if dataset_name == 'KINS':
                    img_dir = Path(config['data']['kins']['root_dir']) / 'testing' / 'image_2'
                    img_path = img_dir / filename
                elif dataset_name == 'ViTASim':
                    img_path = Path(config['data']['vitasim']['root_dir']) / 'val2014' / filename
                    if not img_path.exists():
                        img_path = Path(config['data']['vitasim']['root_dir']) / filename
                else:  # COCOA
                    img_path = Path(config['data']['cocoa']['root_dir']) / 'val2014' / filename
                    if not img_path.exists():
                        img_path = Path(config['data']['cocoa']['root_dir']) / filename
                
                if not img_path.exists():
                    print(f"\n⚠️  Image not found: {img_path}, skipping...")
                    continue
                
                pil_img = Image.open(img_path).convert('RGB')
                W_orig, H_orig = pil_img.size
                
                # Decode FULL resolution GT masks (dataset-specific keys and methods)
                if dataset_name == 'KINS':
                    gt_amodal_full = val_dataset.decode_segmentation(ann['a_segm'], H_orig, W_orig)
                    gt_visible_full = val_dataset.decode_segmentation(ann['i_segm'], H_orig, W_orig)
                elif dataset_name == 'ViTASim':
                    # ViTASimDataset uses decode_rle_mask
                    gt_amodal_full = val_dataset.decode_rle_mask(ann['segmentation'])
                    if 'visible_mask' in ann:
                        gt_visible_full = val_dataset.decode_rle_mask(ann['visible_mask'])
                    else:
                        gt_visible_full = gt_amodal_full.copy()
                else:  # COCOA
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
                
                # Timed inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                if model_type in ['dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused']:
                    preds = model(image, visible_mask)
                    pred_amodal_crop = preds['amodal']  # (1, 1, target_h, target_w)
                    pred_occluded_crop = preds['occluded']
                else:
                    pred_amodal_crop = model(image, visible_mask)  # (1, 1, target_h, target_w)
                    pred_occluded_crop = None
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                inference_time = time.time() - start_time
                
                # Align predictions to original full resolution (using correct parameters)
                pred_amodal_full = align_raw_size(
                    pred_amodal_crop, 
                    bbox,  # Original bbox [x, y, w, h]
                    (H_orig, W_orig),
                    pad_factor=pad_factor,
                    enlarge_coef=enlarge_coef,
                    target_size=config['data']['target_size']
                )
                pred_amodal_full = pred_amodal_full.squeeze(0).squeeze(0).cpu()
                
                if pred_occluded_crop is not None:
                    pred_occluded_full = align_raw_size(
                        pred_occluded_crop,
                        bbox,  # Original bbox [x, y, w, h]
                        (H_orig, W_orig),
                        pad_factor=pad_factor,
                        enlarge_coef=enlarge_coef,
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
                
                # Compute metrics at FULL resolution
                amodal_iou, amodal_dice, amodal_prec, amodal_rec = compute_metrics(pred_amodal_full, gt_amodal_tensor)
                occluded_iou, occluded_dice, occluded_prec, occluded_rec = compute_metrics(pred_occluded_full, gt_occluded_tensor)
                
                # Compute visible region IoU
                visible_binary = (gt_visible_tensor > 0.5)
                pred_amodal_binary = (pred_amodal_full > 0.5)
                gt_amodal_binary = (gt_amodal_tensor > 0.5)
                
                visible_intersection = (pred_amodal_binary & visible_binary & gt_amodal_binary).sum().float()
                # visible_union = ((pred_amodal_binary & visible_binary) | (gt_amodal_binary & visible_binary)).sum().float()
                visible_union = ((pred_amodal_binary & visible_binary) | (visible_binary)).sum().float()
                
                if visible_union.item() == 0:
                    visible_iou = 1.0
                else:
                    visible_iou = (visible_intersection / visible_union).item()
                
                # Compute occlusion rate
                gt_occluded_area = gt_occluded_tensor.sum().item()
                gt_amodal_area = gt_amodal_binary.sum().item()
                occlusion_rate = (gt_occluded_area / (gt_amodal_area + 1e-8)) * 100
                
                # Track samples with actual occlusion (exclude non-occluded samples)
                has_occlusion = occlusion_rate > 0.1  # Consider >0.1% as occluded (threshold to avoid floating point errors)
                if has_occlusion:
                    total_occluded_iou_only_occluded += occluded_iou
                    total_occluded_dice_only_occluded += occluded_dice
                    total_occluded_iou += occluded_iou  # Only add to total if occluded
                    total_occluded_dice += occluded_dice  # Only add to total if occluded
                    num_occluded_samples += 1
                
                # Store results
                results.append({
                    'sample_idx': sample_idx,
                    'image_id': ann['image_id'],
                    'annotation_id': ann['id'],
                    'category_id': ann['category_id'],
                    'filename': filename,
                    'width': W_orig,
                    'height': H_orig,
                    'bbox_x': bbox[0],
                    'bbox_y': bbox[1],
                    'bbox_w': bbox[2],
                    'bbox_h': bbox[3],
                    'occlusion_rate': occlusion_rate,
                    'amodal_iou': amodal_iou,
                    'amodal_dice': amodal_dice,
                    'amodal_precision': amodal_prec,
                    'amodal_recall': amodal_rec,
                    'occluded_iou': occluded_iou,
                    'occluded_dice': occluded_dice,
                    'occluded_precision': occluded_prec,
                    'occluded_recall': occluded_rec,
                    'visible_iou': visible_iou,
                    'inference_time_ms': inference_time * 1000
                })
                
                # Accumulate for averages
                total_amodal_iou += amodal_iou
                total_amodal_dice += amodal_dice
                # Note: occluded IoU/Dice already accumulated above (only for occluded samples)
                total_visible_iou += visible_iou
                total_inference_time += inference_time
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {sample_idx}: {e}")
                continue
    
    # Compute averages
    avg_amodal_iou = total_amodal_iou / num_samples
    avg_amodal_dice = total_amodal_dice / num_samples
    avg_visible_iou = total_visible_iou / num_samples
    avg_inference_time = total_inference_time / num_samples
    
    # Compute averages for occluded samples only
    if num_occluded_samples > 0:
        avg_occluded_iou = total_occluded_iou / num_occluded_samples
        avg_occluded_dice = total_occluded_dice / num_occluded_samples
    else:
        avg_occluded_iou = 0.0
        avg_occluded_dice = 0.0
    
    # Print summary
    print("\n" + "="*80)
    print(f"FULL RESOLUTION EVALUATION RESULTS - {dataset_name.upper()} DATASET")
    print("="*80)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Dataset:           {dataset_name}")
    print(f"   Total samples:     {num_samples}")
    print(f"   Successful:        {len(results)}")
    print(f"   Failed:            {num_samples - len(results)}")
    
    print(f"\n📊 Average Amodal Prediction Metrics:")
    print(f"   IoU:               {avg_amodal_iou:.4f}")
    print(f"   Dice:              {avg_amodal_dice:.4f}")
    
    print(f"\n📊 Average Occluded Region Metrics (only occluded samples):")
    print(f"   IoU:                             {avg_occluded_iou:.4f}")
    print(f"   Dice:                            {avg_occluded_dice:.4f}")
    print(f"   Occluded samples:                {num_occluded_samples}/{num_samples} ({100*num_occluded_samples/num_samples:.1f}%)")
    print(f"   Non-occluded samples (excluded): {num_samples - num_occluded_samples}/{num_samples} ({100*(num_samples-num_occluded_samples)/num_samples:.1f}%)")
    
    print(f"\n📊 Average Region-Specific IoU:")
    print(f"   Visible IoU:       {avg_visible_iou:.4f}")
    print(f"   Occluded IoU:      {avg_occluded_iou:.4f}")
    print(f"   Gap:               {abs(avg_visible_iou - avg_occluded_iou):.4f}")
    
    print(f"\n⏱️  Average Inference Time:")
    print(f"   Time per sample:   {avg_inference_time*1000:.2f} ms ({avg_inference_time:.4f} s)")
    print(f"   Total time:        {total_inference_time:.2f} s ({total_inference_time/60:.2f} min)")
    
    # Compute statistics by occlusion level
    if results:
        occlusion_rates = [r['occlusion_rate'] for r in results]
        
        # Categorize by occlusion level (exclude non-occluded samples from all categories)
        non_occ = [r for r in results if r['occlusion_rate'] <= 0.1]
        low_occ = [r for r in results if 0.1 < r['occlusion_rate'] < 20]
        # low1_occ = [r for r in results if 10 <= r['occlusion_rate'] < 20]
        med_occ = [r for r in results if 20 <= r['occlusion_rate'] < 50]
        high_occ = [r for r in results if r['occlusion_rate'] >= 50]
        
        print(f"\n📊 Performance by Occlusion Level:")
        if non_occ:
            avg_non = sum(r['occluded_iou'] for r in non_occ) / len(non_occ)
            print(f"   Non-occluded (≤0.1%): {len(non_occ):4d} samples, Occ IoU: {avg_non:.4f} (excluded from other metrics)")
        if low_occ:
            avg_low = sum(r['occluded_iou'] for r in low_occ) / len(low_occ)
            print(f"   Low (0.1-20%):        {len(low_occ):4d} samples, Occ IoU: {avg_low:.4f}")
        # if low1_occ:
        #     avg_low1 = sum(r['occluded_iou'] for r in low1_occ) / len(low1_occ)
        #     print(f"   Low (10-20%):         {len(low1_occ):4d} samples, Occ IoU: {avg_low1:.4f}")
        if med_occ:
            avg_med = sum(r['occluded_iou'] for r in med_occ) / len(med_occ)
            print(f"   Medium (20-50%):      {len(med_occ):4d} samples, Occ IoU: {avg_med:.4f}")
        if high_occ:
            avg_high = sum(r['occluded_iou'] for r in high_occ) / len(high_occ)
            print(f"   High (≥50%):          {len(high_occ):4d} samples, Occ IoU: {avg_high:.4f}")
    
    print("="*80)
    
    # Save per-sample results to CSV
    if save_per_sample and results:
        csv_path = save_dir / f'{dataset_name.lower()}_val_fullres_aligned_results.csv'
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n✅ Per-sample results saved to: {csv_path}")
    
    # Save summary statistics
    summary_path = save_dir / f'{dataset_name.lower()}_val_fullres_aligned_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"FULL RESOLUTION EVALUATION RESULTS - {dataset_name.upper()} DATASET\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {run_name if run_name else checkpoint_path}\n")
        f.write(f"Model Type: {model_type}-head\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n\n")
        
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Total samples:     {num_samples}\n")
        f.write(f"  Successful:        {len(results)}\n")
        f.write(f"  Failed:            {num_samples - len(results)}\n\n")
        
        f.write(f"Average Amodal Prediction Metrics:\n")
        f.write(f"  IoU:               {avg_amodal_iou:.4f}\n")
        f.write(f"  Dice:              {avg_amodal_dice:.4f}\n\n")
        
        f.write(f"Average Occluded Region Metrics (only occluded samples):\n")
        f.write(f"  IoU:                             {avg_occluded_iou:.4f}\n")
        f.write(f"  Dice:                            {avg_occluded_dice:.4f}\n")
        f.write(f"  Occluded samples:                {num_occluded_samples}/{num_samples} ({100*num_occluded_samples/num_samples:.1f}%)\n")
        f.write(f"  Non-occluded samples (excluded): {num_samples - num_occluded_samples}/{num_samples} ({100*(num_samples-num_occluded_samples)/num_samples:.1f}%)\n\n")
        
        f.write(f"Average Region-Specific IoU:\n")
        f.write(f"  Visible IoU:       {avg_visible_iou:.4f}\n")
        f.write(f"  Occluded IoU:      {avg_occluded_iou:.4f}\n")
        f.write(f"  Gap:               {abs(avg_visible_iou - avg_occluded_iou):.4f}\n\n")
        
        f.write(f"Average Inference Time:\n")
        f.write(f"  Time per sample:   {avg_inference_time*1000:.2f} ms\n")
        f.write(f"  Total time:        {total_inference_time:.2f} s ({total_inference_time/60:.2f} min)\n\n")
        
        if results:
            f.write(f"Performance by Occlusion Level:\n")
            if non_occ:
                avg_non = sum(r['occluded_iou'] for r in non_occ) / len(non_occ)
                f.write(f"  Non-occluded (≤0.1%): {len(non_occ):4d} samples, Occ IoU: {avg_non:.4f} (excluded from other metrics)\n")
            if low_occ:
                avg_low = sum(r['occluded_iou'] for r in low_occ) / len(low_occ)
                f.write(f"  Low (0.1-20%):        {len(low_occ):4d} samples, Occ IoU: {avg_low:.4f}\n")
            if med_occ:
                avg_med = sum(r['occluded_iou'] for r in med_occ) / len(med_occ)
                f.write(f"  Medium (20-50%):      {len(med_occ):4d} samples, Occ IoU: {avg_med:.4f}\n")
            if high_occ:
                avg_high = sum(r['occluded_iou'] for r in high_occ) / len(high_occ)
                f.write(f"  High (≥50%):          {len(high_occ):4d} samples, Occ IoU: {avg_high:.4f}\n")
    
    print(f"✅ Summary statistics saved to: {summary_path}")
    
    return {
        'avg_amodal_iou': avg_amodal_iou,
        'avg_occluded_iou': avg_occluded_iou,
        'avg_visible_iou': avg_visible_iou,
        'avg_inference_time': avg_inference_time,
        'results': results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test validation set at FULL RESOLUTION using align_raw_size (dataset auto-selected from config)')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the wandb run (e.g., peachy-frost-55)')
    parser.add_argument('--model-type', choices=['single', 'dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused'], default='single',
                        help='Model type: single-head, dual-head, dual-head-fused, dual-head-hybrid, or dual-head-hybrid-fused')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Direct path to checkpoint file (alternative to --model)')
    parser.add_argument('--config', type=str, default=None,
                        help='Direct path to configuration file (alternative to --model)')
    parser.add_argument('--dataset', type=str, default=None, choices=['cocoa', 'kins', 'vitasim'],
                        help='Override dataset selection (default: use dataset from config)')
    parser.add_argument('--no-save-per-sample', action='store_true',
                        help='Do not save per-sample results to CSV')
    
    args = parser.parse_args()
    
    # Test the entire validation set
    try:
        test_cocoa_fullres_aligned(
            run_name=args.model,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            model_type=args.model_type,
            save_per_sample=not args.no_save_per_sample,
            override_dataset=args.dataset
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
