"""
Generate paper-ready figures for IFAC publication.

Creates six 224x224 images with overlaid masks:
1. RGB image + visible mask (green overlay + green contour)
1b. RGB image + GT amodal mask (green overlay 0.3) + visible mask (blue contour)
2. RGB image + amodal mask (green overlay + green contour) + occluded mask (blue contour only)
3. RGB image + occluded mask (green overlay + green contour)
4. RGB image + amodal mask (green overlay + green contour)
5. RGB image only with 50% padding (wider context for comparison)

All datasets (COCOA, ViTASim, KINS):
- Figures 1-4: 20% padding (consistent tight crop)
- Figure 5: 50% padding (wider context)
"""

import torch
import numpy as np
import yaml
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cv2

from model import AmodalViT
from model_dual_head import DualHeadAmodalViT
from model_dual_head_fused import DualHeadFusedAmodalViT
from model_dual_head_hybrid import HybridDualHeadAmodalViT
from model_dual_head_hybrid_fused import HybridFusedDualHeadAmodalViT
from dataset import COCOADataset, ViTASimDataset, KINSDataset


def find_best_checkpoint(run_name, model_type='single'):
    """Find the best checkpoint file in the specified run directory"""
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
    
    checkpoint_files = list(run_dir.glob("best_model_epoch_*.pth"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    
    if len(checkpoint_files) > 1:
        print(f"⚠️  Warning: Multiple checkpoints found. Using the latest one.")
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    checkpoint_path = checkpoint_files[0]
    config_path = run_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return str(checkpoint_path), str(config_path), run_dir


def crop_image_with_padding(image, bbox, pad_factor=0.2, enlarge_coef=None):
    """
    Crop image around bounding box with padding.
    
    Args:
        image: PIL Image or numpy array
        bbox: [x, y, width, height] in COCO format
        pad_factor: Padding factor (0.2 = 20%) - used when enlarge_coef is None
        enlarge_coef: Enlargement coefficient (KINS style: 2.0 means 50% padding on each side)
    
    Returns:
        Cropped image (numpy array)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    H, W = image.shape[:2]
    x, y, w, h = bbox
    
    # Calculate padded bbox
    if enlarge_coef is not None:
        # KINS style: enlarge_coef = 2.0 means (2.0 - 1) / 2 = 0.5 (50% padding)
        pad_w = int(w * (enlarge_coef - 1) / 2)
        pad_h = int(h * (enlarge_coef - 1) / 2)
    else:
        # Standard style: pad_factor = 0.2 means 20% padding
        pad_w = int(w * pad_factor)
        pad_h = int(h * pad_factor)
    
    x1 = max(0, int(x) - pad_w)
    y1 = max(0, int(y) - pad_h)
    x2 = min(W, int(x + w) + pad_w)
    y2 = min(H, int(y + h) + pad_h)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2, y2)


def create_paper_figures(run_name=None, checkpoint_path=None, config_path=None, 
                         sample_idx=0, model_type='single', min_occluded_pixels=0, dataset_override=None):
    """
    Generate paper-ready figures for a single sample.
    
    Args:
        run_name: Name of the wandb run to load checkpoint from
        checkpoint_path: Direct path to checkpoint file
        config_path: Direct path to config file
        sample_idx: Index of the sample to visualize
        model_type: 'single', 'dual', 'dual-fused', 'dual-hybrid', or 'dual-hybrid-fused'
        min_occluded_pixels: Minimum number of pixels for an occluded region to be kept (0=keep all)
        dataset_override: Override dataset selection ('cocoa', 'vitasim', 'kins', or None=use config)
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
        save_dir = Path(checkpoint_path).parent
    
    # Create paper subfolder
    paper_dir = save_dir / "paper"
    paper_dir.mkdir(exist_ok=True)
    print(f"📁 Paper figures will be saved to: {paper_dir}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create validation dataset
    print("\n📂 Loading validation dataset...")
    dataset_choice = dataset_override if dataset_override else config['data'].get('datasets', 'cocoa')
    print(f"📊 Using dataset: {dataset_choice}")
    
    if dataset_choice == 'kins':
        kins_root = config['data']['kins']['root_dir']
        val_dataset = KINSDataset(
            root_dir=kins_root,
            annotation_file=str(Path(kins_root) / config['data']['kins']['val_annotations']),
            split='test',
            fusion_file=str(Path(kins_root) / config['data']['kins']['val_fusion']) if config['data']['kins'].get('val_fusion') else None,
            target_size=tuple(config['data']['target_size'])
        )
    elif dataset_choice in ['vitasim', 'both', 'all']:
        val_dataset = ViTASimDataset(
            root_dir=config['data']['vitasim']['root_dir'],
            annotation_file=config['data']['vitasim']['val_annotations'],
            split='val',
            fusion_file=config['data']['vitasim'].get('val_fusion', None),
            target_size=tuple(config['data']['target_size'])
        )
    else:
        val_dataset = COCOADataset(
            root_dir=config['data']['cocoa']['root_dir'],
            annotation_file=config['data']['cocoa']['val_annotations'],
            split='val',
            target_size=tuple(config['data']['target_size'])
        )
    
    if sample_idx >= len(val_dataset):
        print(f"❌ Error: Index {sample_idx} is out of range. Dataset has {len(val_dataset)} samples.")
        return
    
    # Get annotation info
    ann = val_dataset.annotations[sample_idx]
    image_info = val_dataset.images[ann['image_id']]
    filename = image_info['file_name']
    
    # Get bbox - KINS uses 'i_bbox', others use 'bbox'
    if dataset_choice == 'kins':
        bbox = ann['i_bbox']
    else:
        bbox = ann['bbox']
    
    print(f"\n{'='*60}")
    print(f"Creating paper figures for sample {sample_idx}")
    print(f"{'='*60}")
    
    # Load full resolution image
    if dataset_choice == 'kins':
        img_path = Path(config['data']['kins']['root_dir']) / 'testing' / 'image_2' / filename
        if not img_path.exists():
            img_path = Path(config['data']['kins']['root_dir']) / 'training' / 'image_2' / filename
        if not img_path.exists():
            img_path = Path(config['data']['kins']['root_dir']) / filename
    elif dataset_choice in ['vitasim', 'both', 'all']:
        img_path = Path(config['data']['vitasim']['root_dir']) / 'val2014' / filename
        if not img_path.exists():
            img_path = Path(config['data']['vitasim']['root_dir']) / filename
    else:
        img_path = Path(config['data']['cocoa']['root_dir']) / 'val2014' / filename
        if not img_path.exists():
            img_path = Path(config['data']['cocoa']['root_dir']) / filename
    
    pil_img = Image.open(img_path).convert('RGB')
    img_array = np.array(pil_img)
    
    # Decode GT masks - different field names for different datasets
    if dataset_choice == 'kins':
        # KINS uses 'a_segm' for amodal and 'i_segm' for inmodal (visible)
        H, W = image_info['height'], image_info['width']
        gt_amodal = val_dataset.decode_segmentation(ann['a_segm'], H, W)
        gt_visible = val_dataset.decode_segmentation(ann['i_segm'], H, W)
    elif dataset_choice in ['vitasim', 'both', 'all']:
        # ViTASim uses RLE format with 'segmentation' and 'visible_mask'
        gt_amodal = val_dataset.decode_rle_mask(ann['segmentation'])
        if 'visible_mask' in ann:
            gt_visible = val_dataset.decode_rle_mask(ann['visible_mask'])
        else:
            gt_visible = gt_amodal.copy()
    else:
        # COCOA uses polygon format with 'segmentation' and 'inmodal_seg' or 'visible_mask'
        gt_amodal = val_dataset.decode_segmentation(ann['segmentation'])
        if 'visible_mask' in ann:
            gt_visible = val_dataset.decode_segmentation(ann['visible_mask'])
        elif 'inmodal_seg' in ann:
            gt_visible = val_dataset.decode_segmentation(ann['inmodal_seg'])
        else:
            gt_visible = gt_amodal.copy()
    
    # Crop image around bbox with appropriate padding FOR VISUALIZATION
    # ALL datasets: figures 1-4 use 20% padding, figure 5 uses 50% padding
    img_cropped, crop_coords = crop_image_with_padding(img_array, bbox, pad_factor=0.2)
    # Figure 5 with wider context (50% padding)
    img_cropped_70, crop_coords_70 = crop_image_with_padding(img_array, bbox, pad_factor=0.5)
    
    # Manually crop and preprocess for inference with same 20% padding
    x1, y1, x2, y2 = crop_coords
    img_crop_for_inference = pil_img.crop((x1, y1, x2, y2))
    img_crop_for_inference = img_crop_for_inference.resize((224, 224), Image.BILINEAR)
    
    # Convert to tensor and normalize
    from torchvision import transforms
    img_tensor = transforms.ToTensor()(img_crop_for_inference)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
    
    # Get visible mask with same 20% padding
    visible_crop_for_inference = gt_visible[y1:y2, x1:x2]
    visible_crop_for_inference = cv2.resize(visible_crop_for_inference, (224, 224), interpolation=cv2.INTER_LINEAR)
    visible_tensor = torch.from_numpy(visible_crop_for_inference).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Run inference with aligned crops
    with torch.no_grad():
        if model_type in ['dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused']:
            preds = model(img_tensor, visible_tensor)
            pred_amodal_crop = preds['amodal']
            pred_occluded_crop = preds['occluded']
        else:
            pred_amodal_crop = model(img_tensor, visible_tensor)
            pred_occluded_crop = None
    
    # Resize predictions to 224x224
    pred_amodal_224 = torch.nn.functional.interpolate(
        pred_amodal_crop, size=(224, 224), mode='bilinear', align_corners=False
    )
    pred_amodal_224 = (pred_amodal_224.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    
    if pred_occluded_crop is not None:
        pred_occluded_224 = torch.nn.functional.interpolate(
            pred_occluded_crop, size=(224, 224), mode='bilinear', align_corners=False
        )
        pred_occluded_224 = (pred_occluded_224.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    else:
        # Derive occluded from amodal - visible
        visible_224 = torch.nn.functional.interpolate(
            visible_tensor, size=(224, 224), mode='bilinear', align_corners=False
        )
        visible_224 = (visible_224.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        pred_occluded_224 = pred_amodal_224 * (1 - visible_224)
    
    # Resize cropped image to 224x224
    img_cropped_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_cropped_resized_70 = cv2.resize(img_cropped_70, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Crop GT masks to match the cropped region (for comparison)
    visible_cropped = gt_visible[y1:y2, x1:x2]
    amodal_cropped = gt_amodal[y1:y2, x1:x2]
    
    # Resize masks to 224x224
    visible_cropped_224 = cv2.resize(visible_cropped, (224, 224), interpolation=cv2.INTER_NEAREST)
    amodal_cropped_224 = cv2.resize(amodal_cropped, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Compute occluded mask
    occluded_cropped_224 = ((amodal_cropped_224 > 0.5) & ~(visible_cropped_224 > 0.5)).astype(np.uint8)
    
    # Use predicted masks instead of GT
    visible_mask_to_use = (visible_cropped_224 > 0.5).astype(np.uint8)
    # amodal_mask_to_use_gt = (amodal_cropped_224 > 0.5).astype(np.uint8)
    amodal_mask_to_use = pred_amodal_224
    occluded_mask_to_use = pred_occluded_224
    
    # Filter small regions if threshold is set
    if min_occluded_pixels > 0:
        print(f"\n🔍 Filtering small regions (min pixels: {min_occluded_pixels})")
        
        # Filter AMODAL mask
        num_labels_amodal, labels_amodal, stats_amodal, _ = cv2.connectedComponentsWithStats(
            amodal_mask_to_use, connectivity=8
        )
        
        filtered_amodal = np.zeros_like(amodal_mask_to_use)
        kept_amodal = 0
        filtered_amodal_count = 0
        
        for label_id in range(1, num_labels_amodal):
            area = stats_amodal[label_id, cv2.CC_STAT_AREA]
            
            if area >= min_occluded_pixels:
                filtered_amodal[labels_amodal == label_id] = 1
                kept_amodal += 1
            else:
                filtered_amodal_count += 1
        
        amodal_mask_to_use = filtered_amodal
        print(f"   AMODAL: {num_labels_amodal - 1} regions → kept {kept_amodal}, filtered {filtered_amodal_count}")
        
        # Filter OCCLUDED mask
        num_labels_occluded, labels_occluded, stats_occluded, _ = cv2.connectedComponentsWithStats(
            occluded_mask_to_use, connectivity=8
        )
        
        filtered_occluded = np.zeros_like(occluded_mask_to_use)
        kept_occluded = 0
        filtered_occluded_count = 0
        
        for label_id in range(1, num_labels_occluded):
            area = stats_occluded[label_id, cv2.CC_STAT_AREA]
            
            if area >= min_occluded_pixels:
                filtered_occluded[labels_occluded == label_id] = 1
                kept_occluded += 1
            else:
                filtered_occluded_count += 1
        
        occluded_mask_to_use = filtered_occluded
        print(f"   OCCLUDED: {num_labels_occluded - 1} regions → kept {kept_occluded}, filtered {filtered_occluded_count}\n")
    else:
        # Still print debug info even without filtering
        print(f"\n🔍 Debug: Analyzing mask regions...")
        
        # Analyze AMODAL mask
        num_labels_amodal, labels_amodal, stats_amodal, _ = cv2.connectedComponentsWithStats(
            amodal_mask_to_use, connectivity=8
        )
        if num_labels_amodal > 1:
            print(f"\n  � AMODAL MASK: {num_labels_amodal - 1} separated regions:")
            for label_id in range(1, num_labels_amodal):
                area = stats_amodal[label_id, cv2.CC_STAT_AREA]
                print(f"     - Region {label_id}: {area} pixels")
        
        # Analyze OCCLUDED mask
        num_labels_occluded, labels_occluded, stats_occluded, _ = cv2.connectedComponentsWithStats(
            occluded_mask_to_use, connectivity=8
        )
        if num_labels_occluded > 1:
            print(f"\n  📍 OCCLUDED MASK: {num_labels_occluded - 1} separated regions:")
            for label_id in range(1, num_labels_occluded):
                area = stats_occluded[label_id, cv2.CC_STAT_AREA]
                print(f"     - Region {label_id}: {area} pixels")
        
        print()
    
    print(f"📐 Image size: {img_cropped_resized.shape}")
    print(f"📐 Visible mask size: {visible_mask_to_use.shape}")
    print(f"📐 Amodal mask size: {amodal_mask_to_use.shape}")
    print(f"📐 Occluded mask size: {occluded_mask_to_use.shape}")
    
    # Determine suffix based on model type
    model_suffix = "_SH" if args.model_type == "single" else "_DH"
    
    # === Figure 1: RGB + Visible Mask (green overlay + green contour) ===
    print("\n🎨 Creating Figure 1: RGB + Visible Mask...")
    
    fig1 = plt.figure(figsize=(224/100, 224/100), dpi=300)
    ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
    ax1.set_axis_off()
    fig1.add_axes(ax1)
    
    # Create composite image with green overlay
    img_with_overlay = img_cropped_resized.copy().astype(np.float32) / 255.0
    
    # Add green overlay where mask exists
    green_overlay = np.zeros_like(img_with_overlay)
    green_overlay[:, :, 1] = visible_mask_to_use  # Green channel
    img_with_overlay = img_with_overlay * 0.7 + green_overlay * 0.3
    
    # Draw green contours
    img_with_contours = (img_with_overlay * 255).astype(np.uint8)
    contours_visible, _ = cv2.findContours(visible_mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_with_contours, contours_visible, -1, (0, 0, 255), 2)  # Red in BGR
    
    # Display
    ax1.imshow(img_with_contours)
    
    # Save Figure 1
    fig1_path = paper_dir / f'sample_{sample_idx}_fig1_visible{model_suffix}.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    print(f"✅ Figure 1 saved: {fig1_path}")
    
    # === Figure 1b: RGB + GT Amodal Mask (green overlay 0.3) + Visible Mask (blue contour) ===
    print("🎨 Creating Figure 1b: RGB + GT Amodal (green overlay) + Visible (blue contour)...")
    
    fig1b = plt.figure(figsize=(224/100, 224/100), dpi=300)
    ax1b = plt.Axes(fig1b, [0., 0., 1., 1.])
    ax1b.set_axis_off()
    fig1b.add_axes(ax1b)
    
    # Create composite image with green overlay for GT amodal (30% overlay)
    img_with_overlay1b = img_cropped_resized.copy().astype(np.float32) / 255.0
    
    # Add green overlay where GT amodal mask exists (reduced to 0.3)
    green_overlay1b = np.zeros_like(img_with_overlay1b)
    gt_amodal_mask_to_use = (amodal_cropped_224 > 0.5).astype(np.uint8)
    green_overlay1b[:, :, 1] = gt_amodal_mask_to_use  # Green channel
    img_with_overlay1b = img_with_overlay1b * 0.7 + green_overlay1b * 0.3
    
    # Draw blue contours for visible mask
    img_with_contours1b = (img_with_overlay1b * 255).astype(np.uint8)
    contours_visible_1b, _ = cv2.findContours(gt_amodal_mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_with_contours1b, contours_visible_1b, -1, (255, 0, 255), 2)  # Green in BGR
    
    # Display
    ax1b.imshow(img_with_contours1b)
    
    # Save Figure 1b
    fig1b_path = paper_dir / f'sample_{sample_idx}_fig1b_gt_amodal_visible{model_suffix}.png'
    plt.savefig(fig1b_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig1b)
    print(f"✅ Figure 1b saved: {fig1b_path}")
    
    # === Figure 2: RGB + Amodal Mask (green overlay + green contour) + Occluded (blue contour only) ===
    print("🎨 Creating Figure 2: RGB + Amodal + Occluded Masks...")
    
    fig2 = plt.figure(figsize=(224/100, 224/100), dpi=300)
    ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
    ax2.set_axis_off()
    fig2.add_axes(ax2)
    
    # Create composite image with green overlay for amodal
    img_with_overlay2 = img_cropped_resized.copy().astype(np.float32) / 255.0
    
    # Add green overlay where amodal mask exists
    green_overlay2 = np.zeros_like(img_with_overlay2)
    green_overlay2[:, :, 1] = amodal_mask_to_use  # Green channel
    img_with_overlay2 = img_with_overlay2 * 0.7 + green_overlay2 * 0.3
    
    # Draw contours
    img_with_contours2 = (img_with_overlay2 * 255).astype(np.uint8)
    contours_amodal, _ = cv2.findContours(amodal_mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_occluded, _ = cv2.findContours(occluded_mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img_with_contours2, contours_amodal, -1, (0, 255, 0), 2)  # Green contour for amodal (BGR)
    cv2.drawContours(img_with_contours2, contours_occluded, -1, (255, 0, 0), 2)  # Blue contour for occluded (BGR)
    
    # Display
    ax2.imshow(img_with_contours2)
    
    # Save Figure 2
    fig2_path = paper_dir / f'sample_{sample_idx}_fig2_amodal_occluded{model_suffix}.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    print(f"✅ Figure 2 saved: {fig2_path}")
    
    # === Figure 3: RGB + Occluded Mask (green overlay + green contour) ===
    print("🎨 Creating Figure 3: RGB + Occluded Mask...")
    
    fig3 = plt.figure(figsize=(224/100, 224/100), dpi=300)
    ax3 = plt.Axes(fig3, [0., 0., 1., 1.])
    ax3.set_axis_off()
    fig3.add_axes(ax3)
    
    # Create composite image with green overlay for occluded
    img_with_overlay3 = img_cropped_resized.copy().astype(np.float32) / 255.0
    
    # Add green overlay where occluded mask exists
    green_overlay3 = np.zeros_like(img_with_overlay3)
    green_overlay3[:, :, 1] = occluded_mask_to_use  # Green channel
    img_with_overlay3 = img_with_overlay3 * 0.7 + green_overlay3 * 0.3
    
    # Draw green contours
    img_with_contours3 = (img_with_overlay3 * 255).astype(np.uint8)
    contours_occluded_only, _ = cv2.findContours(occluded_mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_with_contours3, contours_occluded_only, -1, (255, 0, 0), 2)  # Red in BGR
    
    # Display
    ax3.imshow(img_with_contours3)
    
    # Save Figure 3
    fig3_path = paper_dir / f'sample_{sample_idx}_fig3_occluded{model_suffix}.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig3)
    print(f"✅ Figure 3 saved: {fig3_path}")
    
    # === Figure 4: RGB + Amodal Mask (green overlay + green contour) ===
    print("🎨 Creating Figure 4: RGB + Amodal Mask...")
    
    fig4 = plt.figure(figsize=(224/100, 224/100), dpi=300)
    ax4 = plt.Axes(fig4, [0., 0., 1., 1.])
    ax4.set_axis_off()
    fig4.add_axes(ax4)
    
    # Create composite image with green overlay for amodal
    img_with_overlay4 = img_cropped_resized.copy().astype(np.float32) / 255.0
    
    # Add green overlay where amodal mask exists
    green_overlay4 = np.zeros_like(img_with_overlay4)
    green_overlay4[:, :, 1] = amodal_mask_to_use  # Green channel
    img_with_overlay4 = img_with_overlay4 * 0.7 + green_overlay4 * 0.3
    
    # Draw green contours
    img_with_contours4 = (img_with_overlay4 * 255).astype(np.uint8)
    contours_amodal_only, _ = cv2.findContours(amodal_mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_with_contours4, contours_amodal_only, -1, (0, 255, 0), 2)  # Green in BGR
    
    # Display
    ax4.imshow(img_with_contours4)
    
    # Save Figure 4
    fig4_path = paper_dir / f'sample_{sample_idx}_fig4_amodal{model_suffix}.png'
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig4)
    print(f"✅ Figure 4 saved: {fig4_path}")
    
    # === Figure 5: RGB image only with 50% padding ===
    print("🎨 Creating Figure 5: RGB with 50% padding (wider context)...")
    
    fig5 = plt.figure(figsize=(224/100, 224/100), dpi=300)
    ax5 = plt.Axes(fig5, [0., 0., 1., 1.])
    ax5.set_axis_off()
    fig5.add_axes(ax5)
    
    # Display RGB image with wider context
    ax5.imshow(img_cropped_resized_70)
    
    # Save Figure 5
    fig5_path = paper_dir / f'sample_{sample_idx}_fig5_rgb_context{model_suffix}.png'
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig5)
    print(f"✅ Figure 5 saved: {fig5_path}")
    
    print("\n" + "="*60)
    print("✅ Paper figures generated successfully!")
    print(f"   Resolution: 224x224")
    print(f"   DPI: 300")
    print(f"   Location: {paper_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate paper-ready figures for IFAC publication')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the wandb run (e.g., peachy-frost-55)')
    parser.add_argument('--model-type', choices=['single', 'dual', 'dual-fused', 'dual-hybrid', 'dual-hybrid-fused'], 
                        default='single',
                        help='Model type: single-head or dual-head')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Direct path to checkpoint file (alternative to --model)')
    parser.add_argument('--config', type=str, default=None,
                        help='Direct path to configuration file (alternative to --model)')
    parser.add_argument('--index', type=int, required=True,
                        help='Sample index to visualize')
    parser.add_argument('--min-occluded-pixels', type=int, default=0,
                        help='Minimum number of pixels for regions (filters both amodal and occluded, default: 0=keep all)')
    parser.add_argument('--dataset', choices=['cocoa', 'vitasim', 'kins'], default=None,
                        help='Override dataset selection (default: use config file setting)')
    
    args = parser.parse_args()
    
    try:
        create_paper_figures(
            run_name=args.model,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            sample_idx=args.index,
            model_type=args.model_type,
            min_occluded_pixels=args.min_occluded_pixels,
            dataset_override=args.dataset
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
