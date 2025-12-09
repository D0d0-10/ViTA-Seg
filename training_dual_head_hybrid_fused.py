"""
Training script for Hybrid+Fused Dual-Head ViTA-Seg model.

This script trains the ultimate architecture combining:
1. Shared early decoder (common low-level features)
2. Task-specific late decoders (specialized high-level features)
3. Fusion of occluded into amodal (cross-task collaboration)

Expected improvements:
- Amodal IoU: +2-3% over standard dual-head
- Occluded IoU: +3-4% over standard dual-head
- Best of both worlds: specialization + collaboration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import wandb
from tqdm import tqdm
from pathlib import Path
import yaml
import os
import argparse
from datetime import datetime
import numpy as np
from typing import Dict, Optional
import random

from model_dual_head_hybrid_fused import HybridFusedDualHeadAmodalViT, HybridFusedDualHeadLoss
from dataset import ViTASimDataset, COCOADataset, KINSDataset, CombinedDataset

# Try to import evaluation metrics
try:
    from evaluation_metrics import compute_c2f_compatible_metrics
    EVAL_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import evaluation metrics: {e}")
    EVAL_METRICS_AVAILABLE = False


def safe_collate_fn(batch):
    """Custom collate function that handles missing fields gracefully"""
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
    
    result = {}
    
    for key in all_keys:
        values = []
        for sample in batch:
            if key in sample:
                values.append(sample[key])
            else:
                # Provide default values for missing keys
                if key == 'occlude_rate':
                    values.append(0.0)
                elif key == 'iscrowd':
                    values.append(0)
                elif key == 'area':
                    values.append(0.0)
                elif key == 'dataset':
                    values.append('unknown')
                else:
                    values.append(None)
        
        # Filter out None values and collate
        valid_values = [v for v in values if v is not None]
        if valid_values:
            try:
                if isinstance(valid_values[0], torch.Tensor):
                    result[key] = torch.stack(valid_values)
                elif isinstance(valid_values[0], (int, float)):
                    result[key] = torch.tensor(valid_values)
                elif isinstance(valid_values[0], str):
                    result[key] = valid_values
                else:
                    result[key] = valid_values
            except Exception as e:
                print(f"Warning: Could not collate field '{key}': {e}")
                result[key] = valid_values
    
    return result


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(dataset_name: str, root_dir: str, split: str, enlarge_coef: float = 1.0):
    """
    Create dataset based on name.
    
    Args:
        dataset_name: 'ViTASim', 'COCOA', or 'KINS'
        root_dir: Root directory for the dataset
        split: 'train' or 'val'
        enlarge_coef: Enlargement coefficient for bounding boxes (KINS only)
        
    Returns:
        Dataset instance
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == 'vitasim':
        # ViTASim dataset
        if split == 'train':
            annotation_file = os.path.join(root_dir, 'ViTASim_amodal_train2014_with_classes.json')
            fusion_file = os.path.join(root_dir, 'fusion_train.pkl')
        else:
            annotation_file = os.path.join(root_dir, 'ViTASim_amodal_val2014_with_classes.json')
            fusion_file = os.path.join(root_dir, 'fusion_test.pkl')
        
        return ViTASimDataset(
            root_dir=root_dir,
            annotation_file=annotation_file,
            split=split,
            fusion_file=fusion_file if os.path.exists(fusion_file) else None,
            target_size=(224, 224)
        )
    
    elif dataset_name_lower == 'cocoa':
        # COCOA dataset
        if split == 'train':
            annotation_file = os.path.join(root_dir, 'COCOA_train_with_classes.json')
        else:
            annotation_file = os.path.join(root_dir, 'COCOA_val_with_classes.json')
        
        return COCOADataset(
            root_dir=root_dir,
            annotation_file=annotation_file,
            split=split,
            target_size=(224, 224)
        )
    
    elif dataset_name_lower == 'kins':
        # KINS dataset
        if split == 'train':
            annotation_file = os.path.join(root_dir, 'instances_train.json')
            fusion_file = os.path.join(root_dir, 'fusion_train.pkl')
        else:
            annotation_file = os.path.join(root_dir, 'update_val_2020.json')
            fusion_file = os.path.join(root_dir, 'fusion_test.pkl')
        
        return KINSDataset(
            root_dir=root_dir,
            annotation_file=annotation_file,
            split=split,
            fusion_file=fusion_file if os.path.exists(fusion_file) else None,
            target_size=(224, 224),
            enlarge_coef=enlarge_coef
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'ViTASim', 'COCOA', or 'KINS'")


def create_datasets(config: dict):
    """Create train and validation datasets from config"""
    target_size = tuple(config['data']['target_size'])
    datasets_to_use = config['data']['datasets']
    
    train_datasets = []
    val_datasets = []
    
    # Create ViTASim datasets if specified
    if datasets_to_use in ['vitasim', 'both', 'all']:
        print("Loading ViTASim dataset...")
        vitasim_config = config['data']['vitasim']
        
        vitasim_train = ViTASimDataset(
            root_dir=vitasim_config['root_dir'],
            annotation_file=vitasim_config['train_annotations'],
            split='train',
            fusion_file=vitasim_config.get('train_fusion', None),
            target_size=target_size
        )
        
        vitasim_val = ViTASimDataset(
            root_dir=vitasim_config['root_dir'],
            annotation_file=vitasim_config['val_annotations'],
            split='val',
            fusion_file=vitasim_config.get('val_fusion', None),
            target_size=target_size
        )
        
        train_datasets.append(vitasim_train)
        val_datasets.append(vitasim_val)
    
    # Create COCOA datasets if specified
    if datasets_to_use in ['cocoa', 'both', 'all']:
        print("Loading COCOA dataset...")
        cocoa_config = config['data']['cocoa']
        
        if os.path.exists(cocoa_config['train_annotations']):
            cocoa_train = COCOADataset(
                root_dir=cocoa_config['root_dir'],
                annotation_file=cocoa_config['train_annotations'],
                split='train',
                target_size=target_size
            )
            train_datasets.append(cocoa_train)
        else:
            print(f"Warning: COCOA train annotations not found at {cocoa_config['train_annotations']}")
        
        if os.path.exists(cocoa_config['val_annotations']):
            cocoa_val = COCOADataset(
                root_dir=cocoa_config['root_dir'],
                annotation_file=cocoa_config['val_annotations'],
                split='val',
                target_size=target_size
            )
            val_datasets.append(cocoa_val)
        else:
            print(f"Warning: COCOA val annotations not found at {cocoa_config['val_annotations']}")
    
    # Create KINS datasets if specified
    if datasets_to_use in ['kins', 'all']:
        print("Loading KINS dataset...")
        kins_config = config['data']['kins']
        enlarge_coef = kins_config.get('enlarge_coef', 2.0)
        
        train_ann_path = os.path.join(kins_config['root_dir'], kins_config['train_annotations'])
        val_ann_path = os.path.join(kins_config['root_dir'], kins_config['val_annotations'])
        
        if os.path.exists(train_ann_path):
            train_fusion = None
            if 'train_fusion' in kins_config and kins_config['train_fusion']:
                train_fusion = os.path.join(kins_config['root_dir'], kins_config['train_fusion'])
                if not os.path.exists(train_fusion):
                    train_fusion = None
            
            kins_train = KINSDataset(
                root_dir=kins_config['root_dir'],
                annotation_file=train_ann_path,
                split='train',
                fusion_file=train_fusion,
                target_size=target_size,
                enlarge_coef=enlarge_coef
            )
            train_datasets.append(kins_train)
        
        if os.path.exists(val_ann_path):
            val_fusion = None
            if 'val_fusion' in kins_config and kins_config['val_fusion']:
                val_fusion = os.path.join(kins_config['root_dir'], kins_config['val_fusion'])
                if not os.path.exists(val_fusion):
                    val_fusion = None
            
            kins_val = KINSDataset(
                root_dir=kins_config['root_dir'],
                annotation_file=val_ann_path,
                split='test',
                fusion_file=val_fusion,
                target_size=target_size,
                enlarge_coef=enlarge_coef
            )
            val_datasets.append(kins_val)
    
    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = CombinedDataset(train_datasets)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        raise ValueError("No valid datasets found!")
    
    if len(val_datasets) > 1:
        val_dataset = CombinedDataset(val_datasets)
    elif len(val_datasets) == 1:
        val_dataset = val_datasets[0]
    else:
        raise ValueError("No valid validation datasets found!")
    
    return train_dataset, val_dataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU metric"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if pred.sum() == 0 and target.sum() == 0 else 0.0
    
    return (intersection / union).item()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set with comprehensive metrics.
    
    Returns metrics compatible with C2F coarse-to-fine segmentation.
    """
    model.eval()
    
    total_loss = 0.0
    total_amodal_loss = 0.0
    total_occluded_loss = 0.0
    
    # C2F-compatible metrics
    amodal_ious = []
    occluded_ious = []
    visible_ious = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            visible_masks = batch['visible_mask'].to(device)
            amodal_masks = batch['amodal_mask'].to(device)
            
            # Forward pass
            predictions = model(images, visible_masks)
            
            # Compute loss
            losses = criterion(predictions, amodal_masks, visible_masks)
            
            total_loss += losses['total'].item()
            total_amodal_loss += losses['amodal'].item()
            total_occluded_loss += losses['occluded'].item()
            
            # Compute IoU metrics
            batch_size = images.size(0)
            for i in range(batch_size):
                # Amodal IoU (fused prediction)
                amodal_iou = compute_iou(
                    predictions['amodal'][i],
                    amodal_masks[i]
                )
                amodal_ious.append(amodal_iou)
                
                # Visible IoU
                visible_iou = compute_iou(
                    visible_masks[i],
                    visible_masks[i]  # Perfect match for reference
                )
                visible_ious.append(visible_iou)
                
                # Occluded IoU
                amodal_bool = amodal_masks[i] > 0.5
                visible_bool = visible_masks[i] < 0.5
                occluded_target = (amodal_bool & visible_bool).float()
                
                occluded_iou = compute_iou(
                    predictions['occluded'][i],
                    occluded_target
                )
                occluded_ious.append(occluded_iou)
    
    num_batches = len(dataloader)
    
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_amodal_loss': total_amodal_loss / num_batches,
        'val_occluded_loss': total_occluded_loss / num_batches,
        'val_amodal_iou': np.mean(amodal_ious),
        'val_occluded_iou': np.mean(occluded_ious),
        'val_visible_iou': np.mean(visible_ious)
    }
    
    return metrics


def get_custom_run_name(config: dict, occluded_weight_override: Optional[float] = None) -> str:
    """
    Generate a custom run name for WandB based on configuration.
    
    Args:
        config: Configuration dictionary
        occluded_weight_override: Optional override for occluded weight
        
    Returns:
        Custom run name string
    """
    # Use override if provided, otherwise use config value
    occluded_weight = occluded_weight_override if occluded_weight_override is not None else config['training']['loss_weights']['occluded']
    
    # Format: hybridfused_occ_{weight:.2f}-lr_{lr}-bs_{bs}-{timestamp}
    lr = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    run_name = f"hybridfused_occ_{occluded_weight:.2f}-lr_{lr}-bs_{batch_size}-{timestamp}"
    
    return run_name


def train_hybrid_fused_model(
    config_path: str,
    resume_from: Optional[str] = None,
    occluded_weight_override: Optional[float] = None,
    reset_epoch: bool = False
):
    """
    Train the hybrid+fused dual-head model.
    
    Args:
        config_path: Path to configuration file
        resume_from: Optional path to checkpoint to resume from
        occluded_weight_override: Optional override for occluded loss weight
        reset_epoch: If True, reset epoch counter to 0 when resuming (useful for new dataset)
    """
    # Enable CUDA error debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Apply occluded weight override if provided
    if occluded_weight_override is not None:
        config['training']['loss_weights']['occluded'] = occluded_weight_override
        print(f"✅ Overriding occluded weight: {occluded_weight_override}")
    
    # Initialize wandb with custom run name
    custom_run_name = get_custom_run_name(config, occluded_weight_override)
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity'),
        config=config,
        name=custom_run_name
    )
    
    print(f"WandB run name: {custom_run_name}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_datasets(config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True,
        collate_fn=safe_collate_fn
    )
    
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val dataset: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Create model
    print("\nCreating hybrid+fused dual-head model...")
    model = HybridFusedDualHeadAmodalViT(**config['model']).to(device)
    
    # Print parameter counts
    trainable_params = model.get_trainable_params()
    print(f"\nTrainable parameters:")
    for name, count in trainable_params.items():
        print(f"  {name}: {count:,}")
    
    # Create loss function with weights
    criterion = HybridFusedDualHeadLoss(weights=config['training']['loss_weights'])
    
    # Create optimizer
    optimizer_config = config['training'].get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw')
    
    lr_value = float(config['training']['learning_rate'])
    wd_value = float(config['training']['weight_decay'])
    
    if optimizer_type.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=lr_value, weight_decay=wd_value)
    else:
        optimizer = AdamW(model.parameters(), lr=lr_value, weight_decay=wd_value)
    
    # Learning rate scheduler
    scheduler_config = config['training'].get('scheduler', {})
    if scheduler_config.get('type') == 'warmup_linear':
        warmup_iters = scheduler_config.get('warmup_iters', 1000)
        total_iters = config['training']['epochs'] * len(train_loader)
        
        def lr_lambda(current_step):
            if current_step < warmup_iters:
                return float(current_step) / float(max(1, warmup_iters))
            progress = float(current_step - warmup_iters) / float(max(1, total_iters - warmup_iters))
            return max(0.0, 1.0 - progress)
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=float(config['training']['min_lr'])
        )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_iou = 0.0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if reset_epoch:
            # Reset epoch counter (useful when training on a new dataset)
            start_epoch = 0
            best_val_iou = 0.0
            best_val_loss = float('inf')
            print(f"✅ Loaded model weights only (epoch counter reset)")
            print(f"   Starting fresh training from epoch 0 on new dataset")
        else:
            # Continue from checkpoint epoch
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_iou = checkpoint.get('best_val_iou', 0.0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"✅ Resumed from epoch {start_epoch}, best IoU: {best_val_iou:.4f}")
    
    # Get wandb run name and create best model directory
    run_name = wandb.run.name if wandb.run else "unknown_run"
    best_model_dir = Path(f"best-models-dual-head-hybrid-fused/{run_name}")
    best_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Best model will be saved to: {best_model_dir}")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    global_step = 0
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        
        epoch_loss = 0.0
        epoch_amodal_loss = 0.0
        epoch_occluded_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device, non_blocking=True)
            visible_masks = batch['visible_mask'].to(device, non_blocking=True)
            amodal_masks = batch['amodal_mask'].to(device, non_blocking=True)
            
            # Forward pass
            predictions = model(images, visible_masks)
            
            # Compute loss
            losses = criterion(predictions, amodal_masks, visible_masks)
            loss = losses['total']
            
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config['training']['gradient_clip_norm']
                )
                optimizer.step()
                optimizer.zero_grad()
            
            # Handle scheduler step for warmup_linear
            if scheduler_config.get('type') == 'warmup_linear':
                scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item() * (accumulation_steps if accumulation_steps > 1 else 1)
            epoch_amodal_loss += losses['amodal'].item()
            epoch_occluded_loss += losses['occluded'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'amodal': f"{losses['amodal'].item():.4f}",
                'occluded': f"{losses['occluded'].item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb (step-level, every 10 steps)
            if global_step % 10 == 0:
                wandb.log({
                    'train/step_loss': loss.item(),
                    'train/step_amodal_loss': losses['amodal'].item(),
                    'train/step_occluded_loss': losses['occluded'].item(),
                    'train/step_amodal_bce': losses['amodal_bce'].item(),
                    'train/step_amodal_dice': losses['amodal_dice'].item(),
                    'train/step_occluded_bce': losses['occluded_bce'].item(),
                    'train/step_occluded_dice': losses['occluded_dice'].item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'global_step': global_step
                }, step=global_step)
            
            global_step += 1
        
        # Epoch metrics
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        epoch_amodal_loss /= num_batches
        epoch_occluded_loss /= num_batches
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Log epoch-level metrics
        wandb.log({
            'epoch': epoch + 1,
            'train/epoch_loss': epoch_loss,
            'train/epoch_amodal_loss': epoch_amodal_loss,
            'train/epoch_occluded_loss': epoch_occluded_loss,
            'val/loss': val_metrics['val_loss'],
            'val/amodal_loss': val_metrics['val_amodal_loss'],
            'val/occluded_loss': val_metrics['val_occluded_loss'],
            'val/amodal_iou': val_metrics['val_amodal_iou'],
            'val/occluded_iou': val_metrics['val_occluded_iou'],
            'val/visible_iou': val_metrics['val_visible_iou'],
            'train/learning_rate': optimizer.param_groups[0]['lr']
        }, step=global_step)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Amodal IoU: {val_metrics['val_amodal_iou']:.4f}")
        print(f"  Val Occluded IoU: {val_metrics['val_occluded_iou']:.4f}")
        
        # Save best model based on validation amodal IoU or loss
        if val_metrics['val_amodal_iou'] > best_val_iou or val_metrics['val_loss'] < best_val_loss:
            if val_metrics['val_amodal_iou'] > best_val_iou:
                best_val_iou = val_metrics['val_amodal_iou']
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
            
            # Remove old checkpoints
            for old_file in best_model_dir.glob("best_model_epoch_*.pth"):
                old_file.unlink()
            
            checkpoint_path = best_model_dir / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_metrics['val_loss'],
                'best_val_iou': best_val_iou,
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            
            # Save config
            config_path = best_model_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"  ✅ New best model saved: {checkpoint_path}")
            print(f"  Best Val Amodal IoU: {best_val_iou:.4f}")
            print(f"  Best Val Loss: {best_val_loss:.4f}")
        
        # Update learning rate (only for non-warmup schedulers)
        if scheduler_config.get('type', 'cosine') != 'warmup_linear':
            scheduler.step()
        
        print("-" * 70)
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation amodal IoU: {best_val_iou:.4f}")
    print("="*70)
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid+Fused Dual-Head ViTA-Seg")
    parser.add_argument(
        '--config',
        type=str,
        default='config_dual_head_hybrid_fused.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--occluded-weight',
        type=float,
        default=None,
        help='Override occluded loss weight (useful for hyperparameter sweeps)'
    )
    parser.add_argument(
        '--reset-epoch',
        action='store_true',
        help='Reset epoch counter to 0 when resuming (useful for training on new dataset)'
    )
    
    args = parser.parse_args()
    
    train_hybrid_fused_model(
        args.config,
        resume_from=args.resume,
        occluded_weight_override=args.occluded_weight,
        reset_epoch=args.reset_epoch
    )
