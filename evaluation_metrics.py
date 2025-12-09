"""
Standardized evaluation metrics for fair comparison between ViTA-Seg and C2F-Seg
This module implements the same metrics used in C2F-Seg paper for consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

# Simple implementation without sklearn dependency
def simple_average_precision(binary_labels: np.ndarray, scores: np.ndarray) -> float:
    """Simple Average Precision calculation without sklearn"""
    if len(binary_labels) == 0 or binary_labels.sum() == 0:
        return 0.0
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = binary_labels[sorted_indices]
    
    # Calculate precision at each threshold
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1 - sorted_labels)
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives[-1] + 1e-8)
    
    # Calculate AP using trapezoidal rule (correct version)
    # Add precision=0 at recall=0 for correct AP calculation
    recall_with_zero = np.concatenate(([0], recall))
    precision_with_zero = np.concatenate(([0], precision))
    
    # Compute differences in recall
    recall_diff = np.diff(recall_with_zero)
    
    # AP is sum of precision at each recall level weighted by recall change
    # Use precision[i] for interval [recall[i-1], recall[i]]
    ap = np.sum(precision_with_zero[1:] * recall_diff)
    
    return ap


class AmodalSegmentationMetrics:
    """
    Standardized metrics for amodal segmentation evaluation
    Compatible with C2F-Seg evaluation protocol
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Args:
            iou_thresholds: IoU thresholds for AP calculation (default: [0.5, 0.55, ..., 0.95])
        """
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
    
    @staticmethod
    def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU) using logical operations
        
        Args:
            pred: Predicted mask [B, 1, H, W] or [B, H, W]
            target: Ground truth mask [B, 1, H, W] or [B, H, W]
            threshold: Threshold for binarizing predictions
            
        Returns:
            IoU scores [B]
        """
        # Ensure same shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Binarize predictions (convert to boolean for logical operations)
        pred_binary = (pred > threshold)
        target_binary = (target > threshold)
        
        # Compute IoU using logical operations
        # Intersection: logical AND
        intersection = (pred_binary & target_binary).sum(dim=[1, 2]).float()
        # Union: logical OR
        union = (pred_binary | target_binary).sum(dim=[1, 2]).float()
        
        # Avoid division by zero
        iou = intersection / (union + 1e-8)
        
        return iou
    
    @staticmethod
    def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute Dice coefficient (F1-score)
        
        Args:
            pred: Predicted mask [B, 1, H, W] or [B, H, W]
            target: Ground truth mask [B, 1, H, W] or [B, H, W]
            threshold: Threshold for binarizing predictions
            
        Returns:
            Dice scores [B]
        """
        # Ensure same shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Binarize predictions
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        # Compute Dice
        intersection = (pred_binary * target_binary).sum(dim=[1, 2])
        dice = (2 * intersection) / (pred_binary.sum(dim=[1, 2]) + target_binary.sum(dim=[1, 2]) + 1e-8)
        
        return dice
    
    @staticmethod
    def compute_precision_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Precision and Recall
        
        Args:
            pred: Predicted mask [B, 1, H, W] or [B, H, W]
            target: Ground truth mask [B, 1, H, W] or [B, H, W]
            threshold: Threshold for binarizing predictions
            
        Returns:
            Tuple of (precision, recall) [B]
        """
        # Ensure same shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Binarize predictions
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        # Compute metrics
        tp = (pred_binary * target_binary).sum(dim=[1, 2])
        fp = (pred_binary * (1 - target_binary)).sum(dim=[1, 2])
        fn = ((1 - pred_binary) * target_binary).sum(dim=[1, 2])
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return precision, recall
    
    def compute_ap_at_iou(self, pred: torch.Tensor, target: torch.Tensor, iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision at specific IoU threshold
        
        Args:
            pred: Predicted masks [B, 1, H, W] or [B, H, W] - SOFT predictions [0, 1]
            target: Ground truth masks [B, 1, H, W] or [B, H, W]
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Average Precision score
            
        Note: For semantic/instance segmentation without detection scores,
        we compute IoU at different prediction thresholds and use those
        as our "detections" at different confidence levels.
        """
        # Ensure same shape
        if pred.dim() == 4:
            pred_soft = pred.squeeze(1)
        else:
            pred_soft = pred
        if target.dim() == 4:
            target_binary = target.squeeze(1)
        else:
            target_binary = target
        
        # Binarize target
        target_binary = (target_binary > 0.5).float()
        
        # For each sample, try multiple thresholds and compute IoU
        # This gives us multiple "detections" at different confidence levels
        batch_size = pred_soft.shape[0]
        thresholds = np.linspace(0.1, 0.9, 9)  # 9 different thresholds
        
        all_ious = []
        all_confidences = []
        
        for b in range(batch_size):
            pred_sample = pred_soft[b]
            target_sample = target_binary[b]
            
            for thresh in thresholds:
                # Binarize at this threshold
                pred_binary = (pred_sample > thresh).float()
                
                # Compute IoU
                intersection = (pred_binary * target_sample).sum()
                union = pred_binary.sum() + target_sample.sum() - intersection
                iou = (intersection / (union + 1e-8)).item()
                
                all_ious.append(iou)
                all_confidences.append(thresh)  # Use threshold as confidence
        
        # Convert to numpy
        all_ious = np.array(all_ious)
        all_confidences = np.array(all_confidences)
        
        # Binary labels: 1 if IoU >= threshold, 0 otherwise
        binary_labels = (all_ious >= iou_threshold).astype(np.float32)
        
        # Compute AP
        if binary_labels.sum() == 0:
            return 0.0
        
        try:
            ap = simple_average_precision(binary_labels, all_confidences)
            return ap
        except Exception as e:
            print(f"Warning: AP calculation failed: {e}")
            return 0.0
    
    def compute_comprehensive_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            pred: Predicted masks [B, 1, H, W] or [B, H, W]
            target: Ground truth masks [B, 1, H, W] or [B, H, W]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # IoU metrics
        iou_scores = self.compute_iou(pred, target)
        metrics['mIoU'] = iou_scores.mean().item()
        # metrics['IoU_std'] = iou_scores.std().item()  # REMOVED: Not logged to wandb for speed
        
        # Dice metrics
        dice_scores = self.compute_dice(pred, target)
        # metrics['mDice'] = dice_scores.mean().item()  # REMOVED: Not logged to wandb for speed
        # metrics['Dice_std'] = dice_scores.std().item()  # REMOVED: Not logged to wandb for speed
        
        # Precision and Recall
        precision, recall = self.compute_precision_recall(pred, target)
        metrics['mPrecision'] = precision.mean().item()
        metrics['mRecall'] = recall.mean().item()
        
        # F1 Score (same as Dice but computed differently)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        metrics['mF1'] = f1.mean().item()
        
        # AP at different IoU thresholds
        for iou_thresh in [0.5, 0.75]:
            ap = self.compute_ap_at_iou(pred, target, iou_thresh)
            metrics[f'AP@{iou_thresh}'] = ap
        
        # Mean AP across all thresholds (mAP)
        ap_scores = []
        for iou_thresh in self.iou_thresholds:
            ap = self.compute_ap_at_iou(pred, target, iou_thresh)
            ap_scores.append(ap)
        metrics['mAP'] = np.mean(ap_scores)
        
        return metrics


class ConsistencyMetrics:
    """
    Metrics for evaluating consistency between visible and amodal masks
    """
    
    @staticmethod
    def visible_consistency_error(amodal_pred: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute error where amodal prediction doesn't include visible regions
        
        Args:
            amodal_pred: Predicted amodal mask [B, 1, H, W]
            visible_mask: Ground truth visible mask [B, 1, H, W]
            
        Returns:
            Consistency error [B]
        """
        # Ensure same shape
        if amodal_pred.dim() == 4:
            amodal_pred = amodal_pred.squeeze(1)
        if visible_mask.dim() == 4:
            visible_mask = visible_mask.squeeze(1)
        
        # Amodal should include all visible regions
        error_regions = visible_mask * (1 - amodal_pred)
        error_rate = error_regions.sum(dim=[1, 2]) / (visible_mask.sum(dim=[1, 2]) + 1e-8)
        
        return error_rate
    
    @staticmethod
    def amodal_completion_quality(amodal_pred: torch.Tensor, amodal_gt: torch.Tensor, 
                                visible_mask: torch.Tensor) -> torch.Tensor:
        """
        Evaluate quality of amodal completion (occluded regions)
        
        Args:
            amodal_pred: Predicted amodal mask [B, 1, H, W]
            amodal_gt: Ground truth amodal mask [B, 1, H, W]
            visible_mask: Visible mask [B, 1, H, W]
            
        Returns:
            Completion quality IoU [B]
        """
        # Ensure same shape
        if amodal_pred.dim() == 4:
            amodal_pred = amodal_pred.squeeze(1)
        if amodal_gt.dim() == 4:
            amodal_gt = amodal_gt.squeeze(1)
        if visible_mask.dim() == 4:
            visible_mask = visible_mask.squeeze(1)
        
        # Binarize masks (use boolean for logical operations)
        amodal_pred_binary = (amodal_pred > 0.5)
        amodal_gt_binary = (amodal_gt > 0.5)
        visible_binary = (visible_mask > 0.5)
        
        # Extract occluded regions (boolean operations)
        occluded_gt = amodal_gt_binary & (~visible_binary)
        occluded_pred = amodal_pred_binary & (~visible_binary)
        
        # Compute IoU on occluded regions only using logical operations
        # Intersection: logical AND
        intersection = (occluded_pred & occluded_gt).sum(dim=[1, 2]).float()
        # Union: logical OR
        union = (occluded_pred | occluded_gt).sum(dim=[1, 2]).float()
        
        iou = intersection / (union + 1e-8)
        
        return iou
    
    @staticmethod
    def visible_region_quality(amodal_pred: torch.Tensor, amodal_gt: torch.Tensor, 
                              visible_mask: torch.Tensor) -> torch.Tensor:
        """
        Evaluate quality on visible regions only
        
        Args:
            amodal_pred: Predicted amodal mask [B, 1, H, W]
            amodal_gt: Ground truth amodal mask [B, 1, H, W]
            visible_mask: Visible mask [B, 1, H, W]
            
        Returns:
            Visible region IoU [B]
        """
        # Ensure same shape
        if amodal_pred.dim() == 4:
            amodal_pred = amodal_pred.squeeze(1)
        if amodal_gt.dim() == 4:
            amodal_gt = amodal_gt.squeeze(1)
        if visible_mask.dim() == 4:
            visible_mask = visible_mask.squeeze(1)
        
        # Binarize masks (use boolean for logical operations)
        amodal_pred_binary = (amodal_pred > 0.5)
        amodal_gt_binary = (amodal_gt > 0.5)
        visible_binary = (visible_mask > 0.5)
        
        # Extract visible regions (boolean operations)
        visible_gt = amodal_gt_binary & visible_binary
        visible_pred = amodal_pred_binary & visible_binary
        
        # Compute IoU on visible regions only using logical operations
        # Intersection: logical AND
        intersection = (visible_pred & visible_gt).sum(dim=[1, 2]).float()
        # Union: logical OR
        union = (visible_pred | visible_gt).sum(dim=[1, 2]).float()
        
        iou = intersection / (union + 1e-8)
        
        return iou


def compute_c2f_compatible_metrics(pred_masks: torch.Tensor, 
                                 gt_amodal: torch.Tensor, 
                                 gt_visible: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics compatible with C2F-Seg evaluation protocol
    
    Args:
        pred_masks: Predicted amodal masks [B, 1, H, W]
        gt_amodal: Ground truth amodal masks [B, 1, H, W]
        gt_visible: Ground truth visible masks [B, 1, H, W]
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize metric calculators
    seg_metrics = AmodalSegmentationMetrics()
    consistency_metrics = ConsistencyMetrics()
    
    # Comprehensive segmentation metrics
    segmentation_results = seg_metrics.compute_comprehensive_metrics(pred_masks, gt_amodal)
    
    # Consistency metrics
    consistency_error = consistency_metrics.visible_consistency_error(pred_masks, gt_visible)
    
    # Region-specific IoU metrics
    occluded_iou = consistency_metrics.amodal_completion_quality(pred_masks, gt_amodal, gt_visible)
    visible_iou = consistency_metrics.visible_region_quality(pred_masks, gt_amodal, gt_visible)
    
    # Add all metrics
    # segmentation_results['Consistency_Error'] = consistency_error.mean().item()  # REMOVED: Not logged to wandb for speed
    segmentation_results['Occluded_IoU'] = occluded_iou.mean().item()  # IoU on occluded regions
    segmentation_results['Visible_IoU'] = visible_iou.mean().item()    # IoU on visible regions
    
    # Backward compatibility
    # segmentation_results['Completion_IoU'] = segmentation_results['Occluded_IoU']  # REMOVED: Not logged to wandb for speed (alias)
    
    # Overall quality score (weighted combination)
    # Note: Consistency_Error not available now, adjust formula
    quality_score = (
        0.4 * segmentation_results['mIoU'] + 
        0.4 * segmentation_results['mAP'] + 
        0.2 * segmentation_results['Occluded_IoU']
        # Removed consistency_error component since it's not computed
    )
    # segmentation_results['Overall_Quality'] = quality_score  # REMOVED: Not logged to wandb for speed
    
    return segmentation_results


# Test the metrics
if __name__ == "__main__":
    # Create dummy data for testing
    batch_size = 4
    pred = torch.rand(batch_size, 1, 224, 224)
    gt_amodal = torch.rand(batch_size, 1, 224, 224) > 0.5
    gt_visible = gt_amodal * (torch.rand(batch_size, 1, 224, 224) > 0.3)  # Visible is subset of amodal
    
    # Convert to float
    gt_amodal = gt_amodal.float()
    gt_visible = gt_visible.float()
    
    # Compute metrics
    results = compute_c2f_compatible_metrics(pred, gt_amodal, gt_visible)
    
    print("📊 Evaluation Metrics Test:")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")