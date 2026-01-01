"""
Evaluation metrics for segmentation tasks
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.ndimage import distance_transform_edt


def dice_coefficient(pred, target, num_classes=4, smooth=1e-8):
    """
    Calculate Dice coefficient for each class.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        pred: Predicted labels (B, H, W) or (H, W)
        target: Ground truth labels (B, H, W) or (H, W)
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        List of Dice scores per class
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten if batch dimension exists
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred_flat == c)
        target_c = (target_flat == c)
        
        intersection = np.logical_and(pred_c, target_c).sum()
        union_sum = pred_c.sum() + target_c.sum()
        
        if union_sum == 0:
            # No pixels of this class in prediction or target
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection + smooth) / (union_sum + smooth)
        
        dice_scores.append(float(dice))
    
    return dice_scores


def iou_score(pred, target, num_classes=4, smooth=1e-8):
    """
    Calculate Intersection over Union (IoU) for each class.
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        pred: Predicted labels (B, H, W) or (H, W)
        target: Ground truth labels (B, H, W) or (H, W)
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        List of IoU scores per class
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    iou_scores = []
    for c in range(num_classes):
        pred_c = (pred_flat == c)
        target_c = (target_flat == c)
        
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection + smooth) / (union + smooth)
        
        iou_scores.append(float(iou))
    
    return iou_scores


def sensitivity_specificity(pred, target, class_idx, smooth=1e-8):
    """
    Calculate sensitivity and specificity for a specific class.
    
    Sensitivity (Recall) = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    
    Args:
        pred: Predicted labels (B, H, W) or (H, W)
        target: Ground truth labels (B, H, W) or (H, W)
        class_idx: Class index to evaluate
        smooth: Smoothing factor
    
    Returns:
        Tuple of (sensitivity, specificity)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_c = (pred_flat == class_idx)
    target_c = (target_flat == class_idx)
    
    tp = np.logical_and(pred_c, target_c).sum()
    tn = np.logical_and(~pred_c, ~target_c).sum()
    fp = np.logical_and(pred_c, ~target_c).sum()
    fn = np.logical_and(~pred_c, target_c).sum()
    
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return float(sensitivity), float(specificity)


def hausdorff_distance_95(pred, target, class_idx, spacing=(1.0, 1.0)):
    """
    Calculate 95th percentile Hausdorff distance for a specific class.
    
    Measures the maximum surface distance between prediction and ground truth.
    
    Args:
        pred: Predicted labels (H, W)
        target: Ground truth labels (H, W)
        class_idx: Class index to evaluate
        spacing: Pixel spacing (dy, dx) in mm
    
    Returns:
        95th percentile Hausdorff distance in mm
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Extract binary masks for the class
    pred_mask = (pred == class_idx).astype(np.uint8)
    target_mask = (target == class_idx).astype(np.uint8)
    
    # Check if either mask is empty
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return float('inf') if pred_mask.sum() != target_mask.sum() else 0.0
    
    # Compute distance transforms
    pred_dt = distance_transform_edt(1 - pred_mask, sampling=spacing)
    target_dt = distance_transform_edt(1 - target_mask, sampling=spacing)
    
    # Get surface points
    pred_surface = pred_dt[target_mask > 0]
    target_surface = target_dt[pred_mask > 0]
    
    # Combine distances
    all_distances = np.concatenate([pred_surface, target_surface])
    
    if len(all_distances) == 0:
        return 0.0
    
    # Return 95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def pixel_accuracy(pred, target):
    """
    Calculate overall pixel-wise accuracy.
    
    Args:
        pred: Predicted labels (B, H, W) or (H, W)
        target: Ground truth labels (B, H, W) or (H, W)
    
    Returns:
        Pixel accuracy (0-1)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    correct = (pred == target).sum()
    total = pred.size
    
    return float(correct / total)


def mean_iou(pred, target, num_classes=4):
    """
    Calculate mean IoU across all classes.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Mean IoU score
    """
    iou_scores = iou_score(pred, target, num_classes)
    return float(np.mean(iou_scores))


def evaluate_segmentation(pred_logits, target_labels, num_classes=4):
    """
    Comprehensive evaluation of segmentation predictions.
    
    Args:
        pred_logits: Predicted logits (B, C, H, W)
        target_labels: Ground truth labels (B, H, W)
        num_classes: Number of classes
    
    Returns:
        Dictionary with all metrics
    """
    # Convert logits to predictions
    if isinstance(pred_logits, torch.Tensor):
        pred = torch.argmax(pred_logits, dim=1)  # (B, H, W)
    else:
        pred = np.argmax(pred_logits, axis=1)
    
    # Calculate all metrics
    dice_scores = dice_coefficient(pred, target_labels, num_classes)
    iou_scores = iou_score(pred, target_labels, num_classes)
    pixel_acc = pixel_accuracy(pred, target_labels)
    
    # Per-class sensitivity and specificity
    sens_spec = {}
    for c in range(num_classes):
        sens, spec = sensitivity_specificity(pred, target_labels, c)
        sens_spec[f'class_{c}_sensitivity'] = sens
        sens_spec[f'class_{c}_specificity'] = spec
    
    # Compile results
    results = {
        'pixel_accuracy': pixel_acc,
        'mean_dice': float(np.mean(dice_scores)),
        'mean_iou': float(np.mean(iou_scores)),
        **sens_spec
    }
    
    # Add per-class metrics
    class_names = ['background', 'whole_tumor', 'tumor_core', 'enhancing_tumor']
    for c in range(num_classes):
        results[f'dice_{class_names[c]}'] = dice_scores[c]
        results[f'iou_{class_names[c]}'] = iou_scores[c]
    
    return results


class SegmentationMetrics:
    """
    Class to accumulate and track segmentation metrics during training/evaluation.
    """
    
    def __init__(self, num_classes=4):
        """
        Args:
            num_classes: Number of segmentation classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.pixel_accuracies = []
        self.per_class_dice = {c: [] for c in range(self.num_classes)}
        self.per_class_iou = {c: [] for c in range(self.num_classes)}
    
    def update(self, pred_logits, target_labels):
        """
        Update metrics with a new batch.
        
        Args:
            pred_logits: Predicted logits (B, C, H, W)
            target_labels: Ground truth labels (B, H, W)
        """
        # Convert logits to predictions
        if isinstance(pred_logits, torch.Tensor):
            pred = torch.argmax(pred_logits, dim=1)
        else:
            pred = np.argmax(pred_logits, axis=1)
        
        # Calculate metrics for this batch
        dice = dice_coefficient(pred, target_labels, self.num_classes)
        iou = iou_score(pred, target_labels, self.num_classes)
        pixel_acc = pixel_accuracy(pred, target_labels)
        
        # Accumulate
        self.dice_scores.append(np.mean(dice))
        self.iou_scores.append(np.mean(iou))
        self.pixel_accuracies.append(pixel_acc)
        
        for c in range(self.num_classes):
            self.per_class_dice[c].append(dice[c])
            self.per_class_iou[c].append(iou[c])
    
    def compute(self):
        """
        Compute average metrics across all batches.
        
        Returns:
            Dictionary with averaged metrics
        """
        class_names = ['background', 'whole_tumor', 'tumor_core', 'enhancing_tumor']
        
        results = {
            'mean_dice': float(np.mean(self.dice_scores)),
            'mean_iou': float(np.mean(self.iou_scores)),
            'pixel_accuracy': float(np.mean(self.pixel_accuracies)),
        }
        
        # Add per-class metrics
        for c in range(self.num_classes):
            results[f'dice_{class_names[c]}'] = float(np.mean(self.per_class_dice[c]))
            results[f'iou_{class_names[c]}'] = float(np.mean(self.per_class_iou[c]))
        
        return results
    
    def __str__(self):
        """String representation of current metrics."""
        results = self.compute()
        
        s = f"Pixel Accuracy: {results['pixel_accuracy']:.4f}\n"
        s += f"Mean Dice: {results['mean_dice']:.4f}\n"
        s += f"Mean IoU: {results['mean_iou']:.4f}\n"
        
        class_names = ['background', 'whole_tumor', 'tumor_core', 'enhancing_tumor']
        for c, name in enumerate(class_names):
            dice = results[f'dice_{name}']
            iou = results[f'iou_{name}']
            s += f"  {name}: Dice={dice:.4f}, IoU={iou:.4f}\n"
        
        return s


if __name__ == "__main__":
    # Test metrics
    batch_size = 4
    num_classes = 4
    height, width = 224, 224
    
    # Create dummy data
    pred_logits = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test evaluation function
    results = evaluate_segmentation(pred_logits, target, num_classes)
    
    print("Segmentation Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Test metrics accumulator
    print("\nTesting SegmentationMetrics:")
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    for i in range(3):
        pred_logits = torch.randn(batch_size, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch_size, height, width))
        metrics.update(pred_logits, target)
    
    print(metrics)
