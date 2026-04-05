"""
Loss functions for segmentation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice Coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(self, smooth=1.0, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        batch_size, num_classes, height, width = pred.shape
        
        # Flatten spatial dimensions
        pred_probs = pred_probs.view(batch_size, num_classes, -1)  # (B, C, H*W)
        target = target.view(batch_size, -1)  # (B, H*W)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H*W, C)
        target_one_hot = target_one_hot.permute(0, 2, 1).float()  # (B, C, H*W)
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            
            pred_c = pred_probs[:, c, :]
            target_c = target_one_hot[:, c, :]
            
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice across classes and batch
        dice_scores = torch.stack(dice_scores, dim=1)  # (B, num_classes)
        dice_loss = 1.0 - dice_scores.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        
        Returns:
            Focal loss (scalar)
        """
        # Apply log softmax
        log_probs = F.log_softmax(pred, dim=1)  # (B, C, H, W)
        
        batch_size, num_classes, height, width = pred.shape
        
        # Flatten spatial dimensions
        log_probs = log_probs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        log_probs = log_probs.view(-1, num_classes)  # (B*H*W, C)
        target = target.view(-1)  # (B*H*W)
        
        # Get probabilities
        probs = torch.exp(log_probs)  # (B*H*W, C)
        
        # Get target probabilities
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()  # (B*H*W, C)
        pt = (probs * target_one_hot).sum(dim=1)  # (B*H*W)
        
        # Calculate focal term
        focal_term = (1.0 - pt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B*H*W)
        
        # Apply focal term
        focal_loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(pred.device)[target]
                focal_loss = alpha_t * focal_loss
        
        # Mask out ignored indices
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            focal_loss = focal_loss[mask]
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss for segmentation.
    This combination leverages:
    - Dice Loss: Good for class imbalance, focuses on overlap
    - Cross Entropy: Penalizes confident wrong predictions
    """
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None, ignore_index=-100):
        """
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross Entropy loss
            class_weights: Class weights for CE loss (tensor of shape [num_classes])
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return total_loss, dice, ce


class FocalDiceLoss(nn.Module):
    """
    Improved Combined Focal + Dice Loss for better handling of hard examples.
    Focal loss down-weights easy examples, Dice handles class imbalance.
    """
    
    def __init__(self, dice_weight=0.7, focal_weight=0.3, focal_gamma=2.0, 
                 focal_alpha=None, smooth=1.0, ignore_index=-100):
        """
        Args:
            dice_weight: Weight for Dice loss (default 0.7 to prioritize overlap)
            focal_weight: Weight for Focal loss
            focal_gamma: Focusing parameter (higher = focus more on hard examples)
            focal_alpha: Class weights for focal loss
            smooth: Smoothing factor for Dice
            ignore_index: Index to ignore
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.focal_weight * focal
        
        return total_loss, dice, focal


class TverskyLoss(nn.Module):
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        batch_size, num_classes, height, width = pred.shape
        
        # Flatten spatial dimensions
        pred_probs = pred_probs.view(batch_size, num_classes, -1)  # (B, C, H*W)
        target = target.view(batch_size, -1)  # (B, H*W)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H*W, C)
        target_one_hot = target_one_hot.permute(0, 2, 1).float()  # (B, C, H*W)
        
        # Calculate Tversky index for each class
        tversky_scores = []
        for c in range(num_classes):
            pred_c = pred_probs[:, c, :]
            target_c = target_one_hot[:, c, :]
            
            # True Positives, False Positives, False Negatives
            tp = (pred_c * target_c).sum(dim=1)
            fp = (pred_c * (1 - target_c)).sum(dim=1)
            fn = ((1 - pred_c) * target_c).sum(dim=1)
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_scores.append(tversky)
        
        # Average Tversky across classes and batch
        tversky_scores = torch.stack(tversky_scores, dim=1)  # (B, num_classes)
        tversky_loss = 1.0 - tversky_scores.mean()
        
        return tversky_loss


def get_segmentation_loss(loss_type='combined', **kwargs):
    """
    Factory function to create segmentation loss.
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'focal_dice':
        return FocalDiceLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 4
    num_classes = 4
    height, width = 224, 224
    
    # Create dummy data
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test Dice Loss
    dice_loss_fn = DiceLoss()
    dice_loss = dice_loss_fn(pred, target)
    print(f"Dice Loss: {dice_loss.item():.4f}")
    
    # Test Focal Loss
    focal_loss_fn = FocalLoss()
    focal_loss = focal_loss_fn(pred, target)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Test Combined Loss
    combined_loss_fn = CombinedLoss()
    total_loss, dice, ce = combined_loss_fn(pred, target)
    print(f"Combined Loss: {total_loss.item():.4f} (Dice: {dice.item():.4f}, CE: {ce.item():.4f})")
    
    # Test Tversky Loss
    tversky_loss_fn = TverskyLoss()
    tversky_loss = tversky_loss_fn(pred, target)
    print(f"Tversky Loss: {tversky_loss.item():.4f}")
