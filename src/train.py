"""
Training script for Brain Tumor Classification using Vision Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from src.model import create_vit_classifier
from src.dataset import create_dataloaders, get_class_weights

class Trainer:
    """
    Trainer class for Vision Transformer model.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        save_dir='models/checkpoints',
        log_dir='logs',
        num_classes=2
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, epoch):
        """Train for one epoch with mixed precision."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:  # With domain labels
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_preds
    
    def train(self, num_epochs, early_stopping_patience=10):
        """
        Train the model for specified number of epochs.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc, val_precision, val_recall, val_f1, val_labels, val_preds = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            # Logging (reduced frequency for I/O optimization)
            log_freq = getattr(self, 'log_freq', 1)  # Default to every epoch if not set
            if epoch % log_freq == 0:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Metrics/precision', val_precision, epoch)
                self.writer.add_scalar('Metrics/recall', val_recall, epoch)
                self.writer.add_scalar('Metrics/f1', val_f1, epoch)
            
            if self.scheduler:
                if epoch % log_freq == 0:
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Val   - P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  New best model saved! (Val Acc: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Removed periodic checkpoint saving to save storage
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            print("-" * 70)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        self.writer.close()
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        # Only save best model checkpoint
        if is_best:
            save_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
    
    def save_history(self):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        
        return checkpoint['epoch']


def train_model(
    train_csv,
    val_csv,
    test_csv,
    num_classes=2,
    img_size=224,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=0.05,
    use_class_weights=True,
    device='cuda',
    save_dir='models/checkpoints',
    log_dir='logs'
):
    """
    Main training function.
    """
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        train_csv,
        val_csv,
        test_csv,
        batch_size=batch_size,
        num_workers=4,
        augment_train=True,
        img_size=img_size
    )
    
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_vit_classifier(
        num_classes=num_classes,
        img_size=img_size,
        in_channels=3
    )
    
    # Loss function
    if use_class_weights:
        class_weights = get_class_weights(train_csv).to(device)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        num_classes=num_classes
    )
    
    # Train
    trainer.train(num_epochs=num_epochs, early_stopping_patience=15)
    
    return trainer


class SegmentationTrainer:
    """
    Trainer class for segmentation tasks with Vision Transformer.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        save_dir='experiments/segmentation_checkpoints',
        log_dir='experiments/segmentation_logs',
        num_classes=4,
        checkpoint_freq=10  # Added: Save checkpoint every N epochs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.checkpoint_freq = checkpoint_freq  # Store checkpoint frequency
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Tracking
        self.best_val_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        from src.seg_metrics import SegmentationMetrics
        
        self.model.train()
        running_loss = 0.0
        running_dice_loss = 0.0
        running_ce_loss = 0.0
        metrics = SegmentationMetrics(num_classes=self.num_classes)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            if hasattr(self.criterion, 'dice_weight'):  # CombinedLoss
                loss, dice_loss, ce_loss = self.criterion(outputs, masks)
                running_dice_loss += dice_loss.item()
                running_ce_loss += ce_loss.item()
            else:
                loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            metrics.update(outputs.detach(), masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def validate(self, epoch):
        """Validate the model."""
        from src.seg_metrics import SegmentationMetrics
        
        self.model.eval()
        running_loss = 0.0
        metrics = SegmentationMetrics(num_classes=self.num_classes)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if hasattr(self.criterion, 'dice_weight'):
                    loss, _, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                metrics.update(outputs, masks)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_metrics = metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def train(self, num_epochs, early_stopping_patience=15):
        """
        Complete training loop.
        """
        print(f"\n{'='*60}")
        print(f"Starting Segmentation Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice_scores.append(train_metrics['mean_dice'])
            self.val_dice_scores.append(val_metrics['mean_dice'])
            
            # Log to TensorBoard (reduced frequency for I/O optimization)
            log_freq = getattr(self, 'log_freq', 1)  # Default to every epoch
            if epoch % log_freq == 0:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train', train_metrics['mean_dice'], epoch)
                self.writer.add_scalar('Dice/val', val_metrics['mean_dice'], epoch)
                self.writer.add_scalar('IoU/train', train_metrics['mean_iou'], epoch)
                self.writer.add_scalar('IoU/val', val_metrics['mean_iou'], epoch)
                
                # Log per-class metrics
                class_names = ['background', 'whole_tumor', 'tumor_core', 'enhancing_tumor']
                for i, name in enumerate(class_names):
                    self.writer.add_scalar(f'Dice_{name}/train', train_metrics[f'dice_{name}'], epoch)
                    self.writer.add_scalar(f'Dice_{name}/val', val_metrics[f'dice_{name}'], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_metrics['mean_dice']:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_metrics['mean_dice']:.4f}")
            print(f"Val IoU:    {val_metrics['mean_iou']:.4f}")
            
            # Save best model
            if val_metrics['mean_dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['mean_dice']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                patience_counter = 0
                print(f" New best model! Dice: {self.best_val_dice:.4f}")
            else:
                patience_counter += 1
            
            # Removed periodic checkpoint saving to save storage
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Save final training history
        self.save_training_history()
        self.writer.close()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Dice: {self.best_val_dice:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'metrics': metrics
        }
        if is_best:
            save_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dice_scores': self.train_dice_scores,
            'val_dice_scores': self.val_dice_scores,
            'best_val_dice': float(self.best_val_dice)
        }
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {history_path}")


def train_segmentation_model(
    train_csv,
    val_csv,
    test_csv,
    data_root="Dataset/brats_preprocessed/slices",
    num_classes=4,
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-4,
    encoder_lr=None,  # NEW: Separate LR for encoder (defaults to learning_rate)
    decoder_lr=None,  # NEW: Separate LR for decoder (defaults to learning_rate)
    weight_decay=0.05,
    device='cuda',
    save_dir='experiments/segmentation_checkpoints',
    log_dir='experiments/segmentation_logs',
    img_size=224,
    loss_type='combined',
    dice_weight=0.7,  # NEW: Weight for dice in combined losses
    focal_weight=0.3,  # NEW: Weight for focal in focal_dice loss
    focal_gamma=2.0,  # NEW: Focal loss gamma parameter
    use_skip_connections=True,  # NEW: Use improved model with skip connections
    num_workers=0,
    checkpoint_freq=10,
    early_stopping_patience=15,
    augment_config=None  # NEW: Custom augmentation configuration
):
    """
    Complete pipeline for training segmentation model with improvements.
    
    New features:
    - Differential learning rates for encoder/decoder
    - Improved loss functions (focal_dice)
    - Skip connections in model architecture
    - Configurable loss weights
    - Custom augmentation control
    """
    from src.model import create_vit_segmentation
    from src.dataset import create_segmentation_dataloaders
    from src.losses import get_segmentation_loss
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_segmentation_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=True,
        augment_config=augment_config,  # Pass custom config
        img_size=img_size
    )
    
    print(f"  Using num_workers={num_workers}")
    
    # Create model
    print(f"\nCreating segmentation model (skip_connections={use_skip_connections})...")
    model = create_vit_segmentation(
        num_classes=num_classes,
        img_size=img_size,
        in_channels=4,  # T1, T1-CE, T2, FLAIR
        use_skip_connections=use_skip_connections
    )
    
    # Loss function with improved parameters
    loss_kwargs = {}
    if loss_type == 'focal_dice':
        loss_kwargs = {
            'dice_weight': dice_weight,
            'focal_weight': focal_weight,
            'focal_gamma': focal_gamma
        }
    elif loss_type == 'combined':
        loss_kwargs = {
            'dice_weight': dice_weight,
            'ce_weight': 1.0 - dice_weight
        }
    
    criterion = get_segmentation_loss(loss_type=loss_type, **loss_kwargs)
    print(f"Using {loss_type} loss with weights: {loss_kwargs}")
    
    # Optimizer with differential learning rates
    if encoder_lr is None:
        encoder_lr = learning_rate
    if decoder_lr is None:
        decoder_lr = learning_rate
    
    # Group parameters by encoder/decoder for differential LR
    if use_skip_connections and (encoder_lr != decoder_lr):
        print(f"\nUsing differential learning rates:")
        print(f"  Encoder LR: {encoder_lr}")
        print(f"  Decoder LR: {decoder_lr}")
        
        encoder_params = []
        decoder_params = []
        
        for name, param in model.named_parameters():
            if 'decoder' in name or 'skip_proj' in name or 'seg_head' in name or 'upsample' in name:
                decoder_params.append(param)
            else:
                encoder_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': decoder_lr}
        ], weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        num_classes=num_classes,
        checkpoint_freq=checkpoint_freq  # Pass checkpoint frequency
    )
    
    # Train
    trainer.train(num_epochs=num_epochs, early_stopping_patience=early_stopping_patience)
    
    return trainer


if __name__ == '__main__':
    # Example usage for classification
    # trainer = train_model(
    #     train_csv='Dataset/preprocessed_data/train_metadata.csv',
    #     val_csv='Dataset/preprocessed_data/val_metadata.csv',
    #     test_csv='Dataset/preprocessed_data/test_metadata.csv',
    #     num_classes=2,
    #     batch_size=16,
    #     num_epochs=100,
    #     device='cuda'
    # )
    
    # Example usage for segmentation
    trainer = train_segmentation_model(
        train_csv='Dataset/brats_preprocessed/train_metadata.csv',
        val_csv='Dataset/brats_preprocessed/val_metadata.csv',
        test_csv='Dataset/brats_preprocessed/test_metadata.csv',
        data_root='Dataset/brats_preprocessed/slices',
        num_classes=4,
        batch_size=8,
        num_epochs=100,
        device='cuda'
    )
