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
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:  # With domain labels
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
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
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Metrics/precision', val_precision, epoch)
            self.writer.add_scalar('Metrics/recall', val_recall, epoch)
            self.writer.add_scalar('Metrics/f1', val_f1, epoch)
            
            if self.scheduler:
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
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
            
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
        
        if is_best:
            save_path = self.save_dir / 'best_model.pth'
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
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


if __name__ == '__main__':
    # Example usage
    trainer = train_model(
        train_csv='Dataset/preprocessed_data/train_metadata.csv',
        val_csv='Dataset/preprocessed_data/val_metadata.csv',
        test_csv='Dataset/preprocessed_data/test_metadata.csv',
        num_classes=2,  # Adjust based on your dataset
        batch_size=16,
        num_epochs=100,
        device='cuda'
    )
