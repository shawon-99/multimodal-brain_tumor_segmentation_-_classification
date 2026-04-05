"""
Utility functions for training and evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path


def plot_training_history(history_path, save_path=None):
    """
    Plot training history (loss and accuracy curves).
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print best metrics
    best_val_acc = max(history['val_accs'])
    best_epoch = history['val_accs'].index(best_val_acc) + 1
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def save_predictions(predictions, labels, image_paths, save_path):
    """
    Save predictions to CSV file.
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'image_path': image_paths,
        'true_label': labels,
        'predicted_label': predictions,
        'correct': predictions == labels
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA, DirectML, or CPU)."""
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    # Check for DirectML (AMD/Intel GPUs on Windows)
    elif hasattr(torch, 'dml') or 'dml' in dir(torch):
        try:
            import torch_directml
            device = torch_directml.device()
            print("Using DirectML GPU (AMD/Intel)")
        except:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, epoch, best_val_acc


def create_experiment_dirs(base_dir='experiments'):
    """
    Create directory structure for experiments.
    """
    base_path = Path(base_dir)
    
    dirs = {
        'checkpoints': base_path / 'checkpoints',
        'logs': base_path / 'logs',
        'results': base_path / 'results',
        'figures': base_path / 'results' / 'figures',
        'predictions': base_path / 'results' / 'predictions'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs
