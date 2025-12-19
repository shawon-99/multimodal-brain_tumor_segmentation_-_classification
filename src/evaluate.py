"""
Evaluation module for model performance assessment
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from tqdm import tqdm
import json
from pathlib import Path


class Evaluator:
    """Model evaluator with comprehensive metrics."""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, dataloader, class_names=None):
        """
        Evaluate model on dataloader.
        
        Args:
            dataloader: DataLoader for evaluation
            class_names: List of class names
        
        Returns:
            Dictionary with metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, all_preds, all_probs, class_names
        )
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_probs, class_names=None):
        """Calculate all metrics."""
        num_classes = len(np.unique(y_true))
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f'class_{i}'
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # ROC-AUC
        if num_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
            except:
                metrics['roc_auc_ovr'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def print_metrics(self, metrics, class_names=None):
        """Print metrics in readable format."""
     
        print("EVALUATION METRICS")
        print("="*60)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        elif 'roc_auc_ovr' in metrics and metrics['roc_auc_ovr'] is not None:
            print(f"  ROC-AUC (OVR):      {metrics['roc_auc_ovr']:.4f}")
        
        # Per-class metrics
        num_classes = len(metrics['confusion_matrix'])
        if num_classes > 1:
            print("\nPer-Class Metrics:")
            for i in range(num_classes):
                class_name = class_names[i] if class_names else f'Class {i}'
                precision_key = f'precision_{class_name}' if class_names else f'precision_class_{i}'
                recall_key = f'recall_{class_name}' if class_names else f'recall_class_{i}'
                f1_key = f'f1_{class_name}' if class_names else f'f1_class_{i}'
                
                print(f"\n  {class_name}:")
                print(f"    Precision: {metrics[precision_key]:.4f}")
                print(f"    Recall:    {metrics[recall_key]:.4f}")
                print(f"    F1-Score:  {metrics[f1_key]:.4f}")
        
        print("\n" + "="*60)


class CrossDatasetEvaluator:
    """Evaluator for cross-dataset (domain generalization) evaluation."""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize cross-dataset evaluator.
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.evaluator = Evaluator(model, device)
    
    def evaluate_all_datasets(self, dataloaders_dict, class_names=None):
        """
        Evaluate on multiple datasets.
        
        Args:
            dataloaders_dict: Dictionary {dataset_name: dataloader}
            class_names: List of class names
        
        Returns:
            Dictionary {dataset_name: metrics}
        """
        results = {}
        
        for dataset_name, dataloader in dataloaders_dict.items():
            print(f"\nEvaluating on {dataset_name}...")
            metrics = self.evaluator.evaluate(dataloader, class_names)
            results[dataset_name] = metrics
            
            # Print metrics
            self.evaluator.print_metrics(metrics, class_names)
        
        # Calculate generalization gap
        if len(results) > 1:
            self._calculate_generalization_gap(results)
        
        return results
    
    def _calculate_generalization_gap(self, results):
        """Calculate generalization gap between datasets."""
        print("\n" + "="*60)
        print("GENERALIZATION ANALYSIS")
        print("="*60)
        
        # Extract accuracies
        dataset_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in dataset_names]
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\nMean Accuracy across datasets: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Min Accuracy: {min(accuracies):.4f}")
        print(f"Max Accuracy: {max(accuracies):.4f}")
        print(f"Generalization Gap (max-min): {max(accuracies) - min(accuracies):.4f}")
        
        print("\nPer-Dataset Performance:")
        for name, acc in zip(dataset_names, accuracies):
            print(f"  {name}: {acc:.4f}")
        
        print("="*60)
    
    def leave_one_out_evaluation(self, all_dataloaders, dataset_names, class_names=None):
        """
        Perform leave-one-domain-out evaluation.
        
        Args:
            all_dataloaders: List of dataloaders for each dataset
            dataset_names: List of dataset names
            class_names: List of class names
        
        Note: This requires retraining for each held-out domain
        """
        print("\n" + "="*60)
        print("LEAVE-ONE-DOMAIN-OUT EVALUATION")
        print("="*60)
        print("\nNote: This requires retraining models for each configuration.")
        print("Please use train.py with appropriate train/test split for each fold.")
        print("="*60)


def save_evaluation_results(results, save_path):
    """
    Save evaluation results to JSON.
    
    Args:
        results: Dictionary with evaluation metrics
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {save_path}")


def compare_models(results_dict):
    """
    Compare multiple models.
    
    Args:
        results_dict: Dictionary {model_name: metrics_dict}
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    model_names = list(results_dict.keys())
    
    # Compare key metrics
    metrics_to_compare = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 78)
    
    for model_name in model_names:
        metrics = results_dict[model_name]
        print(f"{model_name:<30} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision_macro']:<12.4f} "
              f"{metrics['recall_macro']:<12.4f} "
              f"{metrics['f1_macro']:<12.4f}")
    
    print("="*60)
