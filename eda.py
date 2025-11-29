import os
from pathlib import Path
from collections import defaultdict


def load_dataset_info(dataset_path):
    """
    Scan and summarize dataset structure.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        dict: Dataset information including structure and file counts
    """
    dataset_path = Path(dataset_path)

    
    info = {
        "dataset_name": dataset_path.name,
        "path": str(dataset_path),
        "classes": {},
        "total_images": 0
    }
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.nii', '.gz'}
    
    # Walk through directory structure
    for root, dirs, files in os.walk(dataset_path):
        # Filter image files
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions or f.endswith('.nii.gz')]
        
        if image_files:
            # Get relative path from dataset root
            rel_path = Path(root).relative_to(dataset_path)
            class_name = str(rel_path)
            
            info["classes"][class_name] = len(image_files)
            info["total_images"] += len(image_files)
    
    return info


def get_class_distribution(dataset_paths):
    """
    Count images per class/category across multiple datasets.
    
    Args:
        dataset_paths: List of paths to dataset directories
        
    Returns:
        dict: Distribution of images per dataset and class
    """
    distribution = {}
    
    for path in dataset_paths:
        dataset_info = load_dataset_info(path)
        
        if "error" not in dataset_info:
            distribution[dataset_info["dataset_name"]] = {
                "classes": dataset_info["classes"],
                "total": dataset_info["total_images"]
            }
    
    return distribution
