import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil


def resize_image(image, target_size=(224, 224), keep_aspect_ratio=False):
    """
    Resize image to target size.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height)
        keep_aspect_ratio: If True, pad to maintain aspect ratio
        
    Returns:
        PIL Image: Resized image
    """
    if keep_aspect_ratio:
        # Create a new image with target size and paste the resized image
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', target_size, (0, 0, 0))
        paste_position = ((target_size[0] - image.size[0]) // 2,
                         (target_size[1] - image.size[1]) // 2)
        new_image.paste(image, paste_position)
        return new_image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)


def normalize_image(image_array, method='minmax'):
    """
    Normalize image pixel values.
    
    Args:
        image_array: numpy array of image
        method: 'minmax' for [0,1] range or 'standard' for mean=0, std=1
        
    Returns:
        numpy array: Normalized image
    """
    image_array = image_array.astype(np.float32)
    
    if method == 'minmax':
        # Normalize to [0, 1]
        return image_array / 255.0
    elif method == 'standard':
        # Standardize to mean=0, std=1
        mean = np.mean(image_array)
        std = np.std(image_array)
        if std > 0:
            return (image_array - mean) / std
        return image_array - mean
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")


def create_data_split(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create train/val/test splits from dataset.
    
    Args:
        dataset_path: Path to dataset directory
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' keys containing file paths and labels
    """
    dataset_path = Path(dataset_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    data = {'file_path': [], 'label': [], 'class_name': []}
    
    # Collect all images and labels
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in image_extensions:
                    data['file_path'].append(str(img_file))
                    data['class_name'].append(class_name)
                    data['label'].append(class_name)
    
    if len(data['file_path']) == 0:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: separate validation from training
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    return {
        'train': train_df.reset_index(drop=True),
        'val': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True)
    }


def organize_processed_data(splits, output_dir, copy_files=True):
    """
    Organize preprocessed data into structured directories.
    
    Args:
        splits: Dictionary from create_data_split
        output_dir: Base directory for organized data
        copy_files: If True, copy files; if False, just create metadata
        
    Returns:
        dict: Paths to created directories and metadata files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    
    for split_name, df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        if copy_files:
            # Create class directories
            for class_name in df['class_name'].unique():
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
        
        # Save metadata CSV
        metadata_path = output_dir / f"{split_name}_metadata.csv"
        df.to_csv(metadata_path, index=False)
        metadata[split_name] = metadata_path
    
    # Save combined metadata
    all_data = []
    for split_name, df in splits.items():
        df_copy = df.copy()
        df_copy['split'] = split_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_path = output_dir / 'all_metadata.csv'
    combined_df.to_csv(combined_path, index=False)
    metadata['combined'] = combined_path
    
    return metadata


def preprocess_and_save(image_path, output_path, target_size=(224, 224), normalize_method='minmax'):
    """
    Preprocess a single image and save it.
    
    Args:
        image_path: Path to input image
        output_path: Path to save preprocessed image
        target_size: Target size for resizing
        normalize_method: Normalization method
        
    Returns:
        bool: True if successful
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img_resized = resize_image(img, target_size)
        
        # Convert to array for normalization
        img_array = np.array(img_resized)
        
        # Normalize
        img_normalized = normalize_image(img_array, method=normalize_method)
        
        # Convert back to image (scale to 0-255 for saving)
        if normalize_method == 'minmax':
            img_to_save = (img_normalized * 255).astype(np.uint8)
        else:
            # For standard normalization, scale back to 0-255 range
            img_min = img_normalized.min()
            img_max = img_normalized.max()
            if img_max > img_min:
                img_to_save = ((img_normalized - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_to_save = np.zeros_like(img_normalized, dtype=np.uint8)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_to_save).save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def get_image_statistics(image_path):
    """
    Get basic statistics about an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Image statistics
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        return {
            'width': img.size[0],
            'height': img.size[1],
            'mode': img.mode,
            'file_size_kb': os.path.getsize(image_path) / 1024,
            'mean_pixel': np.mean(img_array),
            'std_pixel': np.std(img_array),
            'min_pixel': np.min(img_array),
            'max_pixel': np.max(img_array)
        }
    except Exception as e:
        return {'error': str(e)}
