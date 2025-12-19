import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import shutil
from scipy.ndimage import map_coordinates, gaussian_filter
import random


def resize_image(image, target_size=(224, 224), keep_aspect_ratio=False):
    """
    Resize image to target size.
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


def zscore_normalize(image_array, per_channel=True, epsilon=1e-8):
    """
    Z-score normalization: (x - μ) / σ
    
    Standardizes image intensity values to have mean=0 and std=1.
    Recommended for medical imaging and cross-dataset generalization.
    """
    image_array = image_array.astype(np.float32)
    
    if per_channel and len(image_array.shape) == 3:
        # Multi-channel image (H, W, C) - normalize each channel independently
        normalized = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            channel = image_array[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)
            normalized[:, :, c] = (channel - mean) / (std + epsilon)
        return normalized
    else:
        # Single channel or grayscale image
        mean = np.mean(image_array)
        std = np.std(image_array)
        return (image_array - mean) / (std + epsilon)


# ============================================================================
# Data Augmentation Functions (For Thesis - Domain Generalization)
# ============================================================================

def augment_rotation(image, angle_range=(-30, 30)):
    """
    Rotate image by a random angle.
    """
    angle = random.uniform(angle_range[0], angle_range[1])
    return image.rotate(angle, resample=Image.BICUBIC, fillcolor=0)


def augment_flip(image, horizontal=True, vertical=False):
    """
    Randomly flip image horizontally and/or vertically.
    """
    if horizontal and random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if vertical and random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def augment_zoom(image, zoom_range=(0.8, 1.2)):
    """
    Random zoom in/out with center crop/pad.
    """
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
    w, h = image.size
    
    # Calculate new size
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    
    # Resize image
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    
    # Center crop or pad to original size
    if zoom_factor > 1.0:
        # Crop from center
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    else:
        # Pad to center
        new_image = Image.new(image.mode, (w, h), 0)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        new_image.paste(resized, (paste_x, paste_y))
        return new_image


def augment_intensity(image, intensity_range=(0.9, 1.1)):
    """
    Vary image brightness/intensity.
    """
    factor = random.uniform(intensity_range[0], intensity_range[1])
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def augment_elastic_deform(image_array, alpha=30, sigma=5, random_state=None):
    """
    Elastic deformation for medical image augmentation.
    Particularly useful for segmentation tasks.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image_array.shape[:2]
    
    # Generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Apply deformation
    if len(image_array.shape) == 3:
        # Multi-channel image
        deformed = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            deformed[:, :, c] = map_coordinates(
                image_array[:, :, c], indices, order=1, mode='reflect'
            ).reshape(shape)
        return deformed
    else:
        # Single channel
        return map_coordinates(image_array, indices, order=1, mode='reflect').reshape(shape)


def apply_augmentation_pipeline(image, augment_config=None):
    """
    Apply a pipeline of augmentation transforms.
    """
    if augment_config is None:
        augment_config = {
            'rotation': True,
            'flip': True,
            'zoom': True,
            'intensity': True,
            'elastic': False  # Computationally expensive, use for segmentation
        }
    
    # Apply augmentations
    if augment_config.get('rotation', False):
        image = augment_rotation(image, angle_range=(-30, 30))
    
    if augment_config.get('flip', False):
        image = augment_flip(image, horizontal=True, vertical=True)
    
    if augment_config.get('zoom', False):
        image = augment_zoom(image, zoom_range=(0.8, 1.2))
    
    if augment_config.get('intensity', False):
        image = augment_intensity(image, intensity_range=(0.9, 1.1))
    
    if augment_config.get('elastic', False):
        img_array = np.array(image)
        img_array = augment_elastic_deform(img_array, alpha=30, sigma=5)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    return image


def create_data_split(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create train/val/test splits from dataset.
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


def preprocess_and_save(image_path, output_path, target_size=(224, 224)):
    """
    Preprocess a single image and save it.
    Uses Z-score normalization as per thesis requirements.
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
        
        # Apply Z-score normalization (per-channel for RGB)
        img_normalized = zscore_normalize(img_array, per_channel=True)
        
        # Convert back to image (scale to 0-255 for saving)
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
    except Exception:
        return False


def get_image_statistics(image_path):
    """
    Get basic statistics about an image.
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


def batch_preprocess_images(metadata_path, output_base_dir, target_size=(224, 224)):
    """
    Batch preprocess all images according to metadata CSV.
    Uses Z-score normalization as per thesis requirements.
    """
    df = pd.read_csv(metadata_path)
    output_base_dir = Path(output_base_dir)
    
    stats = {'total': len(df), 'success': 0, 'failed': 0, 'failed_files': []}
    
    for idx, row in df.iterrows():
        # Get split (train/val/test) and class name
        split = row['split']
        class_name = row['class_name']
        file_path = Path(row['file_path'])
        
        # Construct output path
        output_path = output_base_dir / split / class_name / file_path.name
        
        # Preprocess and save with Z-score normalization
        success = preprocess_and_save(file_path, output_path, target_size)
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
            stats['failed_files'].append(str(file_path))
    
    return stats
