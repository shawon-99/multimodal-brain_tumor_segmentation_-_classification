"""
BRATs 2020 Dataset Preprocessing
Loads multi-modal NIfTI volumes and extracts 2D slices for training
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def load_nifti_volume(nifti_path: str) -> np.ndarray:
    """
    Load a NIfTI file and return the volume as numpy array.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
    
    Returns:
        3D numpy array (H, W, D)
    """
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()
    return volume


def load_brats_patient(patient_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load all modalities and segmentation for a single BRATs patient.
    
    Args:
        patient_dir: Path to patient folder (e.g., BraTS20_Training_001)
    
    Returns:
        Dictionary with keys: 't1', 't1ce', 't2', 'flair', 'seg'
    """
    patient_id = patient_dir.name
    
    volumes = {}
    for modality in ['t1', 't1ce', 't2', 'flair', 'seg']:
        file_path = patient_dir / f"{patient_id}_{modality}.nii"
        if not file_path.exists():
            file_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
        
        if file_path.exists():
            volumes[modality] = load_nifti_volume(str(file_path))
        else:
            raise FileNotFoundError(f"Missing {modality} file for patient {patient_id}")
    
    return volumes


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize a single 2D slice using Z-score normalization.
    
    Args:
        slice_data: 2D array (H, W)
    
    Returns:
        Normalized 2D array
    """
    # Only normalize non-zero regions (brain tissue)
    mask = slice_data > 0
    if mask.sum() > 0:
        mean = slice_data[mask].mean()
        std = slice_data[mask].std()
        if std > 0:
            normalized = slice_data.copy()
            normalized[mask] = (slice_data[mask] - mean) / std
            return normalized
    return slice_data


def convert_brats_labels(seg_slice: np.ndarray) -> np.ndarray:
    """
    Convert BRATs segmentation labels to multi-class format.
    
    BRATs labels: 0=background, 1=necrotic/non-enhancing, 2=edema, 4=enhancing
    
    Output classes:
        0: Background
        1: Whole Tumor (any tumor region: 1, 2, 4)
        2: Tumor Core (necrotic + enhancing: 1, 4)
        3: Enhancing Tumor (4)
    
    Args:
        seg_slice: 2D segmentation mask with BRATs labels
    
    Returns:
        2D segmentation mask with 0-3 labels
    """
    output = np.zeros_like(seg_slice, dtype=np.uint8)
    
    # Class 1: Whole tumor (any tumor)
    output[seg_slice > 0] = 1
    
    # Class 2: Tumor core (labels 1 and 4)
    output[(seg_slice == 1) | (seg_slice == 4)] = 2
    
    # Class 3: Enhancing tumor (label 4)
    output[seg_slice == 4] = 3
    
    return output


def has_significant_tumor(seg_slice: np.ndarray, min_pixels: int = 50) -> bool:
    """
    Check if a slice contains significant tumor (not just a few pixels).
    
    Args:
        seg_slice: 2D segmentation mask
        min_pixels: Minimum number of tumor pixels to be considered significant
    
    Returns:
        True if slice has significant tumor
    """
    tumor_pixels = (seg_slice > 0).sum()
    return tumor_pixels >= min_pixels


def extract_2d_slices(
    volumes: Dict[str, np.ndarray],
    patient_id: str,
    output_dir: Path,
    slice_range: Optional[Tuple[int, int]] = None,
    min_tumor_pixels: int = 50,
    save_slices: bool = True
) -> List[Dict]:
    """
    Extract 2D slices from 3D volumes.
    
    Args:
        volumes: Dictionary with modality volumes
        patient_id: Patient identifier
        output_dir: Directory to save extracted slices
        slice_range: (start, end) indices for axial slices, or None for auto-detect
        min_tumor_pixels: Minimum tumor pixels for slice to be included
        save_slices: Whether to save slices to disk (False for on-the-fly loading)
    
    Returns:
        List of dictionaries with slice metadata
    """
    # Get dimensions (assumes all volumes have same shape)
    height, width, depth = volumes['t1'].shape
    
    # Auto-detect slice range if not provided (focus on central slices with brain)
    if slice_range is None:
        # Find slices with significant brain tissue
        brain_slices = []
        for i in range(depth):
            if volumes['t1'][:, :, i].max() > 0:
                brain_slices.append(i)
        
        if len(brain_slices) > 0:
            start_slice = brain_slices[0] + 10  # Skip very top slices
            end_slice = brain_slices[-1] - 10   # Skip very bottom slices
            slice_range = (max(0, start_slice), min(depth, end_slice))
        else:
            slice_range = (depth // 4, 3 * depth // 4)
    
    slice_metadata = []
    
    if save_slices:
        # Create patient directory
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    for slice_idx in range(slice_range[0], slice_range[1]):
        # Extract slice from each modality
        t1_slice = volumes['t1'][:, :, slice_idx]
        t1ce_slice = volumes['t1ce'][:, :, slice_idx]
        t2_slice = volumes['t2'][:, :, slice_idx]
        flair_slice = volumes['flair'][:, :, slice_idx]
        seg_slice = volumes['seg'][:, :, slice_idx]
        
        # Check if slice has significant content
        has_brain = t1_slice.max() > 0
        has_tumor = has_significant_tumor(seg_slice, min_tumor_pixels)
        
        if not has_brain:
            continue  # Skip empty slices
        
        # Normalize imaging slices
        t1_norm = normalize_slice(t1_slice)
        t1ce_norm = normalize_slice(t1ce_slice)
        t2_norm = normalize_slice(t2_slice)
        flair_norm = normalize_slice(flair_slice)
        
        # Convert segmentation labels
        seg_converted = convert_brats_labels(seg_slice)
        
        if save_slices:
            # Save as numpy arrays
            slice_filename = f"slice_{slice_idx:03d}"
            np.save(patient_output_dir / f"{slice_filename}_t1.npy", t1_norm)
            np.save(patient_output_dir / f"{slice_filename}_t1ce.npy", t1ce_norm)
            np.save(patient_output_dir / f"{slice_filename}_t2.npy", t2_norm)
            np.save(patient_output_dir / f"{slice_filename}_flair.npy", flair_norm)
            np.save(patient_output_dir / f"{slice_filename}_seg.npy", seg_converted)
            
            # Create relative paths for metadata
            slice_paths = {
                't1': str(patient_output_dir.relative_to(output_dir) / f"{slice_filename}_t1.npy"),
                't1ce': str(patient_output_dir.relative_to(output_dir) / f"{slice_filename}_t1ce.npy"),
                't2': str(patient_output_dir.relative_to(output_dir) / f"{slice_filename}_t2.npy"),
                'flair': str(patient_output_dir.relative_to(output_dir) / f"{slice_filename}_flair.npy"),
                'seg': str(patient_output_dir.relative_to(output_dir) / f"{slice_filename}_seg.npy")
            }
        else:
            # Store original NIfTI paths for on-the-fly loading
            slice_paths = {
                'patient_dir': str(volumes.get('patient_dir', '')),
                'slice_idx': slice_idx
            }
        
        # Add to metadata
        slice_metadata.append({
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'has_tumor': int(has_tumor),
            'tumor_pixels': int((seg_slice > 0).sum()),
            **slice_paths
        })
    
    return slice_metadata


def process_brats_dataset(
    brats_root: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    save_slices: bool = True,
    min_tumor_pixels: int = 50,
    max_patients: Optional[int] = None
):
    """
    Process entire BRATs dataset: extract slices and create metadata.
    
    Args:
        brats_root: Path to BRATs root (e.g., 'Dataset/Extracted data/BRATs_2020/...')
        output_dir: Where to save processed data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        save_slices: Whether to save extracted slices (False for on-the-fly loading)
        min_tumor_pixels: Minimum tumor pixels for slice inclusion
        max_patients: Maximum number of patients to process (for testing)
    """
    brats_root = Path(brats_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if BRATs root exists
    if not brats_root.exists():
        raise FileNotFoundError(f"BRATs root directory not found: {brats_root}")
    
    # Find all patient directories
    patient_dirs = sorted(list(brats_root.glob("BraTS20_Training_*")))
    
    if len(patient_dirs) == 0:
        # Try alternative naming patterns
        patient_dirs = sorted(list(brats_root.glob("BraTS*_Training_*")))
    
    if len(patient_dirs) == 0:
        raise FileNotFoundError(
            f"No patient directories found in {brats_root}\n"
            f"Expected pattern: BraTS20_Training_* or BraTS*_Training_*\n"
            f"Please verify the directory structure."
        )
    
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
    
    print(f"Found {len(patient_dirs)} patients in {brats_root}")
    
    all_metadata = []
    failed_patients = []
    
    # Process each patient
    for patient_dir in tqdm(patient_dirs, desc="Processing BRATs patients"):
        try:
            # Load all modalities
            volumes = load_brats_patient(patient_dir)
            volumes['patient_dir'] = str(patient_dir)
            
            # Extract slices
            patient_id = patient_dir.name
            slice_metadata = extract_2d_slices(
                volumes=volumes,
                patient_id=patient_id,
                output_dir=output_dir / "slices" if save_slices else output_dir,
                min_tumor_pixels=min_tumor_pixels,
                save_slices=save_slices
            )
            
            all_metadata.extend(slice_metadata)
            
        except Exception as e:
            print(f"Error processing {patient_dir.name}: {e}")
            failed_patients.append(patient_dir.name)
            continue
    
    # Check if any data was extracted
    if len(all_metadata) == 0:
        error_msg = (
            f"No slices were extracted from {len(patient_dirs)} patients!\n"
            f"Failed patients: {len(failed_patients)}\n"
        )
        if failed_patients:
            error_msg += f"Examples of failed patients: {failed_patients[:5]}\n"
        error_msg += (
            "Possible issues:\n"
            "1. NIfTI files may be missing or corrupted\n"
            "2. File naming might not match expected pattern\n"
            "3. All slices might be empty (no brain tissue)\n"
        )
        raise ValueError(error_msg)
    
    # Create DataFrame
    df = pd.DataFrame(all_metadata)
    print(f"\nExtracted {len(df)} total slices from {len(patient_dirs) - len(failed_patients)} patients")
    print(f"Slices with tumor: {df['has_tumor'].sum()}")
    print(f"Slices without tumor: {len(df) - df['has_tumor'].sum()}")
    if failed_patients:
        print(f"Failed patients: {len(failed_patients)}")
    
    # Split into train/val/test
    patients = df['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(patients)
    
    n_train = int(len(patients) * train_ratio)
    n_val = int(len(patients) * val_ratio)
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    # Save metadata
    df.to_csv(output_dir / "all_metadata.csv", index=False)
    train_df.to_csv(output_dir / "train_metadata.csv", index=False)
    val_df.to_csv(output_dir / "val_metadata.csv", index=False)
    test_df.to_csv(output_dir / "test_metadata.csv", index=False)
    
    print(f"\nSplit summary:")
    print(f"Train: {len(train_df)} slices from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} slices from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} slices from {len(test_patients)} patients")
    
    # Save summary statistics
    summary = {
        'total_patients': len(patients),
        'total_slices': len(df),
        'slices_with_tumor': int(df['has_tumor'].sum()),
        'train_patients': len(train_patients),
        'train_slices': len(train_df),
        'val_patients': len(val_patients),
        'val_slices': len(val_df),
        'test_patients': len(test_patients),
        'test_slices': len(test_df),
    }
    
    with open(output_dir / "preprocessing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPreprocessing complete! Metadata saved to {output_dir}")
    return df, train_df, val_df, test_df


def visualize_brats_slice(
    volumes: Dict[str, np.ndarray],
    slice_idx: int,
    save_path: Optional[str] = None
):
    """
    Visualize all modalities and segmentation for a single slice.
    
    Args:
        volumes: Dictionary with modality volumes
        slice_idx: Slice index to visualize
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    modalities = ['t1', 't1ce', 't2', 'flair', 'seg']
    titles = ['T1', 'T1-CE', 'T2', 'FLAIR', 'Segmentation', 'Overlay']
    
    for idx, (modality, title) in enumerate(zip(modalities, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        slice_data = volumes[modality][:, :, slice_idx]
        
        if modality == 'seg':
            ax.imshow(slice_data, cmap='tab10', vmin=0, vmax=4)
        else:
            ax.imshow(slice_data, cmap='gray')
        
        ax.set_title(title)
        ax.axis('off')
    
    # Create overlay (T1-CE + segmentation)
    ax = axes[1, 2]
    t1ce_slice = volumes['t1ce'][:, :, slice_idx]
    seg_slice = volumes['seg'][:, :, slice_idx]
    
    ax.imshow(t1ce_slice, cmap='gray')
    masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
    ax.imshow(masked_seg, cmap='tab10', alpha=0.5, vmin=0, vmax=4)
    ax.set_title('T1-CE + Segmentation Overlay')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    BRATS_ROOT = "Dataset/Extracted data/BRATs_2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    OUTPUT_DIR = "Dataset/brats_preprocessed"
    
    # Process dataset (set save_slices=False for on-the-fly loading)
    process_brats_dataset(
        brats_root=BRATS_ROOT,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        save_slices=True,
        min_tumor_pixels=50,
        max_patients=None  # Set to small number for testing, None for all
    )
