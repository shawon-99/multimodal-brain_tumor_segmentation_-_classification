"""
BRATs 2020 Dataset Preprocessing
Loads multi-modal NIfTI volumes and extracts 2D slices for training
"""

import io
import gzip
import boto3
import botocore.exceptions
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def parse_s3_uri(uri: str):
    """Parse an S3 URI into (bucket, key). Normalises s3:/ → s3://."""
    if uri.startswith("s3:/") and not uri.startswith("s3://"):
        uri = "s3://" + uri[4:]
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key.rstrip("/")


def get_s3_client():
    """Return a boto3 S3 client. Auto-detects SageMaker IAM role or local credentials."""
    return boto3.client("s3")


def s3_list_patient_keys(s3_client, bucket: str, prefix: str) -> List[str]:
    """Return sorted list of patient S3 key prefixes (subdirectories) under prefix."""
    prefix = prefix.rstrip("/") + "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    dirs = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").split("/")[-1]
            if "Training_" in name:
                dirs.append(cp["Prefix"].rstrip("/"))
    return sorted(dirs)


def _s3_save_npy(s3_client, bucket: str, key: str, array: np.ndarray):
    """Serialize a numpy array to .npy format and upload to S3."""
    buf = io.BytesIO()
    np.save(buf, array)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def load_nifti_volume(nifti_path: str, s3_client=None) -> np.ndarray:
    """
    Load a NIfTI file from an S3 URI and return the volume as numpy array.
    """
    bucket, key = parse_s3_uri(nifti_path)
    raw = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    buf = io.BytesIO(raw)
    fh = nib.FileHolder(fileobj=buf)
    nifti_img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    volume = nifti_img.get_fdata()
    return volume


def load_brats_patient(patient_s3_prefix: str, s3_client) -> Dict[str, np.ndarray]:
    """
    Load all modalities and segmentation for a single BRATs patient from S3.
    patient_s3_prefix: full S3 URI to the patient directory,
                       e.g. s3://bucket/path/BraTS20_Training_001
    """
    bucket, prefix = parse_s3_uri(patient_s3_prefix)
    patient_id = prefix.split("/")[-1]

    volumes = {}
    for modality in ['t1', 't1ce', 't2', 'flair', 'seg']:
        uri = None
        for ext in [".nii", ".nii.gz"]:
            key = f"{prefix}/{patient_id}_{modality}{ext}"
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                uri = f"s3://{bucket}/{key}"
                break
            except botocore.exceptions.ClientError:
                continue
        if uri is None:
            raise FileNotFoundError(f"Missing {modality} file for patient {patient_id}")
        volumes[modality] = load_nifti_volume(uri, s3_client)

    return volumes


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize a single 2D slice using Z-score normalization.
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
    """
    tumor_pixels = (seg_slice > 0).sum()
    return tumor_pixels >= min_pixels


def extract_2d_slices(
    volumes: Dict[str, np.ndarray],
    patient_id: str,
    output_dir: str,
    s3_client,
    slice_range: Optional[Tuple[int, int]] = None,
    min_tumor_pixels: int = 50,
    save_slices: bool = True
) -> List[Dict]:
    """
    Extract 2D slices from 3D volumes and save to S3.
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
        output_bucket, output_prefix = parse_s3_uri(output_dir)
        patient_prefix = f"{output_prefix}/{patient_id}"

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
            # Save as numpy arrays to S3
            slice_filename = f"slice_{slice_idx:03d}"
            for suffix, arr in [
                ("_t1", t1_norm), ("_t1ce", t1ce_norm), ("_t2", t2_norm),
                ("_flair", flair_norm), ("_seg", seg_converted)
            ]:
                _s3_save_npy(s3_client, output_bucket,
                             f"{patient_prefix}/{slice_filename}{suffix}.npy", arr)

            slice_paths = {
                't1':   f"s3://{output_bucket}/{patient_prefix}/{slice_filename}_t1.npy",
                't1ce': f"s3://{output_bucket}/{patient_prefix}/{slice_filename}_t1ce.npy",
                't2':   f"s3://{output_bucket}/{patient_prefix}/{slice_filename}_t2.npy",
                'flair':f"s3://{output_bucket}/{patient_prefix}/{slice_filename}_flair.npy",
                'seg':  f"s3://{output_bucket}/{patient_prefix}/{slice_filename}_seg.npy",
            }
        else:
            # Store reference for on-the-fly loading
            slice_paths = {
                'patient_dir': volumes.get('patient_dir', ''),
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
    Process entire BRATs dataset from S3: extract slices and save metadata to S3.
    """
    s3 = get_s3_client()
    brats_bucket, brats_prefix = parse_s3_uri(brats_root)
    output_bucket, output_prefix = parse_s3_uri(output_dir)

    # Find all patient directories in S3
    patient_keys = s3_list_patient_keys(s3, brats_bucket, brats_prefix)

    if len(patient_keys) == 0:
        raise FileNotFoundError(
            f"No patient directories found under s3://{brats_bucket}/{brats_prefix}\n"
            f"Expected pattern: BraTS20_Training_* or BraTS*_Training_*\n"
            f"Please verify the S3 path and bucket permissions."
        )

    if max_patients:
        patient_keys = patient_keys[:max_patients]

    print(f"Found {len(patient_keys)} patients in s3://{brats_bucket}/{brats_prefix}")

    all_metadata = []
    failed_patients = []

    slices_output = f"s3://{output_bucket}/{output_prefix}/slices" if save_slices else output_dir

    # Process each patient
    for patient_key in tqdm(patient_keys, desc="Processing BRATs patients"):
        patient_s3_uri = f"s3://{brats_bucket}/{patient_key}"
        patient_id = patient_key.split("/")[-1]
        try:
            volumes = load_brats_patient(patient_s3_uri, s3)
            volumes['patient_dir'] = patient_s3_uri

            slice_metadata = extract_2d_slices(
                volumes=volumes,
                patient_id=patient_id,
                output_dir=slices_output,
                s3_client=s3,
                min_tumor_pixels=min_tumor_pixels,
                save_slices=save_slices
            )

            all_metadata.extend(slice_metadata)

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            failed_patients.append(patient_id)
            continue

    # Check if any data was extracted
    if len(all_metadata) == 0:
        error_msg = (
            f"No slices were extracted from {len(patient_keys)} patients!\n"
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
    print(f"\nExtracted {len(df)} total slices from {len(patient_keys) - len(failed_patients)} patients")
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

    # Save metadata CSVs to S3
    for name, frame in [("all_metadata", df), ("train_metadata", train_df),
                        ("val_metadata", val_df), ("test_metadata", test_df)]:
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/{name}.csv",
            Body=buf.getvalue().encode("utf-8")
        )

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

    s3.put_object(
        Bucket=output_bucket,
        Key=f"{output_prefix}/preprocessing_summary.json",
        Body=json.dumps(summary, indent=2).encode("utf-8")
    )

    print(f"\nPreprocessing complete! Metadata saved to s3://{output_bucket}/{output_prefix}/")
    return df, train_df, val_df, test_df


def visualize_brats_slice(
    volumes: Dict[str, np.ndarray],
    slice_idx: int,
    save_path: Optional[str] = None
):
    """
    Visualize all modalities and segmentation for a single slice.
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
    # Example usage with S3 paths
    BRATS_ROOT = "s3://sagemaker-us-east-2-826634839412/unzipped-data/Extracted data/BRATs_2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    OUTPUT_DIR = "s3://sagemaker-us-east-2-826634839412/unzipped-data/brats_preprocessed"

    process_brats_dataset(
        brats_root=BRATS_ROOT,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        save_slices=True,
        min_tumor_pixels=50,
        max_patients=None  # Set to small number for testing, None for all
    )
