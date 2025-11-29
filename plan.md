# Brain Tumor Dataset - EDA Plan

## Overview
Exploratory Data Analysis for brain tumor MRI datasets to understand data structure, distribution, and characteristics.

## File Structure

### `eda.py`
**Purpose**: Utility functions for data loading and exploration

**Key Functions**:
- `load_dataset_info()` - Scan and summarize dataset structure
- `get_class_distribution()` - Count images per class/category

### `main.ipynb`
**Purpose**: Interactive notebook for visualization and analysis

**Sections**:
1. **Dataset Overview**
   - List available datasets (Brain_Tumor_MRI_Dataset, Brain_Tumor_MRI_Scans, BRATs_2020)
   - Total images count

2. **Class Distribution**
   - Bar charts showing image counts per category
   - Training vs Testing splits (for applicable datasets)

3. **Sample Visualization**
   - Display 3-5 sample images from each class

## Implementation Notes
- Use minimal dependencies: `os`, `PIL/cv2`, `matplotlib`, `pandas`
- Focus on three datasets: Brain_Tumor_MRI_Dataset, Brain_Tumor_MRI_Scans, BRATs_2020
- Keep code simple and reusable
- No model training or preprocessing in this phase
