# Brain Tumor Segmentation & Classification - Project Plan
**Multi-Modal Transformer Network with Domain Generalization**

## 📋 Project Overview
**Goal**: Develop a robust brain tumor segmentation and classification system using Vision Transformer (ViT) architecture with domain generalization capabilities to work across different MRI datasets and scanners.

**Datasets**:
1. Brain_Tumor_MRI_Dataset (Binary: tumor vs no tumor)
2. Brain_Tumor_MRI_Scans (Multi-class: glioma, healthy, meningioma, pituitary)
3. BRATs_2020 (Segmentation: T1, T2, FLAIR, T1-CE multi-modal)

---

## ✅ Completed Work

### Phase 1: Data Preparation & Preprocessing
- [x] **Dataset Organization**
  - Organized 3 datasets in `Dataset/Extracted data/`
  - EDA implementation (`eda.py`)
  - Class distribution analysis
  - Total images: 8,277 across all datasets

- [x] **Preprocessing Pipeline**
  - Image resizing to 224×224 (`preprocessing.py`)
  - Z-score normalization (per-channel)
  - Train/Val/Test split (70/15/15)
  - Metadata CSV generation
  - Batch processing of all images
  - Files: `preprocessing.py`, `main.ipynb`

- [x] **BRATs Dataset Preprocessing**
  - NIfTI volume loading (`nibabel`)
  - 2D slice extraction from 3D volumes
  - Multi-modal MRI stacking (T1, T1-CE, T2, FLAIR)
  - Segmentation mask conversion (BRATs → 4-class format)
  - Automated patient-level train/val/test split
  - File: `brats_preprocessing.py`

- [x] **Data Augmentation**
  - Rotation (±30°)
  - Horizontal/Vertical flipping
  - Random zoom (0.8-1.2×)
  - Intensity variation (±10%)
  - Elastic deformation (for segmentation)
  - Synchronized augmentation for image+mask
  - Functions: `augment_rotation()`, `augment_flip()`, `augment_zoom()`, `augment_intensity()`, `augment_elastic_deform()`

### Phase 2: Model Architecture
- [x] **Vision Transformer Implementation**
  - Patch embedding (16×16 patches, 196 total)
  - Multi-head self-attention (12 heads)
  - Transformer encoder (12 layers, 768 dim)
  - Positional embeddings
  - Classification head (binary & multi-class)
  - Segmentation head (with decoder)
  - File: `model.py`

- [x] **Model Variants**
  - ViT Binary Classification (~86M parameters)
  - ViT Multi-class Classification (~86M parameters)
  - ViT Segmentation with decoder (~95M parameters)
  - Support for 3-channel RGB and 4-channel multi-modal input

- [x] **Model Testing**
  - Architecture validation
  - Forward pass testing
  - Real image inference
  - File: `model_training.ipynb`

### Phase 3: Data Loading & Training Infrastructure

- [x] **Custom Dataset & DataLoader**
  - `BrainTumorDataset` class for classification
  - `BraTSSegmentationDataset` class for segmentation
  - Z-score normalization integration
  - On-the-fly augmentation during training
  - Multi-modal MRI loading (4-channel)
  - Class imbalance handling
  - Domain label support
  - File: `dataset.py`

- [x] **Training Loop Implementation - Classification**
  - `Trainer` class for classification
  - Forward/backward pass with gradient clipping
  - Validation loop with metrics tracking
  - Model checkpointing (best + periodic)
  - TensorBoard logging
  - Early stopping (patience=15)
  - File: `train.py`

- [x] **Training Loop Implementation - Segmentation**
  - `SegmentationTrainer` class
  - Segmentation-specific loss handling
  - Dice score and IoU tracking
  - Per-class metric logging
  - Visualization during training
  - File: `train.py`

- [x] **Loss Functions**
  - CrossEntropyLoss for classification
  - DiceLoss for segmentation
  - FocalLoss for class imbalance
  - CombinedLoss (Dice + CrossEntropy)
  - TverskyLoss for FP/FN control
  - File: `losses.py`

- [x] **Segmentation Metrics**
  - Dice coefficient (per-class)
  - IoU (Intersection over Union)
  - Pixel accuracy
  - Sensitivity & Specificity
  - Hausdorff distance (95th percentile)
  - `SegmentationMetrics` accumulator class
  - File: `seg_metrics.py`

- [x] **Training Configuration**
  - Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
  - Learning rate scheduler: Cosine annealing
  - Gradient clipping (max_norm=1.0)
  - Early stopping support

- [x] **Training Notebooks**
  - Classification: `train_model.ipynb`
  - Segmentation: `segmentation_training.ipynb`

### Phase 3 (Current): Model Training & Evaluation

- [x] **Initial Classification Training**
  - Binary classifier trained on Brain_Tumor_MRI_Dataset
  - 10 epochs completed
  - Validation accuracy: 93.48%
  - ⚠️ Test accuracy: 43.40% (severe overfitting issue)
  - Checkpoints saved in `experiments/checkpoints/`

---

## 🔄 Next Steps

### Phase 4: Segmentation Training & Evaluation (READY TO START)

#### 4.1 Preprocess BRATs Dataset
- [ ] **Run BRATs Preprocessing**
  - Execute `brats_preprocessing.py` on full dataset
  - Extract 2D slices from ~369 patients
  - Generate train/val/test metadata
  - Verify slice quality and class distribution

#### 4.2 Train Segmentation Model
- [ ] **Initial Training**
  - Use `segmentation_training.ipynb`
  - Train ViT segmentation model on BRATs
  - Monitor Dice score and IoU
  - Target: Dice >0.85 on validation set

- [ ] **Hyperparameter Tuning**
  - Try different loss functions (Dice, Combined, Focal)
  - Adjust learning rate and batch size
  - Experiment with augmentation strategies

#### 4.3 Segmentation Evaluation
- [ ] **Test Set Evaluation**
  - Compute Dice, IoU, Hausdorff distance
  - Per-class metrics (whole tumor, core, enhancing)
  - Visualize predictions vs ground truth
  - Generate segmentation quality report

### Phase 5: Fix Classification Overfitting

#### 5.1 Diagnose Current Issues
- [ ] **Investigate Train-Test Gap**
  - Verify data split integrity (no leakage)
  - Check preprocessing consistency
  - Analyze test set distribution
  - Visualize misclassified samples

#### 5.2 Improve Regularization
- [ ] **Model Modifications**
  - Increase dropout rate (0.1 → 0.2)
  - Add stronger data augmentation
  - Try pretrained ViT weights (transfer learning)
  - Reduce model size (ViT-Small instead of ViT-Base)

#### 5.3 Retrain Classification Model
- [ ] **Improved Training**
  - Apply fixes from diagnosis
  - Train with better regularization
  - Monitor train-val-test performance closely
  - Target: Test accuracy >85%

### Phase 6: Domain Generalization Mechanisms

### Phase 6: Domain Generalization Mechanisms

#### 6.1 Domain Adversarial Training
- [ ] **Domain Discriminator**
  - Gradient Reversal Layer (GRL)
  - Domain classifier network
  - Adversarial loss function
  - Balance task loss and domain loss (λ parameter)

- [ ] **Domain Labels**
  - Assign domain IDs to each dataset
  - Multi-source domain training
  - Domain confusion objective

#### 6.2 Multi-Task Learning (Classification + Segmentation)
- [ ] **Unified Architecture**
  - Shared ViT encoder for both tasks
  - Dual heads: classification + segmentation
  - Multi-task loss balancing
  - File: Update `model.py` with `ViTMultiTask`

- [ ] **Training Strategy**
  - Mixed-batch sampling (classification & segmentation)
  - Alternating task optimization
  - Joint training experiments

#### 6.3 Feature Alignment
- [ ] **Alignment Loss Functions**
  - Maximum Mean Discrepancy (MMD)
  - Correlation Alignment (CORAL)
  - Multi-kernel feature matching

- [ ] **Style Transfer**
  - Instance normalization per domain
  - Adaptive Instance Normalization (AdaIN)
  - Domain randomization

### Phase 7: Multi-Domain Training Strategy

#### 7.1 Cross-Dataset Experiments
- [ ] **Train on Multiple Datasets**
  - Joint training on all 3 datasets
  - Domain adversarial training
  - Leave-one-domain-out evaluation
  - Measure generalization gap

#### 7.2 Transfer Learning
- [ ] **Pre-training & Fine-tuning**
  - Pre-train on largest dataset (BRATs)
  - Fine-tune on smaller datasets
  - Progressive unfreezing
  - Learning rate scheduling

### Phase 8: Evaluation & Testing

#### 8.1 Cross-Dataset Evaluation
- [ ] **Leave-One-Domain-Out Protocol**
  - Train on 2 datasets, test on 3rd
  - All 3 combinations
  - Measure generalization gap
  - Report cross-domain performance

#### 8.2 Metrics Computation
- [x] **Classification Metrics**
  - Accuracy, Precision, Recall, F1-score
  - AUC-ROC curve
  - Per-class performance
  - Confusion matrix
  - File: `evaluate.py`

- [x] **Segmentation Metrics**
  - Dice coefficient (per region)
  - IoU (Intersection over Union)
  - Hausdorff Distance (95th percentile)
  - Sensitivity & Specificity
  - File: `seg_metrics.py`

#### 8.3 Ablation Studies
- [ ] **Component Analysis**
  - ViT vs CNN baseline (ResNet, EfficientNet)
  - With/without domain adversarial training
  - With/without multi-task learning
  - With/without feature alignment
  - Impact of Z-score normalization
  - Effect of data augmentation

### Phase 9: Results & Visualization

### Phase 9: Results & Visualization

#### 9.1 Performance Visualization
- [ ] **Training Curves**
  - Loss curves (train/val)
  - Accuracy curves
  - Learning rate schedule
  - Domain confusion metrics

- [ ] **Prediction Visualization**
  - Confusion matrices
  - ROC curves
  - Example predictions (correct & incorrect)
  - Attention map visualization

- [ ] **Segmentation Visualization**
  - Ground truth vs prediction overlay
  - Multi-modal input visualization
  - Per-class segmentation quality

#### 9.2 Results Analysis
- [ ] **Performance Tables**
  - Single-domain results
  - Cross-domain results
  - Comparison with baselines
  - Ablation study results

- [ ] **Statistical Analysis**
  - Mean ± standard deviation
  - Statistical significance tests
  - Confidence intervals

### Phase 10: Documentation & Thesis Writing

#### 10.1 Code Documentation
- [ ] **README.md**
  - Project overview
  - Installation instructions
  - Usage examples
  - Model architecture diagram
  - Results summary

- [ ] **Code Comments**
  - Docstrings for all functions
  - Inline comments for complex logic
  - Type hints

#### 10.2 Thesis Sections
- [ ] **Methodology Chapter**
  - Architecture description
  - Domain generalization techniques
  - Training procedure

- [ ] **Results Chapter**
  - Quantitative results
  - Qualitative analysis
  - Comparison with literature

- [ ] **Discussion & Conclusion**
  - Findings summary
  - Limitations
  - Future work

---

## 📂 Project Structure

```
E:\Thesis\
├── Dataset/
│   ├── Extracted data/
│   │   ├── Brain_Tumor_MRI_Dataset/
│   │   ├── Brain_Tumor_MRI_Scans/
│   │   └── BRATs_2020/
│   ├── preprocessed_data/                  # ✅ Classification data
│   │   ├── train/ (no tumor, tumor)
│   │   ├── val/
│   │   ├── test/
│   │   └── *.csv (metadata)
│   └── brats_preprocessed/                 # ✅ Segmentation data
│       ├── slices/                         # Extracted 2D slices
│       └── *.csv (metadata)
│
├── experiments/                            # ✅ Training outputs
│   ├── checkpoints/                        # Classification checkpoints
│   │   ├── best_model.pth
│   │   └── training_history.json
│   ├── segmentation_checkpoints/          # Segmentation checkpoints
│   ├── logs/                              # TensorBoard logs
│   ├── segmentation_logs/
│   └── results/                           # Evaluation results
│       └── segmentation_results/
│
├── src/                                    # ✅ Source code
│   ├── eda.py                             # ✅ Exploratory data analysis
│   ├── preprocessing.py                   # ✅ Preprocessing & augmentation
│   ├── brats_preprocessing.py             # ✅ BRATs NIfTI processing
│   ├── model.py                           # ✅ ViT architecture
│   ├── dataset.py                         # ✅ PyTorch Dataset classes
│   ├── train.py                           # ✅ Training loops (classification + segmentation)
│   ├── losses.py                          # ✅ Segmentation loss functions
│   ├── seg_metrics.py                     # ✅ Segmentation metrics
│   ├── evaluate.py                        # ✅ Evaluation metrics
│   └── utils.py                           # ✅ Utility functions
│
├── notebooks/
│   ├── main.ipynb                         # ✅ EDA & preprocessing
│   ├── model_training.ipynb               # ✅ Model testing
│   ├── train_model.ipynb                  # ✅ Classification training
│   └── segmentation_training.ipynb        # ✅ Segmentation training
│
├── requirements.txt                        # ✅ Dependencies (updated)
├── .gitignore                             # ✅ Git ignore rules
└── plan.md                                # ✅ This file (updated)
```

---

## 🎯 Immediate Next Actions

**Priority 1: Start Segmentation Pipeline**

1. **Preprocess BRATs Dataset** (30-60 minutes):
   ```python
   # Run in notebook or script
   from src.brats_preprocessing import process_brats_dataset
   
   process_brats_dataset(
       brats_root='Dataset/Extracted data/BRATs_2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
       output_dir='Dataset/brats_preprocessed',
       save_slices=True,
       max_patients=None  # Process all patients
   )
   ```

2. **Train Segmentation Model**:
   - Open `notebooks/segmentation_training.ipynb`
   - Follow the step-by-step workflow
   - Monitor Dice score and IoU metrics
   - Target: Dice >0.85 on validation set

3. **Evaluate Segmentation Results**:
   - Test set evaluation
   - Visualize predictions
   - Generate results report

**Priority 2: Fix Classification Overfitting**

4. **Investigate Train-Test Gap**:
   - Verify data split (check for leakage)
   - Analyze test set distribution
   - Visualize misclassified examples

5. **Retrain with Improvements**:
   - Increase dropout or reduce model size
   - Add stronger regularization
   - Use pretrained weights if available

---

## 📊 Expected Timeline (Updated)

| Phase | Duration | Status |
|-------|----------|--------|
| Data Preparation | Week 1-2 | ✅ Complete |
| Model Architecture | Week 3-4 | ✅ Complete |
| Data Loading & Training Infrastructure | Week 5-6 | ✅ Complete |
| Initial Classification Training | Week 7 | ✅ Complete (needs fixing) |
| **Segmentation Implementation** | **Week 8** | **🔄 In Progress** |
| Fix Classification & Train Segmentation | Week 9-10 | ⏳ Next |
| Multi-Task Learning | Week 11 | ⏳ Pending |
| Domain Generalization | Week 12-13 | ⏳ Pending |
| Cross-Domain Evaluation | Week 14 | ⏳ Pending |
| Results & Analysis | Week 15-16 | ⏳ Pending |
| Thesis Writing | Week 17-18 | ⏳ Pending |

---

## 🔧 Dependencies (Updated)

```bash
# Core ML/DL
torch>=2.0.0
torchvision>=0.15.0
einops>=0.7.0
tensorboard>=2.14.0

# Data processing
matplotlib
pandas
Pillow
numpy
scikit-learn
scipy

# Medical imaging
nibabel>=5.0.0          # ✅ Installed - NIfTI file handling
SimpleITK>=2.3.0        # ✅ Installed - Medical image processing

# Utilities
tqdm>=4.65.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 📈 Success Metrics

### Classification:
- **Single-domain accuracy**: >90%
- **Cross-domain accuracy**: >85%
- **F1-score**: >0.85
- **Current Status**: Overfitting issue (Val: 93.48%, Test: 43.40%) ⚠️

### Segmentation:
- **Dice coefficient**: >0.85 (single-domain), >0.80 (cross-domain)
- **IoU**: >0.75
- **Hausdorff Distance**: <5mm
- **Current Status**: Ready to train ✅

### Domain Generalization:
- **Generalization gap**: <10%
- **Domain confusion**: >80%
- **Current Status**: Not started ⏳

---

## 📝 Implementation Notes

### Completed:
- ✅ Full segmentation pipeline implemented (preprocessing, dataset, losses, metrics, trainer)
- ✅ Classification training infrastructure complete
- ✅ Multi-modal MRI support (4-channel input)
- ✅ Comprehensive metrics for both tasks
- ✅ TensorBoard logging integrated
- ✅ Both classification and segmentation notebooks ready

### Known Issues:
- ⚠️ Classification model severe overfitting (50% train-test gap)
- ⚠️ Need to verify data split integrity
- ⚠️ May need pretrained weights or smaller model

### Next Priorities:
1. 🔥 Run BRATs preprocessing (30-60 min)
2. 🔥 Train segmentation model (expect Dice >0.85)
3. 🔥 Fix classification overfitting
4. 🔥 Implement multi-task learning (shared encoder)
5. 🔥 Add domain adversarial training

### Architecture Flexibility:
- Both classification (RGB, 3-channel) and segmentation (multi-modal, 4-channel) supported
- Can train independently or jointly (multi-task)
- Ready for domain generalization experiments
