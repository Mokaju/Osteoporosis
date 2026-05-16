# 🦷 AI-Based Osteoporosis Detection from Dental Panoramic Radiographs

> Automated detection and severity classification of osteoporosis in females using deep learning on dental panoramic radiographs (DPRs) — an affordable alternative to DEXA scanning.

[![Journal](https://img.shields.io/badge/Journal-Imaging%20Informatics%20in%20Medicine-blue)](https://doi.org/10.1007/s10278-025-01809-8)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs10278--025--01809--8-green)](https://doi.org/10.1007/s10278-025-01809-8)
[![Data](https://img.shields.io/badge/Dataset-Harvard%20Dataverse-orange)](https://doi.org/10.7910/DVN/NVPPRA)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

---

## 📌 Overview

Osteoporosis affects millions globally, yet remains widely undiagnosed — especially in low-resource settings where DEXA scanners are expensive and scarce. This project presents an AI pipeline that leverages **routine dental panoramic radiographs (DPRs)** for early osteoporosis screening, making diagnosis accessible at a fraction of the cost.

**Key contributions:**
- A **U-Net–inspired segmentation model** for automatic extraction of the mandibular cortex region of interest (ROI) from DPRs
- A **CNN feature extractor** combined with three classifiers: DenseNet, SVM (SVC), and Random Forest (RFC)
- Support for both **binary classification** (osteoporosis vs. normal) and **three-class grading** (normal / mild / severe)
- Training on **~18,000 augmented images** derived from 919 real DPRs

---

## 📊 Results Summary

### Binary Classification (Osteoporosis vs. Normal)

| Model         | Accuracy | Sensitivity | Specificity | Precision | F1 Score |
|---------------|----------|-------------|-------------|-----------|----------|
| CNN-DenseNet  | **0.97** | 0.96        | 0.98        | 0.98      | **0.97** |
| CNN-RF        | 0.94     | 0.93        | 0.96        | 0.96      | 0.94     |
| CNN-SVC       | 0.94     | 0.93        | 0.95        | 0.95      | 0.94     |

### Three-Class Classification (Normal / Mild / Severe)

| Model         | Accuracy | Sensitivity | Specificity | Precision | F1 Score |
|---------------|----------|-------------|-------------|-----------|----------|
| CNN-DenseNet  | **0.96** | 0.97        | 0.94        | 0.94      | **0.95** |
| CNN-RF        | 0.94     | 0.93        | 0.94        | 0.93      | 0.93     |
| CNN-SVC       | 0.91     | 0.90        | 0.91        | 0.90      | 0.90     |

### Segmentation Model (U-Net)

| Metric | Score |
|--------|-------|
| DSC    | 0.91  |
| IoU    | 0.86  |
| SSIM   | 0.99  |

---

## 🏗️ Project Pipeline

```
DPRs (919 images)
       │
       ▼
  Data Labeling (Klemetti's C1/C2/C3 classification)
       │
       ▼
  Manual ROI Drawing (AFNI) → Binary Mask Export
       │
       ▼
  U-Net Segmentation Model Training
       │
       ▼
  Automatic ROI Extraction (400×200 px patches)
       │
       ▼
  Data Augmentation (rotation ±20°, center crop → 200×100 px)
  ~18,000 augmented images
       │
       ├──────────────────────────┐
       ▼                          ▼
  CNN-DenseNet              CNN Feature Extractor
  (end-to-end)               (1280 features)
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                       CNN-SVC           CNN-RF
```

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                    # Original DPR images (see Dataset section)
│   ├── masks/                  # Binary ROI masks from AFNI
│   ├── patches/                # Extracted 400×200 px patches
│   └── augmented/              # Augmented 200×100 px images for training
│
├── segmentation/
│   ├── unet_model.py           # U-Net architecture definition
│   ├── train_segmentation.py   # Training script for segmentation
│   └── predict_segmentation.py # Run inference on new DPRs
│
├── classification/
│   ├── cnn_feature_extractor.py    # Shared CNN backbone
│   ├── cnn_densenet_binary.py      # Binary classifier (DenseNet head)
│   ├── cnn_densenet_multiclass.py  # 3-class classifier (DenseNet head)
│   ├── cnn_svc.py                  # CNN + SVM classifier
│   ├── cnn_rf.py                   # CNN + Random Forest classifier
│   └── train_classifiers.py        # Training entry point
│
├── utils/
│   ├── augmentation.py         # Data augmentation utilities
│   ├── metrics.py              # DSC, IoU, SSIM, confusion matrix
│   └── preprocessing.py        # Image loading and centroid extraction
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_training.ipynb
│   └── 03_classification_training.ipynb
│
├── results/
│   ├── confusion_matrices/
│   └── learning_curves/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 
- CUDA-compatible GPU (recommended but not necessary)

### Install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### `requirements.txt`

```
tensorflow
scikit-learn
opencv-python
pandas
numpy
matplotlib
seaborn
scikit-image
```

> **Note:** AFNI (v25.2.06) is required only for manual ROI labeling. Pre-labeled masks are included in the dataset download.

---

## 📦 Dataset

The dataset of 919 DPRs (acquired using the Carestream CS 9300C Select system) is publicly available:

**Harvard Dataverse:** [https://doi.org/10.7910/DVN/NVPPRA](https://doi.org/10.7910/DVN/NVPPRA)

| Group | Age Range | Count | Description |
|-------|-----------|-------|-------------|
| G1    | 25–45 yrs | 456   | Control group (younger females) |
| G2    | >45 yrs   | 463   | Study group (post-menopausal females) |

Labels follow **Klemetti's classification**:
- **C1** — Normal cortex (no osteoporosis)
- **C2** — Mild erosion (mild osteoporosis)
- **C3** — Severe erosion (severe osteoporosis)

---

## 🚀 Usage

### Step 1: Run Segmentation on a DPR

```python
from segmentation.predict_segmentation import predict_roi

mask = predict_roi("path/to/dpr_image.png", model_weights="segmentation/weights/unet.h5")
```

### Step 2: Extract ROI Patch

```python
from utils.preprocessing import extract_patch

patch = extract_patch(image="path/to/dpr_image.png", mask=mask, patch_size=(400, 200))
```

### Step 3: Classify Osteoporosis Severity

```python
from classification.cnn_densenet_multiclass import predict

label, confidence = predict(patch, model_weights="classification/weights/cnn_densenet_3class.h5")
# Returns: ("normal" | "mild" | "severe", confidence_score)
```

### Step 4: Training from Scratch

```bash
# Train segmentation model
python segmentation/train_segmentation.py --data_dir data/masks/ --epochs 100

# Train classification models
python classification/train_classifiers.py --mode binary --model densenet
python classification/train_classifiers.py --mode multiclass --model densenet
python classification/train_classifiers.py --mode binary --model svc
python classification/train_classifiers.py --mode binary --model rf
```

---

## 🧠 Model Architectures

### U-Net Segmentation Model
- **Encoder:** 4× Conv2D layers (16 → 32 → 64 → 128 filters) + MaxPooling
- **Bottleneck:** Conv2D (256 filters)
- **Decoder:** 4× Conv2DTranspose (128 → 64 → 32 → 16 filters) + skip connections
- **Output:** Binary mask (256×512 px)
- **Total Parameters:** 958,577

### CNN Feature Extractor
- 4× Conv2D layers (32 filters each, kernel 3×3) + MaxPooling
- Output: 1,280-dimensional feature vector per image (200×100 px input)
- **Total Parameters:** 28,064

### CNN-DenseNet Classifier
- CNN Feature Extractor → Flatten → Dense(512) → Output
- Binary: sigmoid activation + binary cross-entropy
- 3-class: softmax activation + categorical cross-entropy
- **Total Parameters:** ~684,449 (binary) / ~685,475 (3-class)

---

## 📐 Training Configuration

| Hyperparameter | Segmentation | Classification |
|----------------|-------------|----------------|
| Optimizer      | ADAM (lr=0.0001) | ADAM (lr=0.0001) |
| Loss Function  | Binary cross-entropy | Binary / Categorical cross-entropy |
| Batch Size     | 8           | 32             |
| Epochs         | 100         | 100            |
| Regularization | L2          | L2             |
| Dropout        | 0.1         | 0.1            |
| Validation     | 5-fold CV   | 5-fold CV      |

---

## 📈 Hardware Used

| Hardware | Specs |
|----------|-------|
| NVIDIA DGX H100 Server | 8× 80 GB H100 GPUs (640 GB total), 2 TB RAM, Dual Intel Xeon Platinum (112 cores) |
| MSI Laptop (dev) | Intel Core i7-11800H, 16 GB RAM, NVIDIA RTX 3050 Ti |

---

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{meitei2026osteoporosis,
  title     = {AI-Based Automated Detection and Classification of Osteoporosis in Females from Dental Panoramic Radiographs},
  author    = {Meitei, Chanambam Mokaju and Sawhney, Hemant and Kumar, Ashok},
  journal   = {Journal of Imaging Informatics in Medicine},
  year      = {2026},
  doi       = {10.1007/s10278-025-01809-8},
  publisher = {Springer}
}
```

---

## 🏛️ Ethics & Approvals

- Approved by the **Institutional Ethics Committee (IEC)**, School of Medical Sciences and Research, Sharda University, Greater Noida, India
- IEC Reference No.: **SU/SMS&R/76-A/2024/80**
- Conducted in accordance with **ICMR Guidelines 2017**, the **Belmont Report**, and the **Declaration of Helsinki**
- Written informed consent obtained from all participants

---

## ⚠️ Limitations

- Dataset sourced from a **single institution** in India; generalizability to other ethnicities and geographies requires validation
- Restricted to **adult female patients**; not validated for males or other demographics
- **Cross-sectional study** — no longitudinal follow-up to validate fracture prevention outcomes
- External validation on datasets from different imaging equipment or centers is yet to be performed

---

## 🔭 Future Work

- Expand dataset with multi-center, multi-ethnic DPR collections
- Integrate **explainable AI (XAI)** techniques (Grad-CAM, SHAP) for clinical transparency
- Develop a **clinical-grade UI** for seamless integration into dental practice PACS workflows
- Extend methodology to detect other bone and dental diseases

---

## 👥 Authors

| Author | Role | Affiliation |
|--------|------|-------------|
| **Chanambam Mokaju Meitei** | Data processing, model development, manuscript writing | Sharda University, Greater Noida |
| **Hemant Sawhney** | Conceptualization, data acquisition | Sharda University, Greater Noida |
| **Ashok Kumar** | Project supervision, critical revision | Sharda University, Greater Noida |

📧 Contact: ashok.kumar6@sharda.ac.in

---

## 📜 License

This project is for academic and research use. The dataset is distributed under Harvard Dataverse terms. Please refer to the published paper for full usage rights.

---

*Center for Artificial Intelligence in Medicine, Imaging & Forensics — Sharda University, Greater Noida, Uttar Pradesh, India*
