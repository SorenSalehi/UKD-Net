# 🧠 SAXNet-Distill: Knowledge Distillation with SAXNet Teacher for 3D Brain Tumor Segmentation (BraTS 2023 & MSD)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS_2023_%7C_MSD_Brain-blue.svg)](https://www.synapse.org/#!Synapse:syn51156910)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end framework for high-accuracy, resource-efficient 3D brain tumor segmentation on benchmark datasets including **BraTS 2023** and **MSD (Medical Segmentation Decathlon - Task01 Brain Tumour)**. This repository implements **SAXNetV1** (a 3D U-Net enhanced with multi-scale Spatial and Channel Attention mechanisms) as a high-capacity Teacher model, which is subsequently distilled into **LATUPStudent**—an ultra-lightweight 3D model designed for fast, low-memory clinical inference.

> 💡 **Dataset Flexibility**: The entire pipeline seamlessly supports both **BraTS 2023** and **MSD (Task01 Brain Tumour)**. The data loader (`data/dataset.py`) automatically detects whether the dataset path points to BraTS or MSD structure and adapts modality parsing, pre-processing, and SDF caching accordingly. Simply supply your dataset directory via `--data-dir`.

---

## 🌟 Key Features & Innovations

1. **Teacher Architecture (SAXNetV1)**:
   - Integrates 3D Spatial Attention (`SpatialAttention3D`) and 3D Channel Squeeze-and-Excitation Attention (`ChannelAttention3D`) across every encoder/decoder stage.
   - Captures fine-grained boundaries for complex sub-regions (Whole Tumor, Tumor Core, Enhancing Tumor).

2. **Ultra-Lightweight Student (LATUPStudent)**:
   - Utilizes initial multi-scale parallel convolutions (`1x1`, `3x3`, `5x5` receptive fields) in `ParallelConvBlock`.
   - Compresses model size to **~1.52M parameters** (a **~31x parameter reduction** compared to the Teacher).

3. **Knowledge Distillation Formulation**:
   - **DWKD Loss (Decoupled Weighted Knowledge Distillation)** with temperature scaling.
   - **SDF-Guided Boundary Loss**: Enforces precise sub-voxel tumor margin alignment using precomputed Signed Distance Functions.
   - **Deep Supervision**: Multi-scale auxiliary heads supervising lower decoder layers.

---

## 📊 Comprehensive Evaluation & Benchmark Results

All evaluation benchmarks were strictly executed on **BraTS 2023 (Fold 4 Validation - 250 cases)** with an input ROI of `(128, 128, 128)`.

### 1. Segmentation Quality (Student Model on 250 Test Cases)

| Target Sub-Region | Dice Score ↑ | HD95 (mm) ↓ | NSD (2mm) ↑ | Precision ↑ | Recall ↑ | Specificity ↑ | Lesion Recall ↑ | Abs RVE ↓ | Vol MAE (mL) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Whole Tumor (WT)** | **0.9279** | **4.22** | **0.9153** | 0.9524 | 0.9123 | 0.9982 | 0.7181 | 0.0811 | 6.51 |
| **Tumor Core (TC)** | **0.9217** | **3.00** | **0.9212** | 0.9345 | 0.9270 | 0.9989 | 0.8045 | 0.1112 | 2.67 |
| **Enhancing Tumor (ET)**| **0.8703** | **6.71** | **0.9314** | 0.8754 | 0.8844 | 0.9992 | 0.7833 | 0.2306 | 1.68 |
| **Mean / Overall** | **0.9066** | **4.65** | **0.9226** | **0.9208** | **0.9079** | **0.9988** | **0.7687** | **0.1403** | **3.62** |

---

### 2. Efficiency & Computational Footprint (Teacher vs. Distilled Student)

Benchmarked on NVIDIA GPU with input tensor size `(1, 4, 128, 128, 128)`:

| Model Architecture | Params (M) ↓ | MACs (G) ↓ | TFLOPS ↓ | Latency (ms) ↓ | Infer Mem (MB) ↓ | Train Mem (MB) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher (SAXNetV1)** | 47.15 M | 1027.24 G | 2.0545 | 1093.75 ms | 2460.0 MB | 9637.2 MB |
| **Student (LATUP)** | **1.52 M** | **213.61 G** | **0.4272** | **253.38 ms** | **1854.7 MB** | **4363.5 MB** |
| **Efficiency Gain** | **31.0x Smaller** | **4.8x Fewer** | **4.8x Lower** | **4.3x Faster** | **24.6% Saved** | **54.7% Saved** |

---

## 📁 Repository Structure

```text
├── data/
│   └── dataset.py              # Multi-dataset loader (BraTS & MSD), augmentations & SDF generation
├── models/
│   ├── saxnet.py               # SAXNetV1 architecture (Channel & Spatial Attention 3D)
│   └── student.py              # LATUPStudent (Parallel Conv & Deep Supervision)
├── utils/
│   └── loss_metrics.py         # DWKD, Boundary Loss, Dice, HD95, NSD, Lesion Recall
├── train.py                    # Unified Teacher pre-training & Student distillation pipeline
├── evaluate.py                 # Full metric & efficiency benchmarking suite
├── requirements.txt            # Python dependencies
└── README.md
