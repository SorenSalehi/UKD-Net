# UKD-Net

**Teacher–Student Disagreement-Aware Knowledge Distillation from a Spatial-Channel Attention Network for Ultra-Lightweight 3D Brain Tumour Segmentation**

UKD-Net distils a high-capacity attention teacher (**SAXNet**, 47.15 M parameters) into an ultra-compact 3D student (**LATUP**, 1.52 M parameters) for multi-parametric MRI brain tumour segmentation (WT / TC / ET). The student is **31× smaller** than its teacher and runs with 213.61 GMACs on a 4×128³ input, which makes it suitable for resource-constrained clinical workstations and edge settings.

> **Relation to our earlier work.** UKD-Net is the extended and improved version of our MICCAI work **Uni-Light**:
> *Kuang, Salehi, Wu, Tafti, Moemeni. "Uni-Light: An Ultra-Lightweight Framework via Uncertainty-Aware Knowledge Distillation for Brain Tumour Segmentation."* [arXiv:2609.06729](https://arxiv.org/abs/2609.06729) · [Uni-Light code](https://github.com/libinglilykuang/UniLight-miccai)
>
> UKD-Net keeps the compact LATUP student and the SDF boundary loss, replaces the Residual 3D U-Net teacher with the attention-based SAXNet, and adds a more thorough evaluation (see [What is new](#what-is-new-compared-with-uni-light)).

---

## Highlights

- **SAXNet teacher**: residual U-Net whose every encoder/decoder stage and bottleneck contains a *SAX block* (3D channel attention → 3D spatial attention, CBAM-style, no decoder gating signal).
- **LATUP student**: Parallel Convolution Module (1×1×1, 3×3×3, 5×5×5 branches) + hierarchical deep supervision, base width 16.
- **Disagreement-aware dynamic weighted KD (DWKD)**: voxel-wise weights come from the cross-entropy between the softened teacher and student predictions (τ = 3.0), so supervision focuses on voxels where the two networks currently disagree.
- **SDF boundary loss** for geometric supervision along tumour margins.
- Evaluation with **DSC, HD95, NSD (τ = 1 mm), Precision/Recall, Lesion Recall, Abs RVE, Vol MAE**, paired statistical tests, ablations and complexity profiling on **BraTS 2023-GLI** and **MSD Task01 Brain Tumour**.

## Results

5-fold cross-validation split (`KFold(5, shuffle=True, random_state=42)`), evaluated on the held-out **fold 4** (250 BraTS cases, 96 MSD cases).

| Dataset | Model | Params | DSC ET | DSC WT | DSC TC | DSC Avg | HD95 Avg (mm) |
|---|---|---|---|---|---|---|---|
| BraTS 2023-GLI | SAXNet teacher | 47.15 M | 87.47 | 94.20 | 92.81 | 91.49 | 4.71 |
| BraTS 2023-GLI | **LATUP student (UKD-Net)** | **1.52 M** | 87.03 | 92.79 | 92.17 | **90.66** | 4.65 |
| MSD-BTS | SAXNet teacher | 47.15 M | 79.58 | 89.94 | 84.54 | 84.69 | 6.39 |
| MSD-BTS | **LATUP student (UKD-Net)** | **1.52 M** | 79.98 | 90.21 | 85.34 | **85.18** | 5.47 |

Student NSD (τ = 1 mm), mean over regions: **86.87 %** (BraTS 2023) and **75.45 %** (MSD-BTS).

Paired tests (student vs. teacher) show no significant degradation on TC and ET for BraTS 2023 and on all regions for MSD-BTS. On BraTS 2023 **WT** the teacher is significantly better (reported transparently in the paper). All numbers, tables and baselines are in the paper and in the result notebooks.

**Computational cost** (4×128³ input, single NVIDIA T4):

| Model | Params (M) | MACs (G) | Latency (ms) | Inference mem (MB) | Training mem (MB) |
|---|---|---|---|---|---|
| SAXNet teacher | 47.15 | 1027.24 | 1129.39 | 1965.5 | 9140.1 |
| LATUP student | 1.52 | 213.61 | 293.47 | 1368.7 | 3492.6 |

(BraTS 2023 evaluation run; the MSD-BTS run is reported separately in the paper.)

## What is new compared with Uni-Light

| | Uni-Light | UKD-Net |
|---|---|---|
| Teacher | Residual 3D U-Net (MONAI), 19.22 M | **SAXNet** (SAX blocks), 47.15 M |
| Student | LATUP-style, PCM, deep supervision | same compact student (1.52 M) |
| RCE term | used (λ = 0.1) | **disabled (λ = 0)**: accuracy improved without it |
| Boundary loss | SDF | SDF |
| Metrics | DSC, HD95 | DSC, HD95, **NSD**, Lesion Recall, Abs RVE, Vol MAE |
| Analysis | ablation on BraTS | ablations, **paired statistical tests** and complexity profiling on **both** datasets |

## Repository structure

```
UKD-Net/
├── models/
│   ├── saxnet.py             # SAXNet teacher (SAX blocks)
│   └── student.py            # LATUP student (PCM + deep supervision)
├── data/
│   └── dataset.py            # BraTS 2023 / MSD datasets, 5-fold split, SDF pre-computation
├── utils/
│   └── loss_metrics.py       # Dice+BCE, DWKD (+optional RCE), SDF boundary loss, Dice/HD95
├── train.py                  # teacher training, distillation, test
├── BraTS_evaluate.py         # standalone evaluation script
├── BraTS2023-Final Results.ipynb
├── MSD - Final Results.ipynb
├── Sample Results.ipynb      # qualitative visualisations
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/SorenSalehi/UKD-Net.git
cd UKD-Net
pip install -r requirements.txt
```

A CUDA GPU is required for training (all experiments were run on a single NVIDIA T4).

## Data

- **BraTS 2023 Adult Glioma (GLI)**: download from the [BraTS 2023 challenge](https://www.synapse.org/Synapse:syn51156910/wiki/622351) and place the training cases at `data/brats2023/Training/<case>/...` (files named `*-t2f`, `*-t1n`, `*-t1c`, `*-t2w`, `*-seg`).
- **MSD Task01 Brain Tumour**: download from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (`imagesTr/`, `labelsTr/`).

Preprocessing (done on the fly): per-modality z-score over non-zero voxels, centre pad/crop to 128×128×128, and for training random flips (depth/height), random 90° rotation in the H–W plane and Gaussian noise (σ = 0.05). Signed distance maps for the boundary loss are pre-computed once and cached.

**Split protocol.** All reported results use `KFold(n_splits=5, shuffle=True, random_state=42)` over the sorted case list, with **fold 4** as the held-out evaluation fold. This is implemented in `data/dataset.py` (`get_5fold_split`) and is shared by training and evaluation.

## Pretrained weights

Pretrained checkpoints (public Google Drive links) are provided to run the evaluation and to support follow-up research:

| Model | Dataset | File | Link |
|---|---|---|---|
| LATUP student | BraTS 2023 | `best_distilled_student_brats2023.pth` | [download](https://drive.google.com/file/d/1nyFaIoFngoCJ6vDSjVxjlbfiZrhnYWrr/view?usp=sharing) |
| LATUP student | MSD-BTS | `best_distilled_student_msd.pth` | [download](https://drive.google.com/file/d/1W6_7IX7IZlAF-KX9v--GBkxDFrkKOSGq/view?usp=sharing) |
| SAXNet teacher | BraTS 2023 | `best_saxnet_teacher_brats2023.pth` | [download](https://drive.google.com/file/d/1v6GiOGVj599D_Qte0GAiM4DHsJgxZ35J/view?usp=sharing) |
| SAXNet teacher | MSD-BTS | `best_saxnet_teacher_msd.pth` | [download](https://drive.google.com/file/d/1jc_bZLBWhNDAsrrghtioz-KForBPoVOJ/view?usp=sharing) |

Command-line download (needs `pip install gdown`):

```bash
mkdir -p weights && cd weights
gdown 1nyFaIoFngoCJ6vDSjVxjlbfiZrhnYWrr -O best_distilled_student_brats2023.pth
gdown 1W6_7IX7IZlAF-KX9v--GBkxDFrkKOSGq -O best_distilled_student_msd.pth
gdown 1v6GiOGVj599D_Qte0GAiM4DHsJgxZ35J -O best_saxnet_teacher_brats2023.pth
gdown 1jc_bZLBWhNDAsrrghtioz-KForBPoVOJ -O best_saxnet_teacher_msd.pth
```

### Quick inference

```python
import torch
from models.student import LATUPStudent
from models.saxnet import SAXNetV1

device = "cuda" if torch.cuda.is_available() else "cpu"

# Student (1.52 M params)
student = LATUPStudent(in_channels=4, out_channels=3, base_filters=16, deep_supervision=True).to(device)
ckpt = torch.load("weights/best_distilled_student_brats2023.pth", map_location=device, weights_only=False)
student.load_state_dict(ckpt["model_state_dict"])
student.eval()

# x: (B, 4, 128, 128, 128) = [FLAIR, T1, T1ce, T2], z-scored over non-zero voxels
with torch.no_grad():
    logits = student(x.to(device))            # (B, 3, 128, 128, 128)
    pred = (torch.sigmoid(logits) > 0.5)      # channels: WT, TC, ET (nested regions)
```

The teacher is loaded the same way with `SAXNetV1()` and the teacher checkpoints.

## Training

The teacher is trained first and then frozen; the student is trained from scratch under the joint objective.

```bash
# teacher + student + test (all stages)
python train.py --data-dir data/brats2023/Training --save-dir runs/ukd_brats --rce-weight 0.0

# individual stages
python train.py --stage teacher --data-dir data/brats2023/Training --save-dir runs/ukd_brats
python train.py --stage student --data-dir data/brats2023/Training --save-dir runs/ukd_brats \
                --teacher-checkpoint runs/ukd_brats/best_saxnet_teacher.pth --rce-weight 0.0
```

> **Important:** the final UKD-Net configuration disables the optional RCE term, so run the student with `--rce-weight 0.0`.

Configuration used for the reported results:

| | Teacher (SAXNet) | Student (LATUP) |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-5 | AdamW, weight decay 1e-5 |
| Learning rate | 5e-5 (cosine annealing to 1e-6) | 1e-4 (constant) |
| Epochs / batch size | 100 / 1 | 100 / 2 |
| Precision | mixed (AMP), grad-clip 1.0 | FP32 |
| Loss | Dice + BCE (channel weights [1, 1, 3]) | see below |

Student objective:

```
L = 0.6·L_task + 0.2·L_DWKD + 0.1·L_bound
    + 0.3·L_task(1/2 res) + 0.15·L_task(1/4 res)          (λ_RCE = 0)
```

with distillation temperature τ = 3.0 (no τ² rescaling). The best checkpoint is selected by validation Dice.

## Evaluation

Run `BraTS2023-Final Results.ipynb` (BraTS 2023) or `MSD - Final Results.ipynb` (MSD-BTS). They compute DSC, HD95, NSD (τ = 1 mm), Precision, Recall, Lesion Recall, Abs RVE, Vol MAE, paired Wilcoxon / t-tests between student and teacher, and complexity profiling. Set the data and weight paths at the top of each notebook (they default to Kaggle paths). `Sample Results.ipynb` produces qualitative overlays and 3D visualisations.

## Citation

If you use this code or the pretrained weights, please cite our earlier Uni-Light work and this repository:

```bibtex
@article{kuang2026unilight,
  title   = {Uni-Light: An Ultra-Lightweight Framework via Uncertainty-Aware Knowledge Distillation for Brain Tumour Segmentation},
  author  = {Kuang, Libing and Salehi, Soren and Wu, Ziling and Tafti, Ahmad P. and Moemeni, Armaghan},
  journal = {arXiv preprint arXiv:2609.06729},
  year    = {2026}
}

@misc{salehi2026ukdnet,
  title        = {UKD-Net: Teacher--Student Disagreement-Aware Knowledge Distillation from a Spatial-Channel Attention Network for Ultra-Lightweight 3D Brain Tumour Segmentation},
  author       = {Salehi, Soren},
  year         = {2026},
  howpublished = {\url{https://github.com/SorenSalehi/UKD-Net}}
}
```
work this project builds on.

## License

TODO: add a license for this repository (e.g. MIT or a research-only license). Please check the BraTS and MSD dataset terms of use before redistributing any data.
