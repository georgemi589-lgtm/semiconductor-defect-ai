**Status: actively being re-validated.** A data-leakage issue was found and is being corrected in the evaluation pipeline — see the Results section for full details. This project is shared to demonstrate the audit and correction process as much as the model itself. 
   [![Evaluation](https://img.shields.io/badge/test%20accuracy-verified%2091.56%25-brightgreen)](https://img.shields.io/badge/test%20accuracy-verified%2091.56%25-brightgreen)

# 🔬 DefectAI — Enterprise Semiconductor Defect Detection

An end-to-end AI system for detecting and classifying semiconductor wafer defects, built from raw data to a live, deployed product.

**🚀 [Live Demo](https://defectai-wafer-inspection.streamlit.app)** — try it yourself, no installation needed.

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)

---

## 📋 Overview

Semiconductor wafer inspection is a critical, expensive quality control step — a single defective wafer can cost $5,000-$50,000 in a fab. This project builds an AI-powered inspection system that automatically classifies wafer defect patterns, replacing slow, manual visual inspection.

Trained on **WM-811K**, a real-world dataset of 811,457 wafer maps (originally released by TSMC), the model classifies 9 defect categories:

| Defect Type | Description |
|---|---|
| `none` | No defect pattern |
| `Center` | Defects concentrated at wafer center |
| `Donut` | Ring-shaped defect in center area |
| `Edge-Loc` | Localized defects at wafer edge |
| `Edge-Ring` | Ring of defects around entire edge |
| `Loc` | Localized cluster of defects |
| `Near-full` | Almost entire wafer defective |
| `Random` | Random scattered defect pattern |
| `Scratch` | Linear scratch across wafer surface |

## 🎯 Results

**Data integrity note:** An internal audit found that the originally reported 92.05% top-1 / 99.97% top-5 accuracy was measured on a test set with confirmed data leakage — class-balancing augmentation (rotation, flip, noise) was applied *before* the train/val/test split, allowing augmented copies of the same source wafer to appear in both training and test sets. A dedicated leakage-check script confirmed **45.11% of test-set wafers had a training-time sibling**.

The pipeline was corrected (raw wafers are now split before any augmentation, and only the training split is balanced), verified to have **0% leakage**, and the model was retrained from scratch with matched hyperparameters (YOLOv8n-cls, imgsz=64, batch=32, 10 epochs).

**Verified result on the leakage-free test set:**

| | Original (leaked) | Verified (clean, 0% leakage) |
|---|---|---|
| Top-1 accuracy | 92.05% | **91.56%** |
| Top-5 accuracy | 99.97% | **100.00%** |

This is a confirmatory finding: despite real, measured leakage, the true accuracy on a genuinely held-out test set is nearly identical to the original figure. A per-class breakdown (in [`EVALUATION_AUDIT.md`](./EVALUATION_AUDIT.md)) further confirms the rarest, most heavily-augmented defect classes (Near-full: 100%, Donut: 97.6%) were not the source of any inflation — the weakest class (Loc, 79.1%) is a mid-frequency category, not a rare one, and its lower accuracy appears to reflect genuine visual similarity to related defect types rather than a leakage artifact.

Full methodology, the exact leakage measurement, the corrected pipeline code, and per-class results are documented in [`EVALUATION_AUDIT.md`](./EVALUATION_AUDIT.md).

- 91.56% Top-1 accuracy (verified, leakage-free test set, 5,334 images)
- 100.00% Top-5 accuracy (verified, leakage-free test set)
- Trained on a class-balanced dataset (training split only — validation and test sets retain their natural, imbalanced class distribution, as is correct practice)
