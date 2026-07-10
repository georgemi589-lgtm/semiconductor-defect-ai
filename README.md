# 🔬 DefectAI — Enterprise Semiconductor Defect Detection

An end-to-end AI system for detecting and classifying semiconductor wafer defects, built from raw data to a live, deployed product.

**🚀 [Live Demo](https://defectai-wafer-inspection.streamlit.app)** — try it yourself, no installation needed.

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Model Accuracy](https://img.shields.io/badge/test%20accuracy-92%25-success)

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

- **92.05% Top-1 accuracy** on a completely held-out test set (9,370 images, never seen during training or validation)
- **99.97% Top-5 accuracy**
- Trained on a class-balanced dataset (fixed an original **5,274:1 class imbalance** between the majority and minority defect classes)

## 🏗️ System Architecture

```
Raw Data (WM-811K) → Preprocessing & Augmentation → YOLOv8n-cls Training
                                                            ↓
                                                     Trained Model
                                                            ↓
                        ┌───────────────────────────────────┴───────────────────────────────────┐
                        ↓                                                                         ↓
                 FastAPI Backend                                                      Streamlit Dashboard
              (REST API for predictions)                                    (Live inspection, batch processing,
                        ↓                                                     analytics, reports, history)
                 SQLite Database                                                              ↓
              (Inspection records)                                                    PDF Report Generator
```

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, YOLOv8 (Ultralytics), Transfer Learning
- **Computer Vision:** OpenCV, Albumentations
- **Backend:** FastAPI, SQLAlchemy, SQLite
- **Frontend:** Streamlit, Plotly
- **Reporting:** ReportLab (PDF generation)
- **Deployment:** Streamlit Community Cloud

## ✨ Features

- 🔍 **Single Inspection** — upload one wafer image, get an instant classification with confidence score
- 📦 **Batch Processing** — inspect multiple wafers at once, export results as CSV
- 📊 **Dataset Analytics** — explore the original WM-811K class distribution and the balancing process
- 📈 **Model Performance** — real training curves, confusion matrix, and test-set metrics (not simulated)
- 📄 **PDF Reports** — generate professional inspection reports for quality records
- 📋 **Inspection History** — every inspection logged to a database, viewable and exportable

## 📂 Project Structure

```
semiconductor-defect-ai/
├── api/                  # FastAPI backend
├── dashboard/            # Streamlit dashboard (app.py)
├── database/             # SQLAlchemy models + CRUD operations
├── models/               # Trained model weights + evaluation artifacts
├── src/
│   ├── data/             # Preprocessing & augmentation
│   ├── reporting/         # PDF report generator
│   └── ...
├── scripts/              # Standalone training/evaluation scripts
└── requirements.txt
```

## 🚀 Running Locally

```bash
git clone https://github.com/georgemi589-lgtm/semiconductor-defect-ai.git
cd semiconductor-defect-ai
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 📈 Project Roadmap

- [x] Week 1 — Data Exploration
- [x] Week 2 — Preprocessing & Augmentation
- [x] Week 3 — YOLOv8 Model Training
- [x] Week 4 — FastAPI Backend
- [x] Week 5 — Dashboard, Database & Reports
- [x] Week 5.5 — PCB Defect Detection (2nd inspection mode)
- [ ] Week 6 — Docker & Deployment
`
## 🔌 PCB Defect Detection (New)

In addition to wafer-level defect classification, DefectAI now includes
a second inspection mode for **PCB (printed circuit board) defect detection**
using object detection — locating and classifying multiple defects per image,
not just a single label per image.

- **Model:** YOLOv8n (object detection), trained on CPU
- **Dataset:** 693 annotated PCB images, 6 defect classes
- **Classes:** `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`
- **Results:** 89.4% mAP50, 46.7% mAP50-95 on held-out validation data
- **Inference speed:** ~0.1s per board
- Fully integrated into the same dashboard — single inspection, batch
  processing, database logging, and PDF reports, mirroring the wafer
  classifier's workflow.


## 📅 Progress Log

| Week | Date | Milestone |
|------|------|-----------|
| 1 | [fill in] | Explored WM-811K dataset (811K wafer maps), analyzed class imbalance |
| 2 | [fill in] | Built preprocessing + augmentation pipeline, fixed 5,274:1 class imbalance |
| 3 | [fill in] | Trained YOLOv8n-cls wafer classifier — 92.05% test accuracy |
| 4 | [fill in] | Built FastAPI backend serving real-time predictions |
| 5 | [fill in] | Built Streamlit dashboard, SQLite logging, PDF report generation; deployed live to Streamlit Community Cloud |
| 5.5 | 2026-07-10 | Added PCB defect detection (YOLOv8n object detection) — 693 images, 6 defect classes, 89.4% mAP50 — fully integrated as a second inspection mode |
| 6 | [upcoming] | Docker containerization for production deployment |

*(Fill in the actual dates for Weeks 1-5 from your own notes/commit history — even approximate dates are fine.)*
## 📜 License

This project uses the WM-811K dataset for research/educational purposes.

---

Built as part of the MIPHI Program | CUBE AI Solutions
