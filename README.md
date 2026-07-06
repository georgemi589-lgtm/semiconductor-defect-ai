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
- [ ] Week 6 — Docker & Deployment

## 📜 License

This project uses the WM-811K dataset for research/educational purposes.

---

Built as part of the MIPHI Program | CUBE AI Solutions
