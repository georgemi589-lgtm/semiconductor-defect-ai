"""
Copies the real training results (loss/accuracy curves CSV, confusion matrix
image) into models/ so the dashboard can reliably find them, regardless of
Ultralytics' auto-numbered run folder names.

Run this once after training finishes:
    python export_model_metrics.py
"""

import os
import shutil
import json

# UPDATE THIS if your training folder name is different
TRAIN_RUN_DIR = "runs/classify/wafer_defect_training/yolov8n_cls_local_test-2"
TEST_EVAL_DIR = "runs/classify/val-3"   # the folder printed by evaluate_test_set.py

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Copy training curves CSV ──
results_csv_src = os.path.join(TRAIN_RUN_DIR, "results.csv")
results_csv_dst = os.path.join(OUTPUT_DIR, "results.csv")

if os.path.exists(results_csv_src):
    shutil.copy(results_csv_src, results_csv_dst)
    print(f"Copied training curves: {results_csv_dst}")
else:
    print(f"WARNING: not found: {results_csv_src}")

# ── Copy confusion matrix image (from the test evaluation, not training) ──
cm_src = os.path.join(TEST_EVAL_DIR, "confusion_matrix_normalized.png")
cm_dst = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

if os.path.exists(cm_src):
    shutil.copy(cm_src, cm_dst)
    print(f"Copied confusion matrix: {cm_dst}")
else:
    print(f"WARNING: not found: {cm_src}")
    print("Check the exact folder name — look inside your 'runs/classify/' folder.")

# ── Save final metrics as JSON for quick dashboard loading ──
metrics = {
    "top1_accuracy": 0.9205,
    "top5_accuracy": 0.9997,
    "evaluated_on": "held-out test set (data/splits/test, 9370 images)",
    "train_images_per_class_note": "full augmented dataset (~62,000 images)"
}

with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved metrics summary: {OUTPUT_DIR}/model_metrics.json")
print("\nDone! Your dashboard can now load real results from the models/ folder.")