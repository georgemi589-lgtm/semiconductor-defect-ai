"""
Final unbiased evaluation on the held-out test set.
Run this in your VS Code terminal with (venv) active:
    python evaluate_test_set.py

This uses data/splits/test — which was NEVER used during training or
validation — to get an honest measure of real-world performance.
"""

from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"
TEST_DIR = "data/splits/test"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH} — make sure you copied best.pt "
        f"into the models/ folder first."
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Could not find {TEST_DIR}")

print("Loading trained model...")
model = YOLO(MODEL_PATH)

print(f"Running evaluation on TRUE held-out test set: {TEST_DIR}\n")

# Ultralytics classification val() can point directly at a folder of
# class subfolders for evaluation, separate from the training data config.
metrics = model.val(
    data="data/splits",   # dataset root (needs train/ to exist alongside, but only test/ is used here)
    split='test'
)

print("\n===== FINAL TEST SET RESULTS (unbiased) =====")
print(f"Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.1f}%)")
print(f"Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.1f}%)")
print("\nConfusion matrix and per-class details saved to the runs/ folder shown above.")