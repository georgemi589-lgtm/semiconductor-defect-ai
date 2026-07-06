"""
Local training script — trains directly on your computer, no Colab needed.
Run this in your VS Code terminal with (venv) active:
    python train_local.py
"""

from ultralytics import YOLO
import os

# This should point to the small sample dataset folder created earlier
DATASET_ROOT = "data/splits"

# Quick sanity check before training starts
for split in ['train', 'val']:
    split_path = os.path.join(DATASET_ROOT, split)
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Could not find {split_path} — make sure you ran "
            f"create_sample_dataset.py first and this script is in your "
            f"project root folder."
        )
    classes = sorted(os.listdir(split_path))
    print(f"{split}: {len(classes)} classes -> {classes}")

print("\nStarting training on CPU (this may take a while, be patient)...\n")

model = YOLO('yolov8n-cls.pt')   # downloads automatically on first run

results = model.train(
    data=DATASET_ROOT,
    epochs=10,
    imgsz=64,
    batch=32,
    patience=5,
    project='wafer_defect_training',
    name='yolov8n_cls_local_test',
    device='cpu'
)

print("\nTraining complete! Evaluating on validation set...\n")
metrics = model.val()
print(metrics)

best_weights = "wafer_defect_training/yolov8n_cls_local_test/weights/best.pt"
print(f"\nBest model weights saved locally at: {best_weights}")
print("You can now copy this into your project's models/ folder.")