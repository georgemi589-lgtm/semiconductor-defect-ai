"""
Create a small sample of the wafer defect dataset for fast upload testing.
Run this LOCALLY on your computer (in VS Code / your venv), not in Colab.

It copies a limited number of images per class from data/splits/train and
data/splits/val into a new data/splits_sample folder, then zips it.
"""

import os
import shutil
import random
import zipfile

# ── SETTINGS — adjust these paths if your project folder is different ──
SOURCE_ROOT = "data/splits"          # your existing full dataset
SAMPLE_ROOT = "data/splits_sample"    # small sample will be created here
IMAGES_PER_CLASS_TRAIN = 100          # how many training images per class to copy
IMAGES_PER_CLASS_VAL = 20             # how many validation images per class to copy
ZIP_OUTPUT = "splits_sample.zip"

random.seed(42)


def copy_sample(split_name, num_per_class):
    src_split = os.path.join(SOURCE_ROOT, split_name)
    dst_split = os.path.join(SAMPLE_ROOT, split_name)

    if not os.path.exists(src_split):
        print(f"WARNING: {src_split} not found, skipping.")
        return

    classes = os.listdir(src_split)
    for class_name in classes:
        src_class_dir = os.path.join(src_split, class_name)
        dst_class_dir = os.path.join(dst_split, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        all_files = os.listdir(src_class_dir)
        sample_files = random.sample(all_files, min(num_per_class, len(all_files)))

        for fname in sample_files:
            shutil.copy(
                os.path.join(src_class_dir, fname),
                os.path.join(dst_class_dir, fname)
            )

        print(f"  {split_name}/{class_name}: copied {len(sample_files)} images")


def zip_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)


if __name__ == "__main__":
    print("Creating sample dataset...")

    if os.path.exists(SAMPLE_ROOT):
        shutil.rmtree(SAMPLE_ROOT)

    print("\nCopying training images...")
    copy_sample("train", IMAGES_PER_CLASS_TRAIN)

    print("\nCopying validation images...")
    copy_sample("val", IMAGES_PER_CLASS_VAL)

    print(f"\nZipping to {ZIP_OUTPUT}...")
    zip_folder(SAMPLE_ROOT, ZIP_OUTPUT)

    size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
    print(f"\nDone! Created {ZIP_OUTPUT} ({size_mb:.1f} MB)")
    print("This small file should upload to Colab in seconds instead of hours.")