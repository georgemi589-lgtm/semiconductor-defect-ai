# scripts/preprocess_dataset.py
# Purpose: Master script that processes the ENTIRE WM-811K dataset
# - Converts every wafer map into a proper image
# - Applies augmentation to balance rare classes
# - Saves everything to data/processed/ organized by class
#
# Run this once. It creates your full training-ready dataset.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from src.data.preprocessor import WaferPreprocessor
from src.data.augmentor import WaferAugmentor

# Fix for old pandas pickle compatibility
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes


def load_dataset():
    """Load the raw WM-811K dataset."""
    print("Loading WM-811K dataset...")
    data_path = Path("data/raw/LSWMD.pkl")
    
    with open(data_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    
    df['failureType_clean'] = df['failureType'].apply(
        lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) 
        and len(x) > 0 and len(x[0]) > 0 else 'none'
    )
    
    print(f"✅ Loaded {len(df):,} wafer maps")
    return df


def create_output_folders(defect_classes):
    """Create one folder per defect class inside data/processed/"""
    base_path = Path("data/processed")
    
    for defect_class in defect_classes:
        # Replace special characters for valid folder names
        safe_name = defect_class.replace(" ", "_")
        folder_path = base_path / safe_name
        folder_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created {len(defect_classes)} class folders in data/processed/")


def process_class(df, defect_class, target_count, preprocessor, augmentor):
    """
    Process all wafer maps of ONE defect class.
    Saves original processed images + augmented copies if needed.
    """
    class_df = df[df['failureType_clean'] == defect_class]
    current_count = len(class_df)
    
    safe_name = defect_class.replace(" ", "_")
    output_dir = Path("data/processed") / safe_name
    
    saved_count = 0
    
    # Determine how many augmentations per image we need
    if current_count >= target_count or defect_class == 'none':
        augs_per_image = 0
    else:
        needed = target_count - current_count
        augs_per_image = int(np.ceil(needed / current_count))
    
    print(f"\nProcessing '{defect_class}': {current_count:,} samples, "
          f"{augs_per_image} augmentations per image")
    
    for idx, row in tqdm(class_df.iterrows(), total=len(class_df), 
                          desc=f"{defect_class}"):
        wafer_map = row['waferMap']
        
        # Process original image
        processed_image = preprocessor.process_single(wafer_map)
        
        # Save original
        save_path = output_dir / f"{safe_name}_{idx}_orig.png"
        preprocessor.save_image(processed_image, str(save_path))
        saved_count += 1
        
        # Generate augmented copies if this class needs more samples
        if augs_per_image > 0:
            augmented_images = augmentor.generate_augmented_set(
                processed_image, count=augs_per_image
            )
            
            for aug_idx, aug_image in enumerate(augmented_images):
                aug_save_path = output_dir / f"{safe_name}_{idx}_aug{aug_idx}.png"
                preprocessor.save_image(aug_image, str(aug_save_path))
                saved_count += 1
    
    return saved_count


def main():
    print("=" * 60)
    print("WAFER DEFECT DATASET PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_dataset()
    
    class_counts = df['failureType_clean'].value_counts()
    defect_classes = class_counts.index.tolist()
    
    # Setup
    create_output_folders(defect_classes)
    preprocessor = WaferPreprocessor(target_size=640)
    augmentor = WaferAugmentor()
    
    TARGET_SAMPLES_PER_CLASS = 5000
    
    # IMPORTANT: For 'none' class with 785,938 samples, 
    # processing all of them would take too long and isn't necessary.
    # We'll cap it at a reasonable training size.
    NONE_CLASS_CAP = 10000
    
    total_saved = 0
    summary = {}
    
    for defect_class in defect_classes:
        if defect_class == 'none':
            # Sample a subset instead of using all 785,938
            class_df_full = df[df['failureType_clean'] == 'none']
            class_df_sample = class_df_full.sample(
                n=min(NONE_CLASS_CAP, len(class_df_full)), 
                random_state=42
            )
            # Temporarily replace df subset for this class
            saved = process_class(
                class_df_sample.reset_index(drop=True), 
                defect_class, NONE_CLASS_CAP, 
                preprocessor, augmentor
            )
        else:
            saved = process_class(
                df, defect_class, TARGET_SAMPLES_PER_CLASS,
                preprocessor, augmentor
            )
        
        summary[defect_class] = saved
        total_saved += saved
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    for defect_class, count in summary.items():
        print(f"  {defect_class:<15} : {count:,} images saved")
    print(f"\n  TOTAL: {total_saved:,} images saved to data/processed/")


if __name__ == "__main__":
    main()