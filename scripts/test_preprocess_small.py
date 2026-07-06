# scripts/test_preprocess_small.py
# Purpose: Test the full pipeline on just a TINY sample first
# Only processes 5 images per class instead of thousands
# This lets us verify everything works before the full 30-60 min run

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.preprocessor import WaferPreprocessor
from src.data.augmentor import WaferAugmentor

import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes


def main():
    print("=" * 60)
    print("SMALL TEST RUN — 5 images per class only")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    data_path = Path("data/raw/LSWMD.pkl")
    with open(data_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    
    df['failureType_clean'] = df['failureType'].apply(
        lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) 
        and len(x) > 0 and len(x[0]) > 0 else 'none'
    )
    print(f"✅ Loaded {len(df):,} wafer maps")
    
    # Setup
    preprocessor = WaferPreprocessor(target_size=640)
    augmentor = WaferAugmentor()
    
    test_output_dir = Path("data/processed/test_run")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    defect_classes = df['failureType_clean'].value_counts().index.tolist()
    
    total_saved = 0
    
    for defect_class in defect_classes:
        class_df = df[df['failureType_clean'] == defect_class].head(5)
        safe_name = defect_class.replace(" ", "_")
        
        print(f"\nProcessing '{defect_class}' (5 samples)...")
        
        for idx, row in tqdm(class_df.iterrows(), total=len(class_df)):
            wafer_map = row['waferMap']
            
            # Process original
            processed_image = preprocessor.process_single(wafer_map)
            save_path = test_output_dir / f"{safe_name}_{idx}_orig.png"
            preprocessor.save_image(processed_image, str(save_path))
            total_saved += 1
            
            # Generate 2 augmented copies for testing
            augmented_images = augmentor.generate_augmented_set(
                processed_image, count=2
            )
            for aug_idx, aug_image in enumerate(augmented_images):
                aug_path = test_output_dir / f"{safe_name}_{idx}_aug{aug_idx}.png"
                preprocessor.save_image(aug_image, str(aug_path))
                total_saved += 1
    
    print("\n" + "=" * 60)
    print(f"✅ TEST COMPLETE! {total_saved} images saved")
    print(f"📁 Check folder: data/processed/test_run/")
    print("=" * 60)


if __name__ == "__main__":
    main()