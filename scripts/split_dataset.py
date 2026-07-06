# scripts/split_dataset.py
# Purpose: Split the balanced processed images into 
# train/val/test sets, organized per class.
#
# Why per-class splitting matters:
# If we split randomly across the whole dataset, a rare class
# could end up entirely in 'train' with zero samples in 'val' or 'test'.
# Splitting within each class folder guarantees every class is
# represented proportionally in all three sets.

import shutil
from pathlib import Path
import random
from collections import defaultdict


def split_dataset(
    source_dir: str = "data/processed",
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split processed images into train/val/test folders, 
    maintaining class balance in each split.
    """
    random.seed(seed)  # Same seed = same split every time (reproducible)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Get all class folders (Center, Scratch, Donut, etc.)
    class_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    print("=" * 60)
    print("DATASET SPLITTING — Train / Validation / Test")
    print("=" * 60)
    print(f"Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")
    
    summary = defaultdict(dict)
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Get all images in this class
        images = list(class_folder.glob("*.png"))
        random.shuffle(images)  # Randomize order before splitting
        
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        # Test gets whatever remains (avoids rounding errors losing images)
        test_count = total - train_count - val_count
        
        # Slice the shuffled list into three parts
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # Create destination folders and copy files
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            dest_dir = output_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)
        
        summary[class_name] = {
            'total': total,
            'train': train_count,
            'val': val_count,
            'test': test_count
        }
        
        print(f"\n{class_name}:")
        print(f"  Total: {total:,} | Train: {train_count:,} | "
              f"Val: {val_count:,} | Test: {test_count:,}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ SPLIT COMPLETE!")
    print("=" * 60)
    
    total_train = sum(s['train'] for s in summary.values())
    total_val = sum(s['val'] for s in summary.values())
    total_test = sum(s['test'] for s in summary.values())
    total_all = sum(s['total'] for s in summary.values())
    
    print(f"\nTrain: {total_train:,} images")
    print(f"Val:   {total_val:,} images")
    print(f"Test:  {total_test:,} images")
    print(f"Total: {total_all:,} images")
    
    return summary


if __name__ == "__main__":
    split_dataset()
    