# augmentor.py
# Purpose: Fix the 5,275x class imbalance by creating synthetic
# variations of rare defect types (Scratch, Donut, Near-full, etc.)
#
# Why this matters:
# We have 785,938 "none" samples but only 149 "Near-full" samples.
# Without augmentation, the AI model would barely ever see rare 
# defects during training and would fail to detect them in production.

import numpy as np
import cv2
from typing import List


class WaferAugmentor:
    """
    Creates variations of wafer map images to balance defect classes.
    
    Each transformation preserves the defect pattern's meaning:
    - A scratch rotated 90° is still recognizably a scratch
    - A center defect flipped horizontally is still a center defect
    """
    
    def __init__(self):
        pass
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Mirror image left-right. A defect pattern flipped is still valid."""
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Mirror image top-bottom."""
        return cv2.flip(image, 0)
    
    def rotate_90(self, image: np.ndarray) -> np.ndarray:
        """Rotate 90 degrees clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def rotate_180(self, image: np.ndarray) -> np.ndarray:
        """Rotate 180 degrees."""
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def rotate_270(self, image: np.ndarray) -> np.ndarray:
        """Rotate 270 degrees clockwise (= 90 counter-clockwise)."""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def rotate_arbitrary(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate by any angle (e.g. 15, 30, 45 degrees).
        
        Used for creating more variety than just 90/180/270 rotations.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # Black fill for empty corners
        )
        return rotated
    
    def add_noise(self, image: np.ndarray, intensity: float = 0.02) -> np.ndarray:
        """
        Add slight random noise to simulate sensor variation.
        
        Real factory cameras have minor noise. Adding a little 
        helps the model generalize better to real-world conditions.
        """
        noise = np.random.normal(0, intensity * 255, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def generate_augmented_set(self, image: np.ndarray, count: int = 5) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of ONE wafer image.
        
        Args:
            image: Original processed wafer image (640x640x3)
            count: How many augmented copies to generate
            
        Returns:
            List of augmented images (does NOT include original)
        """
        augmentations = [
            self.flip_horizontal,
            self.flip_vertical,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            lambda img: self.rotate_arbitrary(img, 15),
            lambda img: self.rotate_arbitrary(img, 30),
            lambda img: self.rotate_arbitrary(img, 45),
            lambda img: self.add_noise(self.flip_horizontal(img)),
            lambda img: self.add_noise(self.rotate_90(img)),
        ]
        
        results = []
        for i in range(count):
            # Cycle through augmentation types
            aug_func = augmentations[i % len(augmentations)]
            augmented = aug_func(image)
            results.append(augmented)
        
        return results


# Quick test
if __name__ == "__main__":
    print("Testing WaferAugmentor...")
    
    # Fake test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    augmentor = WaferAugmentor()
    augmented_images = augmentor.generate_augmented_set(test_image, count=5)
    
    print(f"✅ Generated {len(augmented_images)} augmented versions")
    print(f"✅ Each shape: {augmented_images[0].shape}")
    print(f"✅ Augmentor working correctly!")