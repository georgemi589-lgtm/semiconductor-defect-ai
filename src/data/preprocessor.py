# preprocessor.py
# Purpose: Convert raw wafer map arrays into proper images for YOLO training
#
# Why this matters:
# Our wafer maps are stored as small arrays with values 0,1,2
# YOLO needs actual RGB images of consistent size
# This file bridges that gap

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
import sys
import os

# Allow importing from src folder
sys.path.append(str(Path(__file__).parent.parent.parent))


class WaferPreprocessor:
    """
    Converts raw wafer map arrays (0,1,2 values) into 
    standardized RGB images ready for YOLO training.
    """
    
    def __init__(self, target_size: int = 640):
        """
        Args:
            target_size: Final image size (YOLO needs square images,
                        640 is standard for YOLOv8)
        """
        self.target_size = target_size
        
        # Color mapping for wafer pixel values
        # 0 = outside wafer = black
        # 1 = good chip = green  
        # 2 = defective chip = red
        self.color_map = {
            0: [0, 0, 0],        # Black (outside wafer)
            1: [46, 204, 113],   # Green (good chip)
            2: [231, 76, 60]     # Red (defective chip)
        }
    
    def array_to_image(self, wafer_map: np.ndarray) -> np.ndarray:
        """
        Convert raw wafer array (values 0,1,2) into RGB image.
        
        Args:
            wafer_map: 2D numpy array with values 0, 1, or 2
            
        Returns:
            RGB image as numpy array, shape (H, W, 3)
        """
        h, w = wafer_map.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each pixel value to its corresponding color
        for value, color in self.color_map.items():
            mask = (wafer_map == value)
            rgb_image[mask] = color
        
        return rgb_image
    
    def resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target_size x target_size.
        
        Uses INTER_NEAREST interpolation because our images 
        are categorical (3 distinct colors), not photographic.
        INTER_NEAREST preserves sharp edges between colors,
        unlike INTER_LINEAR which would blend colors at boundaries.
        """
        resized = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST
        )
        return resized
    
    def add_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Make the image square before resizing, to avoid distortion.
        
        Why? Wafer maps aren't always perfectly square (e.g. 45x48).
        If we resize directly to 640x640, the wafer shape gets stretched.
        Padding first preserves the wafer's true circular shape.
        """
        h, w = image.shape[:2]
        
        if h == w:
            return image
        
        # Find the larger dimension
        max_dim = max(h, w)
        
        # Create black square canvas
        padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # Center the original image on the canvas
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        return padded
    
    def process_single(self, wafer_map: np.ndarray) -> np.ndarray:
        """
        Full pipeline: array → RGB → padded → resized
        
        This is the main method you'll call for each wafer map.
        """
        rgb_image = self.array_to_image(wafer_map)
        padded_image = self.add_padding(rgb_image)
        final_image = self.resize_to_target(padded_image)
        
        return final_image
    
    def save_image(self, image: np.ndarray, save_path: str) -> None:
        """
        Save processed image to disk as PNG.
        
        Note: OpenCV expects BGR order when saving,
        but we built our image in RGB order, so we convert.
        """
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_image)


# Quick test function - run this file directly to test
if __name__ == "__main__":
    print("Testing WaferPreprocessor...")
    
    # Create a fake test wafer map (10x10) for testing
    test_wafer = np.random.choice([0, 1, 2], size=(45, 48))
    
    preprocessor = WaferPreprocessor(target_size=640)
    result = preprocessor.process_single(test_wafer)
    
    print(f"✅ Input shape: {test_wafer.shape}")
    print(f"✅ Output shape: {result.shape}")
    print(f"✅ Output dtype: {result.dtype}")
    print(f"✅ Preprocessor working correctly!")