import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
from pathlib import Path

# Add parent dir to path to import dataset
sys.path.append(str(Path(__file__).parent))
from dataset import NailDataset, generate_hard_negatives

def test_hard_negatives():
    print("Testing hard negative generation logic...")
    
    # Create dummy data: 448x448 image with a small "nail" in the corner
    img_t = torch.randn(3, 448, 448)
    msk_t = torch.zeros(1, 448, 448)
    msk_t[0, 10:30, 10:30] = 1.0 # Small nail in top-left
    
    # Test generation
    crops = generate_hard_negatives(img_t, msk_t, crop_size=224, n_crops=5)
    
    print(f"Generated {len(crops)} hard negative crops.")
    for i, (crop, mask) in enumerate(crops):
        print(f"  Crop {i}: img shape {crop.shape}, mask sum {mask.sum().item()}")
        assert mask.sum() == 0, f"Crop {i} should have an all-zero mask!"
        assert crop.shape == (3, 224, 224), f"Crop {i} should be 224x224!"
        
    print("Logic verification PASSED")

if __name__ == "__main__":
    test_hard_negatives()
