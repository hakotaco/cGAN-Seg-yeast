#!/usr/bin/env python3
"""
Numerical test of the improved dataset without any visualization.
"""

import sys
import os
sys.path.append('..')
from data import BasicDataset, get_image_mask_pairs
import torch
import numpy as np
from PIL import Image
from transforms import Compose, ToPILImage, Resize, ToTensor, Normalize

def test_dataset():
    print("Testing improved dataset...")
    
    # Test the dataset
    train_dir = "yeaz_dataset_fixed/train"
    
    if not os.path.exists(train_dir):
        print(f"Directory {train_dir} not found!")
        return
    
    # Get image-mask pairs
    pairs = get_image_mask_pairs(train_dir)
    print(f"Found {len(pairs)} image-mask pairs")
    
    if len(pairs) == 0:
        print("No pairs found!")
        return
    
    # Test first few pairs directly from files
    print("\n=== Testing raw files ===")
    for i, (img_path, mask_path) in enumerate(pairs[:3]):
        print(f"\nPair {i}:")
        print(f"  Image: {os.path.basename(img_path)}")
        print(f"  Mask: {os.path.basename(mask_path)}")
        
        # Load and check raw files
        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            img_arr = np.array(img)
            mask_arr = np.array(mask)
            
            print(f"  Raw image shape: {img_arr.shape}, dtype: {img_arr.dtype}")
            print(f"  Raw image range: [{img_arr.min()}, {img_arr.max()}]")
            print(f"  Raw mask shape: {mask_arr.shape}, dtype: {mask_arr.dtype}")
            print(f"  Raw mask range: [{mask_arr.min()}, {mask_arr.max()}]")
            print(f"  Raw mask unique values: {np.unique(mask_arr)[:10]}...")  # Show first 10
            print(f"  Raw mask non-zero ratio: {(mask_arr > 0).mean():.4f}")
            
        except Exception as e:
            print(f"  Error loading files: {e}")
    
    # Create dataset and test processed samples
    print("\n=== Testing processed dataset ===")
    try:
        # Create simple transforms for testing
        test_transforms = Compose([
            ToPILImage(),
            Resize([512, 768]),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])
        
        dataset = BasicDataset(pairs, transforms=test_transforms, gen_nc=1)
        print(f"Dataset size: {len(dataset)}")
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            image = sample['image']
            mask = sample['mask']
            print(f"\nProcessed sample {i}:")
            print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"  Mask unique values: {torch.unique(mask).numpy()}")
            print(f"  Non-zero mask ratio: {(mask > 0).float().mean():.4f}")
        
        print("\nDataset test completed successfully!")
        
    except Exception as e:
        print(f"Error creating/testing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
