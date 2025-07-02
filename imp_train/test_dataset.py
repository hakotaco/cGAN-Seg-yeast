#!/usr/bin/env python3
"""
Quick validation script to test the improved dataset and training setup.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append('..')
sys.path.append('/home2/s5946301/cGAN-Seg-yeast')
from data import BasicDataset, get_image_mask_pairs

def test_dataset(data_dir, num_samples=3):
    """Test the dataset and visualize some samples"""
    
    print(f"Testing dataset from: {data_dir}")
    
    # Get sample pairs
    sample_pairs = get_image_mask_pairs(data_dir)
    print(f"Found {len(sample_pairs)} sample pairs")
    
    if len(sample_pairs) == 0:
        print("ERROR: No sample pairs found!")
        return False
    
    # Test transforms
    transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    dataset = BasicDataset(sample_pairs, transforms=transforms_test, gen_nc=1)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(num_samples, len(dataset))):
        try:
            image, mask = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"  Mask non-zero ratio: {(mask > 0).float().mean():.4f}")
            
            # Check if mask has meaningful content
            if (mask > 0).float().mean() < 0.01:
                print(f"  WARNING: Mask {i+1} appears mostly empty!")
            else:
                print(f"  OK: Mask {i+1} has good content")
                
        except Exception as e:
            print(f"  ERROR loading sample {i+1}: {e}")
            return False
    
    return True

def visualize_samples(data_dir, output_dir, num_samples=5):
    """Visualize some samples and save to output directory"""
    
    sample_pairs = get_image_mask_pairs(data_dir)
    
    if len(sample_pairs) == 0:
        print("No samples to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(sample_pairs))):
        img_path, mask_path = sample_pairs[i]
        
        # Load image and mask
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Image {i+1}')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask {i+1}')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray', alpha=0.7)
        axes[2].imshow(mask, cmap='Reds', alpha=0.3)
        axes[2].set_title(f'Overlay {i+1}')
        axes[2].axis('off')
        
        # Add statistics
        mask_ratio = np.sum(mask > 0) / mask.size
        fig.suptitle(f'Sample {i+1} - Mask coverage: {mask_ratio:.1%}')
        
        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    print("=== Testing Improved Dataset ===\n")
    
    # Test the fixed dataset
    data_dir = "yeaz_dataset_fixed/train"
    
    if os.path.exists(data_dir):
        print("Testing fixed dataset...")
        success = test_dataset(data_dir)
        
        if success:
            print("\n✓ Dataset test PASSED!")
            
            # Create visualizations
            print("\nCreating sample visualizations...")
            visualize_samples(data_dir, "sample_visualizations", num_samples=5)
            
        else:
            print("\n✗ Dataset test FAILED!")
            return 1
            
    else:
        print(f"Fixed dataset not found at {data_dir}")
        return 1
    
    # Compare with original broken dataset
    original_data_dir = "../yeaz_dataset/train"
    if os.path.exists(original_data_dir):
        print(f"\n=== Comparing with original dataset ===")
        print("Testing original (broken) dataset...")
        test_dataset(original_data_dir, num_samples=2)
    
    return 0

if __name__ == "__main__":
    exit(main())
