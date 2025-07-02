#!/usr/bin/env python3
"""
Improved data preprocessing for yeast cell segmentation.
This script properly converts multi-label instance masks to binary masks
and organizes the data for training.
"""

import os
import glob
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import argparse
import random

def convert_instance_to_binary_mask(instance_mask):
    """
    Convert instance segmentation mask to binary mask.
    
    Args:
        instance_mask: numpy array with instance labels (0=background, 1,2,3...=different cells)
    
    Returns:
        binary_mask: numpy array with binary labels (0=background, 255=cell)
    """
    binary_mask = (instance_mask > 0).astype(np.uint8) * 255
    return binary_mask

def validate_mask_pair(image_path, mask_path):
    """
    Validate that image and mask pair are properly formatted.
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        image = Image.open(image_path)
        mask = np.array(Image.open(mask_path))
        
        # Check if mask has any foreground pixels
        if np.sum(mask > 0) == 0:
            print(f"Warning: Empty mask found: {mask_path}")
            return False
            
        # Check if image and mask dimensions match
        if image.size != (mask.shape[1], mask.shape[0]):
            print(f"Warning: Size mismatch - Image: {image.size}, Mask: {mask.shape}")
            return False
            
        return True
    except Exception as e:
        print(f"Error validating {image_path}, {mask_path}: {e}")
        return False

def process_yeaz_data(input_dir, output_dir, test_size=0.2, random_state=42):
    """
    Process yeaz data and create train/test splits with proper binary masks.
    
    Args:
        input_dir: Directory containing *_im.tif and *_mask.tif files
        output_dir: Output directory for processed data
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducible splits
    """
    
    # Find all image and mask pairs
    image_files = sorted(glob.glob(os.path.join(input_dir, "*_im.tif")))
    mask_files = sorted(glob.glob(os.path.join(input_dir, "*_mask.tif")))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Create pairs and validate
    valid_pairs = []
    for img_file in tqdm(image_files, desc="Validating data"):
        # Find corresponding mask file
        base_name = os.path.basename(img_file).replace("_im.tif", "")
        mask_file = os.path.join(input_dir, f"{base_name}_mask.tif")
        
        if os.path.exists(mask_file):
            if validate_mask_pair(img_file, mask_file):
                valid_pairs.append((img_file, mask_file, base_name))
        else:
            print(f"Warning: No mask found for {img_file}")
    
    print(f"Found {len(valid_pairs)} valid image-mask pairs")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-mask pairs found!")
    
    # Split into train and test
    random.seed(random_state)
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * (1 - test_size))
    train_pairs = valid_pairs[:split_idx]
    test_pairs = valid_pairs[split_idx:]
    
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")
    
    # Create output directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    
    # Process and save train data
    print("Processing training data...")
    for img_file, mask_file, base_name in tqdm(train_pairs):
        # Copy image
        shutil.copy2(img_file, os.path.join(output_dir, 'train', 'images', f"{base_name}.tif"))
        
        # Process and save mask
        instance_mask = np.array(Image.open(mask_file))
        binary_mask = convert_instance_to_binary_mask(instance_mask)
        Image.fromarray(binary_mask).save(os.path.join(output_dir, 'train', 'masks', f"{base_name}.tif"))
    
    # Process and save test data
    print("Processing test data...")
    for img_file, mask_file, base_name in tqdm(test_pairs):
        # Copy image
        shutil.copy2(img_file, os.path.join(output_dir, 'test', 'images', f"{base_name}.tif"))
        
        # Process and save mask
        instance_mask = np.array(Image.open(mask_file))
        binary_mask = convert_instance_to_binary_mask(instance_mask)
        Image.fromarray(binary_mask).save(os.path.join(output_dir, 'test', 'masks', f"{base_name}.tif"))
    
    # Print statistics
    print("\n=== Data Processing Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Test samples: {len(test_pairs)}")
    
    # Validate a few processed masks
    print("\nValidating processed masks...")
    train_masks = glob.glob(os.path.join(output_dir, 'train', 'masks', '*.tif'))[:3]
    for mask_file in train_masks:
        mask = np.array(Image.open(mask_file))
        nonzero_ratio = np.sum(mask > 0) / mask.size
        print(f"  {os.path.basename(mask_file)}: non-zero ratio = {nonzero_ratio:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Process yeaz data for training")
    parser.add_argument("--input_dir", required=True, help="Directory containing yeaz data")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    process_yeaz_data(args.input_dir, args.output_dir, args.test_size, args.random_state)

if __name__ == "__main__":
    main()
