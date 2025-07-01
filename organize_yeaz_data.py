#!/usr/bin/env python3
"""
Script to organize yeaz yeast cell data into proper train/test structure
while keeping original data intact.

The script:
1. Parses filenames in format: <tag>_<num1>_crop_<num2>_<type>.tif
2. Groups image-mask pairs by their base name
3. Splits data 90-10 into train/test sets
4. Creates symlinks to preserve original data
"""

import os
import re
import random
from pathlib import Path
import shutil

def parse_filename(filename):
    """
    Parse filename format: <tag>_<num1>_crop_<num2>_<type>.tif
    Returns: (base_name, file_type) where base_name excludes the type and extension
    """
    # Remove .tif extension
    name_without_ext = filename.replace('.tif', '')
    
    # Split by underscore and check if it ends with 'im' or 'mask'
    if name_without_ext.endswith('_im'):
        base_name = name_without_ext[:-3]  # Remove '_im'
        file_type = 'im'
    elif name_without_ext.endswith('_mask'):
        base_name = name_without_ext[:-5]  # Remove '_mask'
        file_type = 'mask'
    else:
        return None, None
    
    return base_name, file_type

def collect_image_mask_pairs(source_dir):
    """
    Collect all image-mask pairs from the source directory
    Returns: dict of {base_name: {'im': path, 'mask': path}}
    """
    pairs = {}
    
    for filename in os.listdir(source_dir):
        if not filename.endswith('.tif'):
            continue
            
        base_name, file_type = parse_filename(filename)
        if base_name is None:
            print(f"Warning: Could not parse filename: {filename}")
            continue
            
        if base_name not in pairs:
            pairs[base_name] = {}
            
        pairs[base_name][file_type] = os.path.join(source_dir, filename)
    
    # Filter out incomplete pairs
    complete_pairs = {}
    for base_name, files in pairs.items():
        if 'im' in files and 'mask' in files:
            complete_pairs[base_name] = files
        else:
            missing = [t for t in ['im', 'mask'] if t not in files]
            print(f"Warning: Incomplete pair for {base_name}, missing: {missing}")
    
    return complete_pairs

def split_data(pairs, train_ratio=0.8, seed=42):
    """
    Split the data into train and test sets
    """
    random.seed(seed)
    base_names = list(pairs.keys())
    random.shuffle(base_names)
    
    split_idx = int(len(base_names) * train_ratio)
    train_names = base_names[:split_idx]
    test_names = base_names[split_idx:]
    
    return train_names, test_names

def create_symlinks(pairs, names, target_dir):
    """
    Create symlinks for the specified names to the target directory
    The data loader expects matching base names, so we use the same base name for both
    """
    images_dir = os.path.join(target_dir, 'images')
    masks_dir = os.path.join(target_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    for name in names:
        if name in pairs:
            # Create symlink for image (same base name as mask)
            src_image = pairs[name]['im']
            dst_image = os.path.join(images_dir, f"{name}.tif")
            if os.path.exists(dst_image):
                os.remove(dst_image)
            os.symlink(os.path.abspath(src_image), dst_image)
            
            # Create symlink for mask (same base name as image)
            src_mask = pairs[name]['mask']
            dst_mask = os.path.join(masks_dir, f"{name}.tif")
            if os.path.exists(dst_mask):
                os.remove(dst_mask)
            os.symlink(os.path.abspath(src_mask), dst_mask)

def main():
    # Paths
    source_dir = "/home2/s5946301/cGAN-Seg-yeast/yeaz_data"
    target_base_dir = "/home2/s5946301/cGAN-Seg-yeast/yeaz_dataset"
    train_dir = os.path.join(target_base_dir, "train")
    test_dir = os.path.join(target_base_dir, "test")
    
    print("🔍 Analyzing yeaz data...")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Collect image-mask pairs
    pairs = collect_image_mask_pairs(source_dir)
    
    if not pairs:
        print("❌ No valid image-mask pairs found!")
        return
    
    print(f"✅ Found {len(pairs)} complete image-mask pairs")
    
    # Split data
    train_names, test_names = split_data(pairs, train_ratio=0.95)
    
    print(f"📊 Split: {len(train_names)} train, {len(test_names)} test")
    print(f"   Train ratio: {len(train_names)/(len(train_names)+len(test_names)):.2%}")
    
    # Create target directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create symlinks
    print("🔗 Creating symlinks for training data...")
    create_symlinks(pairs, train_names, train_dir)
    
    print("🔗 Creating symlinks for test data...")
    create_symlinks(pairs, test_names, test_dir)
    
    print("✅ Data organization complete!")
    print(f"\nDataset structure:")
    print(f"  📁 {train_dir}/")
    print(f"     📁 images/ ({len(train_names)} files)")
    print(f"     📁 masks/ ({len(train_names)} files)")
    print(f"  📁 {test_dir}/")
    print(f"     📁 images/ ({len(test_names)} files)")
    print(f"     📁 masks/ ({len(test_names)} files)")
    
    # Show some examples
    print(f"\nExample base names in training set:")
    for i, name in enumerate(train_names[:5]):
        print(f"  {i+1}. {name}")
    
    print(f"\nExample base names in test set:")
    for i, name in enumerate(test_names[:5]):
        print(f"  {i+1}. {name}")

if __name__ == "__main__":
    main()
