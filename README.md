# cGAN-Seg: Enhanced Cell Segmentation with Generative Adversarial Networks

This repository is a **fork** of the original cGAN-Seg research, which presents "cGAN-Seg: A Generative Adversarial Network for Enhanced Cell Segmentation with Limited Training Datasets." The original approach utilizes a modified CycleGAN architecture to train cell segmentation models effectively with a limited number of annotated cell images, addressing the challenge of scarce annotated data in microscopy imaging.

![Screenshot](Figure1.png)

## Original Research

The original cGAN-Seg method diversifies training data and improves synthetic sample authenticity, thereby enhancing the segmentation model's accuracy and robustness. For the original research, please refer to the [original cGAN-Seg repository](https://github.com/abzargar/cGAN-Seg).

## Our Contributions

This fork adds the following enhancements to the original cGAN-Seg framework:

1. **Enhanced Yeast Cell Segmentation**: Optimized training approach specifically for high-density yeast cell images (`imp_train/` directory)
2. **Direct Mask-to-Image Generation**: New functionality to generate synthetic images directly from mask files (`generate_from_masks.py`)
3. **TIFF Processing Pipeline**: Specialized processing for multi-frame TIFF movies with automatic segmentation (`tiff_processing/` directory)

## Datasets

The original research employed several segmentation datasets in their experiments, including DeepSea, CellPose, LiveCell, and Cell Tracking Challenge images alongside their corresponding masks. Access the datasets here: [cGAN-Seg_datasets](https://drive.google.com/drive/folders/1ZYkNA4mm6xaAjm51vfg1YL2kgDfp4OPD?usp=sharing)

## Requirements

* Optional: Create a conda or Python virtual environment.
* Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

### Original cGAN-Seg Functionality

#### Training the cGAN-Seg Model (Original)
Train the model with your dataset or use the provided cGAN-Seg dataset. Adjust hyperparameters, including the segmentation model type (DeepSea, CellPose, or UNET) and early stopping 'patience' value (recommended: >=500 if dataset is limited).

```bash
python train.py --seg_model DeepSea --train_set_dir .../cGAN-Seg_datasets/DeepSea_datasets/mouse_embryonic_stem_cell_dataset/train/ --lr 0.0001 --p_vanilla 0.2 --p_diff 0.2 --patience 500 --output_dir tmp/
```

#### Testing the Segmentation Model (Original)
Evaluate your segmentation model using the cGAN-Seg dataset. Download cGAN-Seg checkpoints (Seg.pth) for testing from [cGAN-Seg_checkpoints](https://drive.google.com/drive/folders/19V11wBxALoABvsfq8nNZqP1PmuC4Cl_R?usp=sharing).

```bash
python test_segmentation_model.py --seg_model DeepSea --test_set_dir .../cGAN-Seg_datasets/DeepSea_datasets/mouse_embryonic_stem_cell_dataset/test/ --seg_ckpt_dir .../cGAN-Seg_checkpoints/deepsea_model/mouse_embryonic_stem_cells/Seg.pth --output_dir tmp/
```

#### Testing the Generator Model (Original)
Evaluate the StyleUNET generator's performance using synthetic or real mask images. Download cGAN-Seg checkpoints (Gen.pth) from [cGAN-Seg_checkpoints](https://drive.google.com/drive/folders/19V11wBxALoABvsfq8nNZqP1PmuC4Cl_R?usp=sharing).

```bash
python test_generation_model.py --test_set_dir .../cGAN-Seg_datasets/DeepSea_datasets/mouse_embryonic_stem_cell_dataset/test/ --gen_ckpt_dir .../cGAN-Seg_checkpoints/deepsea_model/mouse_embryonic_stem_cells/Gen.pth --output_dir tmp/
```

#### Generating Synthetic High-Density and Colony-Like Cell Images (Original)
Generate synthetic high-density and colony-like cell images using the trained StyleUNET generator:

```bash
# First, generate synthetic high-density masks
python generate_high_density_masks.py --real_mask_set_dir .../test/masks/ --synthetic_mask_set_dir /synthetic_high_density_dataset/

# Then, generate corresponding synthetic cell images
python test_generation_model.py --test_set_dir /synthetic_high_density_dataset/ --gen_ckpt_dir .../Gen.pth --output_dir /synthetic_high_density_dataset/
```

### Our Enhanced Functionality

#### Enhanced Training for Yeast Cells (New)
For better performance on yeast cell datasets, use our optimized training script:

```bash
python imp_train/train_yeast_optimized.py --seg_model DeepSea --train_set_dir yeaz_dataset_fixed/ --lr 0.0001 --p_vanilla 0.2 --p_diff 0.2 --patience 500 --output_dir imp_train/test_yeast_output/
```

#### Direct Image Generation from Masks (New)
Generate synthetic cell images directly from mask files using a trained generator model:

```bash
python generate_from_masks.py --masks_dir test_gen_from_masks/ --gen_ckpt_dir path/to/Gen.pth --output_dir generated_images/
```

This script:
- Takes a directory of mask images as input
- Uses a trained StyleUNET generator to create synthetic cell images
- Handles both binary and labeled masks automatically
- Saves generated images with original filenames

#### TIFF Processing and Segmentation (New)
For processing multi-frame TIFF movies and segmenting yeast cells, use our specialized processing script:

```bash
# Extract frames 10-19 from a TIFF movie and segment them
python tiff_processing/extract_and_segment.py --tiff_path raw_data/yst_movie.tif

# Custom frame range with overlay visualizations
python tiff_processing/extract_and_segment.py --tiff_path raw_data/yst_movie.tif --start_frame 70 --num_frames 10 --model_path imp_train/test_yeast_output/Seg.pth --output_dir my_results --save_overlays
```

This script:
- Extracts specified frames from multi-frame TIFF files
- Segments each frame using trained DeepSea models
- Applies post-processing (remove small objects, connected components)
- Optionally creates overlay visualizations

## Key Features

### Original cGAN-Seg Architecture
- **Modified CycleGAN**: Enhanced architecture for cell segmentation
- **StyleUNET Generator**: Generates high-quality synthetic cell images
- **Multiple Segmentation Models**: Supports DeepSea, CellPose, and UNET architectures
- **Differential Augmentation**: Improves training stability

### Our Enhancements
- **Optimized Yeast Training**: Better loss balancing for dense cell segmentation
- **Direct Mask Processing**: Generate images directly from mask files
- **TIFF Movie Processing**: Automated frame extraction and segmentation
- **Enhanced Augmentation**: Tailored for brightfield yeast images

## Project Structure

```
├── train.py                          # Original training script
├── test_segmentation_model.py        # Original segmentation testing
├── test_generation_model.py          # Original generator testing
├── generate_high_density_masks.py    # Original high-density generation
├── generate_from_masks.py            # NEW: Generate images from mask files
├── imp_train/                        # NEW: Enhanced training for yeast cells
│   ├── train_yeast_optimized.py     # Optimized training script
│   └── ...
├── tiff_processing/                  # NEW: TIFF movie processing
│   ├── extract_and_segment.py       # Extract frames and segment
│   └── ...
├── unet_models.py                    # Model architectures
├── utils.py                          # Utility functions
├── data.py                           # Data loading and processing
├── loss.py                           # Loss functions
├── evaluate.py                       # Evaluation metrics
└── transforms.py                     # Data transformations
```

## Summary of Contributions

This fork extends the original cGAN-Seg framework with:

1. **Enhanced Yeast Cell Processing**: `imp_train/train_yeast_optimized.py` - Optimized training specifically for high-density yeast cell datasets
2. **Direct Mask-to-Image Generation**: `generate_from_masks.py` - Utility to generate synthetic images directly from mask files using trained generators
3. **TIFF Processing Pipeline**: `tiff_processing/extract_and_segment.py` - Automated processing of multi-frame TIFF movies with segmentation

These additions make the framework more practical for yeast cell analysis workflows while maintaining compatibility with the original cGAN-Seg approach.

## Original Research Citation

For the original cGAN-Seg method, please refer to the original research and repository.

## Contact

For questions about the original cGAN-Seg method, contact: abzargar@ucsc.edu

For questions about the enhancements in this fork, please open an issue in this repository.
