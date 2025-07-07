# TIFF Processing and Segmentation

This folder contains scripts for processing TIFF movie files and segmenting yeast cells using trained cGAN-Seg models.

## Main Script: extract_and_segment.py

This script extracts frames from a multi-frame TIFF file and segments them using a trained **DeepSea** segmentation model. The preprocessing and inference pipeline exactly matches the one used in `test_segmentation_model.py`.

### Key Features

- **Exact Test Pipeline**: Uses the same preprocessing (resize to [512, 768], normalize with mean=0.5, std=0.5)
- **DeepSea Model**: Loads and uses the DeepSea architecture with n_channels=1, n_classes=2
- **Post-processing**: Applies the same post-processing as in evaluation (remove_small_objects, connected components)
- **Visualizations**: Optional overlay generation showing segmentation results

### Usage

```bash
# Basic usage - extract frames 10-19 from a TIFF movie
python extract_and_segment.py --tiff_path /path/to/your/movie.tif

# Custom frame range with overlay visualizations
python extract_and_segment.py --tiff_path /path/to/your/movie.tif --start_frame 5 --num_frames 15 --save_overlays

# Specify custom model and output directory
python extract_and_segment.py \
    --tiff_path ../raw_data/yst_movie.tif \
    --channel 0\
    --model_path ../imp_train/test_yeast_output/Seg.pth \
    --start_frame 70 \
    --num_frames 10 \
    --output_dir my_results_more \
    --save_overlays

# Force CPU usage (if you don't want to use GPU)
python extract_and_segment.py --tiff_path /path/to/your/movie.tif --device cpu
```

### Arguments

- `--tiff_path`: Path to the input TIFF movie file (required)
- `--model_path`: Path to the trained DeepSea segmentation model (default: ../imp_train/test_yeast_output/Gen.pth)
- `--start_frame`: Starting frame number, 0-indexed (default: 10)
- `--num_frames`: Number of frames to extract (default: 10)
- `--output_dir`: Base output directory (default: results)
- `--save_overlays`: Save overlay visualizations (default: False)
- `--device`: Device to use: 'cuda', 'cpu', or 'auto' (default: auto)

### Output Structure

```
results/
├── extracted_frames/
│   ├── frame_0010.tif
│   ├── frame_0011.tif
│   └── ...
└── segmentation_masks/
    ├── masks/
    │   ├── mask_frame_0010.tif  # Labeled segmentation masks
    │   ├── mask_frame_0011.tif
    │   └── ...
    ├── overlays/               # Only if --save_overlays is used
    │   ├── overlay_frame_0010.png
    │   ├── overlay_frame_0011.png
    │   └── ...
    └── input_images/           # Only if --save_overlays is used
        ├── input_frame_0010.png
        ├── input_frame_0011.png
        └── ...
```

### Processing Details

The script now exactly replicates the preprocessing and inference pipeline from `test_segmentation_model.py`:

1. **Preprocessing**: Images are resized to [512, 768] and normalized with mean=0.5, std=0.5
2. **Model**: Uses DeepSea architecture (n_channels=1, n_classes=2)
3. **Inference**: Raw model output → argmax → labeled mask
4. **Post-processing**: remove_small_objects (min_size=15) + connected components labeling
5. **Output**: Labeled masks where each connected component has a unique ID

### Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

### For Your Google Drive TIFF File

After downloading the TIFF file from your Google Drive link, you can run:

```bash
python extract_and_segment.py --tiff_path /path/to/downloaded/movie.tif --start_frame 10 --num_frames 10
```

This will extract frames 10-19 (10 frames total) and segment each one using your trained model.
