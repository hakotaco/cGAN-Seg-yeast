#!/usr/bin/env python3
"""
Script to extract frames from a TIFF movie and segment them using trained cGAN-Seg model.

Usage:
    python extract_and_segment.py --tiff_path path/to/movie.tif --start_frame 10 --num_frames 10 --model_path path/to/Gen.pth

This script:
1. Extracts specified frames from a multi-frame TIFF file
2. Saves them as individual TIFF images
3. Segments each frame using the trained segmentation model
4. Saves the segmentation masks
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tifffile
from pathlib import Path
import cv2
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
import torchvision.transforms as T

# Add parent directory to access the model modules
sys.path.append('..')
sys.path.append('/home2/s5946301/cGAN-Seg-yeast')

from unet_models import DeepSea
from utils import visualize_segmentation

class TIFFProcessor:
    def __init__(self, model_path, channel=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.channel = channel
        self.model = None
        
        # Match the exact preprocessing pipeline from test_segmentation_model.py
        self.image_size = [512, 768]
        self.image_means = [0.5]
        self.image_stds = [0.5]
        
        # Use standard torchvision transforms for inference-only
        # Remove ToPILImage since we'll convert manually
        self.transforms = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=self.image_means, std=self.image_stds)
        ])
        
        self.load_model()
        
    def load_model(self):
        """Load the trained segmentation model"""
        print(f"Loading model from {self.model_path}")
        
        # Initialize the DeepSea segmentation model
        # Using the same parameters as in test_segmentation_model.py
        self.model = DeepSea(n_channels=1, n_classes=2)
        
        # Load the trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def extract_frames(self, tiff_path, start_frame=0, num_frames=10, output_dir="extracted_frames"):
        """Extract frames from TIFF movie using proper multi-dimensional handling"""
        print(f"Reading TIFF file: {tiff_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Use the corrected extraction function
        images = self.extract_channel_timestep_images(tiff_path, self.channel, start_frame, num_frames)
        
        extracted_paths = []
        
        # Save each extracted image
        for i, image in enumerate(images):
            frame_idx = start_frame + i
            print(f"Extracted frame {frame_idx}: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
            
            # Convert to appropriate data type for saving
            if image.dtype == np.float64 or image.dtype == np.float32:
                # Normalize to 0-65535 range for uint16
                image_norm = (image - image.min()) / (image.max() - image.min()) if image.max() > image.min() else image
                image = (image_norm * 65535).astype(np.uint16)
            elif image.dtype == np.uint8:
                # Convert uint8 to uint16 by scaling
                image = image.astype(np.uint16) * 257  # 257 = 65535/255
            elif image.dtype != np.uint16:
                # Convert other types to uint16
                image = image.astype(np.uint16)
            
            # Save frame as individual TIFF
            frame_filename = output_dir / f"frame_{frame_idx:04d}.tif"
            tifffile.imwrite(frame_filename, image)
            extracted_paths.append(frame_filename)
            
            print(f"Saved frame {frame_idx} -> {frame_filename}")
        
        return extracted_paths
    
    def extract_channel_timestep_images(self, tiff_path, channel=0, start_timestep=20, count=10):
        """
        Extracts `count` images from a multi-dimensional TIFF file from a specified channel
        and starting timestep.

        Parameters:
            tiff_path (str): Path to the TIFF file.
            channel (int): Channel index to extract from (0-based).
            start_timestep (int): Starting time index (0-based).
            count (int): Number of frames to extract.

        Returns:
            List of 2D numpy arrays (images).
        """
        with tifffile.TiffFile(tiff_path) as tif:
            arr = tif.asarray()

            # Inspect shape
            shape = arr.shape
            print(f"TIFF shape: {shape}")

            if arr.ndim == 5:
                # Typical shape: (t, c, z, y, x) or (c, z, t, y, x)
                # Let's assume: (t, c, y, x) (no z)
                t, c, y, x = shape
                assert channel < c and start_timestep + count <= t
                images = [arr[start_timestep + i, channel, :, :] for i in range(count)]

            elif arr.ndim == 4:
                # Shape may be (t, c, y, x)
                t, c, y, x = shape
                assert channel < c and start_timestep + count <= t
                images = [arr[start_timestep + i, channel, :, :] for i in range(count)]

            elif arr.ndim == 3:
                raise ValueError("TIFF doesn't have time/channel info (shape is 3D).")

            else:
                raise ValueError(f"Unsupported TIFF shape: {shape}")

        return images
    
    def preprocess_image(self, image):
        """Preprocess image exactly as in test_segmentation_model.py"""
        print(f"Preprocessing image: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
        
        # Handle multi-channel images - take first channel if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # H x W x C
                image = image[:, :, 0]
            elif image.shape[0] == 3:  # C x H x W  
                image = image[0, :, :]
        
        # Convert to uint8 if not already (PIL expects uint8)
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            if image.max() > image.min():
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        
        print(f"After uint8 conversion: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
        
        # Convert to PIL Image first
        pil_image = Image.fromarray(image)
        
        # Apply the same transforms as in test_segmentation_model.py
        image_tensor = self.transforms(pil_image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        print(f"Final tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}, min={image_tensor.min()}, max={image_tensor.max()}")
        
        return image_tensor
    
    def postprocess_mask(self, mask_tensor, input_image=None, save_overlay=False, output_path=None):
        """Convert model output to segmentation mask exactly as in evaluate.py"""
        # Get the predicted class using argmax (exactly as in evaluate_segmentation)
        mask_pred = mask_tensor.argmax(dim=1)
        
        # Move to CPU and convert to numpy
        mask_pred_np = mask_pred.squeeze().cpu().numpy()
        
        print(f"Raw prediction: shape={mask_pred_np.shape}, unique values={np.unique(mask_pred_np)}, sum={np.sum(mask_pred_np)}")
        
        # Create binary mask
        binary_mask = mask_pred_np > 0
        print(f"Binary mask before cleaning: sum={np.sum(binary_mask)}")
        
        # Apply post-processing exactly as in evaluate.py
        # Remove small objects with min_size=15 and connectivity=1
        mask_cleaned = remove_small_objects(binary_mask, min_size=15, connectivity=1)
        print(f"Binary mask after cleaning: sum={np.sum(mask_cleaned)}")
        
        # Apply connected components labeling for visualization
        labeled_mask, num_components = ndi.label(mask_cleaned)
        print(f"Labeled mask: {num_components} components, max label={np.max(labeled_mask)}")
        
        # Create binary mask for saving (0 and 255 for better visualization)
        binary_mask_for_saving = (mask_cleaned > 0).astype(np.uint8) * 255
        
        # Create visualization if requested
        if save_overlay and input_image is not None and output_path is not None:
            # Normalize input image to 0-255 range (as done in evaluate.py)
            if input_image.max() > input_image.min():
                img_norm = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image)) * 255
            else:
                img_norm = np.zeros_like(input_image) + 128  # Gray background if no variation
            
            # Ensure uint8 data type for OpenCV
            img_norm = img_norm.astype(np.uint8)
            
            try:
                overlay_img = visualize_segmentation(labeled_mask, inp_img=img_norm, overlay_img=True)
                cv2.imwrite(str(output_path), overlay_img)
            except Exception as e:
                print(f"Warning: Could not create overlay visualization: {e}")
                # Save a simple grayscale version instead
                cv2.imwrite(str(output_path), img_norm)
        
        return binary_mask_for_saving, labeled_mask
    
    def segment_frames(self, frame_paths, output_dir="segmentation_results", save_overlays=True):
        """Segment extracted frames using the trained model exactly as in test_segmentation_model.py"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"
        input_images_dir = output_dir / "input_images"
        
        masks_dir.mkdir(exist_ok=True)
        if save_overlays:
            overlays_dir.mkdir(exist_ok=True)
            input_images_dir.mkdir(exist_ok=True)
        
        segmented_paths = []
        
        print(f"Segmenting {len(frame_paths)} frames...")
        
        self.model.eval()
        with torch.no_grad():
            for i, frame_path in enumerate(frame_paths):
                # Load frame
                original_frame = tifffile.imread(frame_path)
                print(f"Processing {frame_path}: shape {original_frame.shape}")
                
                # Preprocess exactly as in test pipeline
                input_tensor = self.preprocess_image(original_frame)
                
                # Segment (inference exactly as in evaluate.py)
                mask_preds = self.model(input_tensor)
                
                # Postprocess exactly as in evaluate.py
                overlay_path = overlays_dir / f"overlay_{frame_path.stem}.png" if save_overlays else None
                binary_mask_for_saving, labeled_mask = self.postprocess_mask(
                    mask_preds, 
                    input_image=original_frame, 
                    save_overlay=save_overlays,
                    output_path=overlay_path
                )
                
                # Save the binary mask as TIFF
                mask_filename = masks_dir / f"mask_{frame_path.stem}.tif"
                tifffile.imwrite(mask_filename, binary_mask_for_saving)
                segmented_paths.append(mask_filename)
                
                print(f"Saved binary mask: shape={binary_mask_for_saving.shape}, unique values={np.unique(binary_mask_for_saving)}, sum={np.sum(binary_mask_for_saving)}")
                
                # Save input image for reference if overlays are enabled
                if save_overlays:
                    try:
                        if original_frame.max() > original_frame.min():
                            input_img_norm = (original_frame - np.min(original_frame)) / (np.max(original_frame) - np.min(original_frame)) * 255
                        else:
                            input_img_norm = np.zeros_like(original_frame) + 128
                        
                        input_img_norm = input_img_norm.astype(np.uint8)
                        input_img_path = input_images_dir / f"input_{frame_path.stem}.png"
                        overlay_without_input = visualize_segmentation(labeled_mask, inp_img=input_img_norm, overlay_img=True)
                        cv2.imwrite(str(input_img_path), overlay_without_input)
                    except Exception as e:
                        print(f"Warning: Could not save input image overlay: {e}")
                        # Save just the normalized image
                        cv2.imwrite(str(input_img_path), input_img_norm)
                
                print(f"Segmented {frame_path.name} -> {mask_filename}")
                if save_overlays:
                    print(f"  Overlay saved: {overlay_path}")
        
        return segmented_paths
    
    def process_tiff_movie(self, tiff_path, start_frame=0, num_frames=10, 
                          extract_dir="extracted_frames", segment_dir="segmentation_results", save_overlays=True):
        """Complete pipeline: extract frames and segment them"""
        print("="*50)
        print("Starting TIFF Movie Processing Pipeline")
        print("="*50)
        
        # Step 1: Extract frames
        frame_paths = self.extract_frames(tiff_path, start_frame, num_frames, extract_dir)
        
        print("\n" + "="*50)
        print("Starting Segmentation")
        print("="*50)
        
        # Step 2: Segment frames
        mask_paths = self.segment_frames(frame_paths, segment_dir, save_overlays)
        
        print("\n" + "="*50)
        print("Processing Complete!")
        print("="*50)
        print(f"Extracted frames: {len(frame_paths)}")
        print(f"Segmentation masks: {len(mask_paths)}")
        print(f"Frames saved in: {extract_dir}")
        print(f"Masks saved in: {segment_dir}")
        
        return frame_paths, mask_paths


def main():
    parser = argparse.ArgumentParser(description="Extract frames from TIFF movie and segment using trained model")
    
    parser.add_argument("--tiff_path", type=str, default="../raw_data/yst_movie.tif",
                       help="Path to the input TIFF movie file")
    parser.add_argument("--model_path", type=str, 
                       default="../imp_train/test_yeast_output/Seg.pth",
                       help="Path to the trained DeepSea segmentation model (default: ../imp_train/test_yeast_output/Seg.pth)")
    parser.add_argument("--start_frame", type=int, default=10,
                       help="Starting frame number (0-indexed, default: 10)")
    parser.add_argument("--num_frames", type=int, default=10,
                       help="Number of frames to extract (default: 10)")
    parser.add_argument("--output_dir", type=str, default="results2",
                       help="Base output directory (default: results)")
    parser.add_argument("--channel", type=int, default=0,
                       help="Channel to extract from multi-channel TIFF (0-indexed, default: 0)")
    parser.add_argument("--save_overlays", action="store_true",
                       help="Save overlay visualizations (default: False)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Validate inputs
    if not os.path.exists(args.tiff_path):
        print(f"Error: TIFF file not found: {args.tiff_path}")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Create base output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(exist_ok=True)
    
    extract_dir = output_base / "extracted_frames"
    segment_dir = output_base / "segmentation_masks"
    
    try:
        # Initialize processor
        processor = TIFFProcessor(args.model_path, args.channel, device)
        
        # Process the TIFF movie
        frame_paths, mask_paths = processor.process_tiff_movie(
            args.tiff_path, 
            args.start_frame, 
            args.num_frames,
            extract_dir,
            segment_dir,
            args.save_overlays
        )
        
        print(f"\nSuccess! Check the results in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
