# Import necessary libraries and packages
import os
import argparse
import cv2
from unet_models import StyleUnetGenerator
from utils import set_requires_grad, mixed_list, noise_list, image_noise
import torch.utils.data as data
import transforms as transforms
import torch
import shutil
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset

# Setting the seed for generating random numbers for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_mask_files(masks_folder):
    """Get all mask files from the specified folder."""
    mask_files = []
    for mask_name in sorted(os.listdir(masks_folder)):
        if mask_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            mask_path = os.path.join(masks_folder, mask_name)
            mask_files.append(mask_path)
    return mask_files

class MaskOnlyDataset(Dataset):
    """Dataset class for loading only mask images."""
    def __init__(self, mask_files, transforms=None):
        self.mask_files = mask_files
        self.transforms = transforms

    def __len__(self):
        return len(self.mask_files)

    def preprocess(self, mask, transforms):
        """Preprocess mask using transforms."""
        # For mask-only dataset, we use the same image for both image and mask in transforms
        tensor_img, tensor_mask = transforms(mask, mask)
        return tensor_mask

    def __getitem__(self, idx):
        mask_file = self.mask_files[idx]
        
        # Load mask
        mask = cv2.imread(mask_file, 0)
        
        # Convert labeled mask to binary mask if needed
        mask = self.convert_to_binary_mask(mask)
        mask = mask.astype('float32')
        
        # Normalize mask to 0-255 range for transforms
        mask_normalized = (255 * mask).astype(np.uint8)
        
        # Apply transforms
        tensor_mask = self.preprocess(mask_normalized, self.transforms)
        
        return {
            'mask': tensor_mask,
            'filename': os.path.basename(mask_file)
        }
    
    def convert_to_binary_mask(self, mask):
        """Convert labeled mask to binary mask if needed."""
        # Check if mask is already binary (only contains 0 and 1 values)
        unique_values = np.unique(mask)
        
        if len(unique_values) <= 2 and np.all(np.isin(unique_values, [0, 1])):
            # Already binary
            return mask > 0
        elif len(unique_values) <= 2 and np.all(np.isin(unique_values, [0, 255])):
            # Binary but with 0 and 255 values
            return mask > 0
        else:
            # Labeled mask - convert to binary by treating any non-zero value as foreground
            print(f"Converting labeled mask to binary (found {len(unique_values)} unique values)")
            return mask > 0

def normalize_image(image):
    """Normalize the image to a range of [0, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val + 1e-8)  # Adding a small constant to avoid division by zero

def save_generated_images(output_dir, generated_images, filenames):
    """Save generated images to the output directory."""
    for i, (gen_img, filename) in enumerate(zip(generated_images, filenames)):
        gen_img_save = (gen_img * 255).astype(np.uint8)
        # Use original filename or create numbered filename
        output_filename = filename if filename else f'generated_{i:04d}.png'
        cv2.imwrite(os.path.join(output_dir, output_filename), gen_img_save)

def generate_images(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    """Generate images from masks using the trained generator."""
    # Using CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Applying transformations on the mask data
    mask_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])
    
    # Get mask files
    mask_files = get_mask_files(args.masks_dir)
    assert len(mask_files) > 0, f'No mask files found in {args.masks_dir}'
    print(f"Found {len(mask_files)} mask files")

    # Load the mask dataset and apply the transformations
    mask_data = MaskOnlyDataset(mask_files, transforms=mask_transforms)

    # Create a dataloader for the mask dataset
    mask_iterator = data.DataLoader(mask_data, batch_size=batch_size, shuffle=False)

    # Load the generator model
    Gen = StyleUnetGenerator(style_latent_dim=128, output_nc=1).to(device)
    Gen.load_state_dict(torch.load(args.gen_ckpt_dir, map_location=device))
    print(f"Loaded generator model from {args.gen_ckpt_dir}")

    # Set the model to evaluation mode
    Gen.eval()

    generated_images = []
    filenames = []
    
    set_requires_grad(Gen, False)
    
    with torch.no_grad():
        # Iterating over batches of mask data
        for step, batch in enumerate(tqdm(mask_iterator, desc="Generating images")):
            mask = batch['mask'].to(device=device, dtype=torch.float32)
            filename = batch['filename'][0]  # Get filename from batch
            
            # Generate style and noise
            style = mixed_list(mask.shape[0], 5, 128, device=device) if random.random() < 0.9 else noise_list(mask.shape[0], 5, 128, device=device)
            im_noise = image_noise(mask.shape[0], image_size, device=device)

            # Generate fake image
            fake_img = Gen(mask, style, im_noise)
            fake_img = normalize_image(fake_img.cpu().numpy()[0, 0, :, :])
            
            generated_images.append(fake_img)
            filenames.append(filename)

    # Save generated images
    save_generated_images(args.output_dir, generated_images, filenames)
    print(f"Generated {len(generated_images)} images and saved to {args.output_dir}")

if __name__ == "__main__":
    # Argument parsing
    ap = argparse.ArgumentParser(description="Generate images from masks using a trained GAN generator")
    ap.add_argument("--masks_dir", required=True, type=str, help="Path to the directory containing mask images")
    ap.add_argument("--gen_ckpt_dir", required=True, type=str, help="Path to the generator model checkpoint")
    ap.add_argument("--output_dir", required=True, type=str, help="Path for saving the generated images")
    ap.add_argument("--image_size", nargs=2, type=int, default=[512, 768], help="Image size as height width (default: 512 768)")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (default: 1)")

    args = ap.parse_args()

    # Check if masks directory exists
    assert os.path.isdir(args.masks_dir), f'No such file or directory: {args.masks_dir}'
    
    # Check if generator checkpoint exists
    assert os.path.isfile(args.gen_ckpt_dir), f'No such file: {args.gen_ckpt_dir}'

    # Create output directory
    if os.path.exists(args.output_dir):
        # Remove the existing directory and all its contents
        shutil.rmtree(args.output_dir)
        print(f"Removed existing output directory: {args.output_dir}")

    # Create the new directory
    os.makedirs(args.output_dir)
    print(f"Created output directory: {args.output_dir}")
    
    # Call the generation function
    generate_images(args, image_size=args.image_size, batch_size=args.batch_size)
