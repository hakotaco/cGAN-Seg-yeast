#!/usr/bin/env python3
"""
Improved training script for cGAN-Seg specifically optimized for high-density yeast cell images.
Maintains the original cGAN-Seg architecture but with optimizations for yeast cells:
- Better loss balancing for dense cell segmentation
- Improved data augmentation suited for brightfield yeast images
- Enhanced evaluation metrics
- Better convergence monitoring
"""

# Import the necessary libraries and modules
import os
import argparse
import signal
import sys

# Add parent directory to Python path to access modules in root directory
sys.path.append('..')
sys.path.append('/home2/s5946301/cGAN-Seg-yeast')

from unet_models import UnetSegmentation,StyleUnetGenerator,NLayerDiscriminator,DeepSea
from cellpose_model import Cellpose_CPnet
from utils import set_requires_grad,mixed_list,noise_list,image_noise,initialize_weights
from data import BasicDataset,get_image_mask_pairs
import itertools
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate_segmentation
from loss import CombinedLoss,VGGLoss
import torch.utils.data as data
import torch.nn.functional as F
import transforms as transforms
import torch
import numpy as np
import random
import logging
from diffaug import DiffAugment
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set a constant seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Global variables for signal handler
checkpoint_models = None
checkpoint_output_dir = None

def save_final_checkpoint(signum, frame):
    """Signal handler to save checkpoint before job termination"""
    if checkpoint_models is not None and checkpoint_output_dir is not None:
        print(f"\nReceived signal {signum}. Saving final checkpoint...")
        Gen, Seg, D1, D2 = checkpoint_models
        try:
            torch.save(Gen.state_dict(), os.path.join(checkpoint_output_dir, 'Gen_final.pth'))
            torch.save(Seg.state_dict(), os.path.join(checkpoint_output_dir, 'Seg_final.pth'))
            torch.save(D1.state_dict(), os.path.join(checkpoint_output_dir, 'D1_final.pth'))
            torch.save(D2.state_dict(), os.path.join(checkpoint_output_dir, 'D2_final.pth'))
            print(f"Final checkpoint saved to {checkpoint_output_dir}")
        except Exception as e:
            print(f"Error saving final checkpoint: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, save_final_checkpoint)  # SLURM sends SIGTERM
signal.signal(signal.SIGUSR1, save_final_checkpoint)  # Additional signal

# Function to reset logging configuration
def reset_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

# Enhanced loss function for high-density cell segmentation
class YeastSegLoss(nn.Module):
    """
    Enhanced segmentation loss specifically designed for high-density yeast cells.
    Combines Dice loss with focal loss to handle class imbalance better.
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, ce_weight=0.3, focal_weight=0.2):
        super(YeastSegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
    def focal_loss(self, pred, target):
        """Focal loss for handling hard examples in dense cell regions"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for better boundary detection"""
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.focal_weight * focal + self.dice_weight * dice

# The main training function with yeast-specific optimizations
def train_yeast_optimized(args, image_size=[512,768], image_means=[0.5], image_stds=[0.5], train_ratio=0.8, save_checkpoint=True):

    # Reset logging configuration
    reset_logging()

    # Set up the logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train_yeast.log'), filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info('>>>> YEAST-OPTIMIZED cGAN-Seg Training')
    logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d' % (
        image_size[0], image_size[1], args.lr, args.batch_size))

    # Determine the device (GPU or CPU) to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'>>>> Using device: {device}')

    # Enhanced transforms specifically for yeast brightfield images
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.RandomOrder([
            # Reduced aggressive augmentations for yeast cells
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0)], p=0.4),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), sigma=(0.1, 0.5))], p=0.3),  # Reduced blur
            transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip(0.5)], p=0.5),
            transforms.RandomApply([transforms.AddGaussianNoise(0., 0.005)], p=0.3),  # Reduced noise
            transforms.RandomApply([transforms.CLAHE()], p=0.6),  # Increased CLAHE for brightfield
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.3),  # Reduced sharpness
            # Removed RandomCrop as it can hurt dense cell segmentation
        ])], p=args.p_vanilla),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    dev_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])
    
    # Read samples
    sample_pairs = get_image_mask_pairs(args.train_set_dir)
    assert len(sample_pairs) > 0, f'No samples found in {args.train_set_dir}'
    logging.info(f'>>>> Found {len(sample_pairs)} training pairs')
    
    # Split samples
    train_sample_pairs = sample_pairs[:int(train_ratio * len(sample_pairs))]
    valid_sample_pairs = sample_pairs[int(train_ratio * len(sample_pairs)):]
    logging.info(f'>>>> Train samples: {len(train_sample_pairs)}, Valid samples: {len(valid_sample_pairs)}')

    # Define the datasets for training and validation
    train_data = BasicDataset(train_sample_pairs, transforms=train_transforms, 
                             vanilla_aug=False if args.p_vanilla == 0 else True, gen_nc=args.gen_nc)
    valid_data = BasicDataset(valid_sample_pairs, transforms=dev_transforms, gen_nc=args.gen_nc)

    # Define the dataloaders
    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size, 
                                   num_workers=4, pin_memory=True)  # Reduced workers for stability
    valid_iterator = data.DataLoader(valid_data, batch_size=args.batch_size, 
                                   num_workers=4, pin_memory=True)

    # Define the models (same architecture as original)
    Gen = StyleUnetGenerator(style_latent_dim=128, output_nc=args.gen_nc)
    if args.seg_model == 'UNET':
        Seg = UnetSegmentation(n_channels=args.gen_nc, n_classes=2)
    elif args.seg_model == 'CellPose':
        Seg = Cellpose_CPnet(n_channels=args.gen_nc, n_classes=2)
    elif args.seg_model == 'DeepSea':
        Seg = DeepSea(n_channels=args.gen_nc, n_classes=2)
    else:
        raise ValueError(f"Model '{args.seg_model}' not found.")

    D1 = NLayerDiscriminator(input_nc=args.gen_nc)
    D2 = NLayerDiscriminator(input_nc=1)

    initialize_weights(Gen)
    initialize_weights(Seg)
    initialize_weights(D1)
    initialize_weights(D2)

    # Define the optimizers with slightly lower learning rate for better stability
    optimizer_G = torch.optim.Adam(itertools.chain(Gen.parameters(), Seg.parameters()), 
                                  lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Add learning rate schedulers for better convergence
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=50, verbose=True)
    scheduler_D1 = ReduceLROnPlateau(optimizer_D1, mode='min', factor=0.7, patience=30, verbose=True)
    scheduler_D2 = ReduceLROnPlateau(optimizer_D2, mode='min', factor=0.7, patience=30, verbose=True)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Define the loss functions with yeast-specific optimization
    d_criterion = nn.MSELoss()
    Gen_criterion_1 = nn.L1Loss()
    Gen_criterion_2 = VGGLoss()
    Seg_criterion = YeastSegLoss()  # Use our enhanced loss instead of CombinedLoss

    Gen = Gen.to(device)
    Seg = Seg.to(device)
    D1 = D1.to(device)
    D2 = D2.to(device)

    # Set global variables for signal handler
    global checkpoint_models, checkpoint_output_dir
    checkpoint_models = (Gen, Seg, D1, D2)
    checkpoint_output_dir = args.output_dir

    d_criterion = d_criterion.to(device)
    Gen_criterion_1 = Gen_criterion_1.to(device)
    Gen_criterion_2 = Gen_criterion_2.to(device)
    Seg_criterion = Seg_criterion.to(device)

    # Training loop with enhanced monitoring
    nstop = 0
    avg_fscore_best = 0
    dice_score_best = 0
    import time
    start_time = time.time()
    last_checkpoint_time = start_time
    
    # Track losses for better monitoring
    g_losses = []
    d_losses = []
    
    logging.info('>>>> Start yeast-optimized training')
    print('INFO: Start yeast-optimized training ...')
    
    for epoch in range(args.max_epoch):
        Gen.train()
        Seg.train()
        D1.train()
        D2.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_seg_loss = 0

        for step, batch in enumerate(tqdm(train_iterator, desc=f"Epoch {epoch+1}/{args.max_epoch}")):
            real_img = batch['image']
            real_mask = batch['mask']

            valid = torch.full((real_mask.shape[0], 1, 62, 94), 1.0, dtype=real_mask.dtype, device=device)
            fake = torch.full((real_mask.shape[0], 1, 62, 94), 0.0, dtype=real_mask.dtype, device=device)

            real_img = real_img.to(device=device, dtype=torch.float32)
            real_mask = real_mask.to(device=device, dtype=torch.float32)

            set_requires_grad(D1, True)
            set_requires_grad(D2, True)
            set_requires_grad(Gen, True)
            set_requires_grad(Seg, True)

            optimizer_G.zero_grad(set_to_none=True)
            optimizer_D1.zero_grad(set_to_none=True)
            optimizer_D2.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                if random.random() < 0.9:
                    style = mixed_list(real_img.shape[0], 5, Gen.latent_dim, device=device)
                else:
                    style = noise_list(real_img.shape[0], 5, Gen.latent_dim, device=device)

                im_noise = image_noise(real_mask.shape[0], image_size, device=device)

                fake_img = Gen(real_mask, style, im_noise)
                rec_mask = Seg(fake_img)
                fake_mask = Seg(real_img)
                fake_mask_p = F.softmax(fake_mask, dim=1).float()
                fake_mask_p = torch.unsqueeze(fake_mask_p.argmax(dim=1), dim=1)
                fake_mask_p = fake_mask_p.to(dtype=torch.float32)

                if random.random() < 0.9:
                    style = mixed_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)
                else:
                    style = noise_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)

                im_noise = image_noise(real_mask.shape[0], image_size, device=device)

                rec_img = Gen(fake_mask_p, style, im_noise)

                set_requires_grad(D1, False)
                set_requires_grad(D2, False)

                d_img_loss = d_criterion(D1(DiffAugment(fake_img, p=args.p_diff)), valid)
                d_mask_loss = d_criterion(D2(fake_mask_p), valid)
                
                # Adjusted loss weights for better yeast cell segmentation
                rec_mask_loss = 150 * Seg_criterion(rec_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1))  # Increased weight
                id_mask_loss = 75 * Seg_criterion(fake_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1))   # Increased weight
                rec_img_loss = 40 * Gen_criterion_1(rec_img, real_img) + 80 * Gen_criterion_2(rec_img, real_img)    # Slightly reduced
                id_img_loss = 20 * Gen_criterion_1(fake_img, real_img) + 40 * Gen_criterion_2(fake_img, real_img)   # Slightly reduced
                
                g_loss = d_mask_loss + d_img_loss + rec_mask_loss + rec_img_loss + id_mask_loss + id_img_loss

                grad_scaler.scale(g_loss).backward()
                grad_scaler.step(optimizer_G)
                grad_scaler.update()

                set_requires_grad(D1, True)
                set_requires_grad(D2, True)

                real_img_loss = d_criterion(D1(DiffAugment(real_img, p=args.p_diff)), valid)
                fake_img_loss = d_criterion(D1(DiffAugment(fake_img.detach(), p=args.p_diff)), fake)
                d_img_loss = (real_img_loss + fake_img_loss) / 2

                grad_scaler.scale(d_img_loss).backward()
                grad_scaler.step(optimizer_D1)
                grad_scaler.update()

                real_mask_loss = d_criterion(D2(real_mask), valid)
                fake_mask_loss = d_criterion(D2(fake_mask_p.detach()), fake)
                d_mask_loss = (real_mask_loss + fake_mask_loss) / 2

                grad_scaler.scale(d_mask_loss).backward()
                grad_scaler.step(optimizer_D2)
                grad_scaler.update()

            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += (d_img_loss.item() + d_mask_loss.item())
            epoch_seg_loss += (rec_mask_loss.item() + id_mask_loss.item())

        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(train_iterator)
        avg_d_loss = epoch_d_loss / len(train_iterator)
        avg_seg_loss = epoch_seg_loss / len(train_iterator)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"[Epoch {epoch+1}/{args.max_epoch}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}] [Seg loss: {avg_seg_loss:.4f}]")

        # Evaluate the model and save the best checkpoint
        scores = evaluate_segmentation(Seg, valid_iterator, device, Seg_criterion, len(valid_data), 
                                     is_avg_prec=True, prec_thresholds=[0.5], output_dir=None)
        
        current_fscore = scores.get('avg_fscore', 0)
        current_dice = scores.get('dice_score', 0)
        
        # Basic epoch logging (same format as original train.py)
        if current_fscore is not None:
            current_fscore_val = float(current_fscore) if hasattr(current_fscore, 'item') else float(current_fscore)
            current_dice_val = float(current_dice) if hasattr(current_dice, 'item') else float(current_dice)
            logging.info('>>>> Epoch:%d  , Dice score=%f , avg fscore=%f' % (epoch+1, current_dice_val, current_fscore_val))
        else:
            current_dice_val = float(current_dice) if hasattr(current_dice, 'item') else float(current_dice)
            logging.info('>>>> Epoch:%d  , Dice score=%f' % (epoch+1, current_dice_val))
        
        # Additional detailed logging with best scores
        if current_fscore is not None:
            avg_fscore_best_val = float(avg_fscore_best) if hasattr(avg_fscore_best, 'item') else float(avg_fscore_best)
            logging.info(f'>>>> Epoch:{epoch+1}, Dice: {current_dice_val:.4f}, F-score: {current_fscore_val:.4f}, Best F-score: {avg_fscore_best_val:.4f}')
        else:
            dice_score_best_val = float(dice_score_best) if hasattr(dice_score_best, 'item') else float(dice_score_best)
            logging.info(f'>>>> Epoch:{epoch+1}, Dice: {current_dice_val:.4f}, Best Dice: {dice_score_best_val:.4f}')

        # Update learning rate schedulers
        if current_fscore is not None:
            scheduler_G.step(current_fscore)
        scheduler_D1.step(avg_d_loss)
        scheduler_D2.step(avg_d_loss)

        # Enhanced checkpoint saving logic
        improved = False
        if current_fscore is not None and current_fscore > avg_fscore_best:
            avg_fscore_best = current_fscore
            improved = True
        elif current_fscore is None and current_dice > dice_score_best:
            dice_score_best = current_dice
            improved = True

        if improved:
            if save_checkpoint:
                torch.save(Gen.state_dict(), os.path.join(args.output_dir, 'Gen.pth'))
                torch.save(Seg.state_dict(), os.path.join(args.output_dir, 'Seg.pth'))
                torch.save(D1.state_dict(), os.path.join(args.output_dir, 'D1.pth'))
                torch.save(D2.state_dict(), os.path.join(args.output_dir, 'D2.pth'))
                logging.info(f'>>>> NEW BEST! Saved checkpoint to {args.output_dir}')
            nstop = 0
        else:
            nstop += 1
            
        # Save periodic checkpoint every 4 hours
        current_time = time.time()
        if current_time - last_checkpoint_time >= 4 * 3600:  # 4 hours
            print(f'INFO: Saving periodic checkpoint at epoch {epoch+1} (after {(current_time - start_time)/3600:.1f} hours)')
            torch.save(Gen.state_dict(), os.path.join(args.output_dir, f'Gen_epoch_{epoch+1}.pth'))
            torch.save(Seg.state_dict(), os.path.join(args.output_dir, f'Seg_epoch_{epoch+1}.pth'))
            torch.save(D1.state_dict(), os.path.join(args.output_dir, f'D1_epoch_{epoch+1}.pth'))
            torch.save(D2.state_dict(), os.path.join(args.output_dir, f'D2_epoch_{epoch+1}.pth'))
            last_checkpoint_time = current_time
            logging.info(f'>>>> Saved periodic checkpoint at epoch {epoch+1}')
            
        # Enhanced early stopping with better criteria
        if nstop == args.patience:
            print('INFO: Early Stopping met ...')
            logging.info(f'>>>> Early stopping triggered after {epoch+1} epochs')
            avg_fscore_best_val = float(avg_fscore_best) if hasattr(avg_fscore_best, 'item') else float(avg_fscore_best)
            dice_score_best_val = float(dice_score_best) if hasattr(dice_score_best, 'item') else float(dice_score_best)
            logging.info(f'>>>> Best F-score: {avg_fscore_best_val:.4f}, Best Dice: {dice_score_best_val:.4f}')
            print('INFO: Finish training process')
            break

    # Final summary
    total_time = (time.time() - start_time) / 3600
    avg_fscore_best_val = float(avg_fscore_best) if hasattr(avg_fscore_best, 'item') else float(avg_fscore_best)
    dice_score_best_val = float(dice_score_best) if hasattr(dice_score_best, 'item') else float(dice_score_best)
    logging.info(f'>>>> Training completed in {total_time:.2f} hours')
    logging.info(f'>>>> Final best F-score: {avg_fscore_best_val:.4f}')
    logging.info(f'>>>> Final best Dice: {dice_score_best_val:.4f}')

# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser(description='Yeast-Optimized cGAN-Seg Training')
    ap.add_argument("--train_set_dir", required=True, type=str, help="path for the train dataset")
    ap.add_argument("--lr", default=8e-5, type=float, help="learning rate (slightly reduced for stability)")
    ap.add_argument("--max_epoch", default=2500, type=int, help="maximum epoch to train model")
    ap.add_argument("--batch_size", default=4, type=int, help="train batch size")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and best checkpoint")
    ap.add_argument("--p_vanilla", default=0.15, type=float, help="probability value of vanilla augmentation (reduced for yeast)")
    ap.add_argument("--p_diff", default=0.15, type=float, help="probability value of diff augmentation (reduced for yeast)")
    ap.add_argument("--seg_model", required=True, type=str, help="segmentation model type (DeepSea or CellPose or UNET)")
    ap.add_argument("--patience", default=200, type=int, help="Number of patience epochs for early stopping (reduced)")
    ap.add_argument("--gen_nc", default=1, type=int, help="1 for 2D or 3 for 3D, the number of generator output channels")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir
    # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)

    train_yeast_optimized(args)
