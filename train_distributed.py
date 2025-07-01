# Import the necessary libraries and modules
import os
import argparse
import signal
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
import numpy as np
import random
import logging
from diffaug import DiffAugment
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, args, image_size=[512,768], image_means=[0.5], image_stds=[0.5], train_ratio=0.8):
    """Distributed training function"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set a constant seed for reproducibility
    SEED = 42 + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    # Only setup logging on rank 0
    if rank == 0:
        logging.basicConfig(filename=os.path.join(args.output_dir, 'train_distributed.log'), filemode='w',
                            format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'>>>> Multi-GPU training started with {world_size} GPUs')
        logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d per GPU' % (
            image_size[0], image_size[1], args.lr, args.batch_size))
        print(f'INFO: Starting distributed training on {world_size} GPUs...')

    # Determine the device
    device = torch.device(f'cuda:{rank}')

    # Define the transforms for the training and validation data
    train_transforms = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.RandomApply([transforms.RandomOrder([
                                   transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
                                   transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.5),
                                   transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),
                                   transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),
                                   transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),
                                   transforms.RandomApply([transforms.CLAHE()], p=0.5),
                                   transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
                                   transforms.RandomApply([transforms.RandomCrop()], p=0.5),
                                ])],p=args.p_vanilla),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = image_means,std = image_stds)
                           ])

    dev_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means,std=image_stds)
    ])

    # Read samples
    sample_pairs=get_image_mask_pairs(args.train_set_dir)
    assert len(sample_pairs)>0, f'No samples found in {args.train_set_dir}'
    
    # Split samples
    train_sample_pairs=sample_pairs[:int(train_ratio*len(sample_pairs))]
    valid_sample_pairs=sample_pairs[int(train_ratio * len(sample_pairs)):]

    # Define the datasets for training and validation
    train_data = BasicDataset(train_sample_pairs,transforms=train_transforms,vanilla_aug=False if args.p_vanilla==0 else True,gen_nc=args.gen_nc)
    valid_data = BasicDataset(valid_sample_pairs,transforms=dev_transforms,gen_nc=args.gen_nc)

    # Create distributed samplers
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank, shuffle=False)

    # Define the dataloaders with distributed samplers
    train_iterator = data.DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, 
                                   num_workers=4, pin_memory=True)
    valid_iterator = data.DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size, 
                                   num_workers=4, pin_memory=True)

    # Define the models
    Gen = StyleUnetGenerator(style_latent_dim = 128,output_nc=args.gen_nc)
    if args.seg_model=='UNET':
        Seg = UnetSegmentation(n_channels=args.gen_nc, n_classes=2)
    elif args.seg_model=='CellPose':
        Seg = Cellpose_CPnet(n_channels=args.gen_nc,n_classes=2)
    elif args.seg_model=='DeepSea':
        Seg = DeepSea(n_channels=args.gen_nc, n_classes=2)
    else:
        raise ValueError(f"Model '{args.seg_model}' not found.")

    D1 = NLayerDiscriminator(input_nc=args.gen_nc)
    D2 = NLayerDiscriminator(input_nc=1)

    # Initialize weights
    initialize_weights(Gen)
    initialize_weights(Seg)
    initialize_weights(D1)
    initialize_weights(D2)

    # Move models to device and wrap with DDP
    Gen = Gen.to(device)
    Seg = Seg.to(device)
    D1 = D1.to(device)
    D2 = D2.to(device)

    # Wrap models with DistributedDataParallel
    Gen = DDP(Gen, device_ids=[rank])
    Seg = DDP(Seg, device_ids=[rank])
    D1 = DDP(D1, device_ids=[rank])
    D2 = DDP(D2, device_ids=[rank])

    # Define the optimizers with scaled learning rate
    lr_scaled = args.lr * world_size  # Scale learning rate with number of GPUs
    optimizer_G = torch.optim.Adam(itertools.chain(Gen.parameters(), Seg.parameters()), lr=lr_scaled)
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=lr_scaled)
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=lr_scaled)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Define the loss functions
    d_criterion = nn.MSELoss().to(device)
    Gen_criterion_1 = nn.L1Loss().to(device)
    Gen_criterion_2 = VGGLoss().to(device)
    Seg_criterion = CombinedLoss().to(device)

    # Training loop
    nstop = 0
    avg_fscore_best = 0
    start_time = time.time()
    last_checkpoint_time = start_time
    
    if rank == 0:
        logging.info('>>>> Start distributed training')
        print('INFO: Start distributed training ...')

    for epoch in range(args.max_epoch):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        Gen.train()
        Seg.train()
        D1.train()
        D2.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(tqdm(train_iterator, disable=(rank != 0))):
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
                    style = mixed_list(real_img.shape[0], 5, Gen.module.latent_dim, device=device)
                else:
                    style = noise_list(real_img.shape[0], 5, Gen.module.latent_dim, device=device)

                im_noise = image_noise(real_mask.shape[0], image_size, device=device)

                fake_img = Gen(real_mask, style, im_noise)
                rec_mask = Seg(fake_img)
                fake_mask = Seg(real_img)
                fake_mask_p = F.softmax(fake_mask, dim=1).float()
                fake_mask_p = torch.unsqueeze(fake_mask_p.argmax(dim=1), dim=1)
                fake_mask_p = fake_mask_p.to(dtype=torch.float32)

                if random.random() < 0.9:
                    style = mixed_list(real_mask.shape[0], 4, Gen.module.latent_dim, device=device)
                else:
                    style = noise_list(real_mask.shape[0], 4, Gen.module.latent_dim, device=device)

                im_noise = image_noise(real_mask.shape[0], image_size, device=device)
                rec_img = Gen(fake_mask_p, style, im_noise)

                set_requires_grad(D1, False)
                set_requires_grad(D2, False)

                d_img_loss = d_criterion(D1(DiffAugment(fake_img, p=args.p_diff)), valid)
                d_mask_loss = d_criterion(D2(fake_mask_p), valid)
                rec_mask_loss = 100 * Seg_criterion(rec_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1))
                id_mask_loss = 50 * Seg_criterion(fake_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1))
                rec_img_loss = 50 * Gen_criterion_1(rec_img, real_img) + 100 * Gen_criterion_2(rec_img, real_img)
                id_img_loss = 25 * Gen_criterion_1(fake_img, real_img) + 50 * Gen_criterion_2(fake_img, real_img)
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

            total_d_loss += (d_mask_loss.item() + d_img_loss.item())
            total_g_loss += g_loss.item()
            num_batches += 1

        # Average losses across all GPUs
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches

        if rank == 0:
            print(f"[Epoch {epoch}/{args.max_epoch}] [D loss: {avg_d_loss:.6f}] [G loss: {avg_g_loss:.6f}]")

        # Evaluate only on rank 0
        if rank == 0:
            scores = evaluate_segmentation(Seg.module, valid_iterator, device, Seg_criterion, len(valid_data), 
                                         is_avg_prec=True, prec_thresholds=[0.5], output_dir=None)
            if scores['avg_fscore'] is not None:
                logging.info('>>>> Epoch:%d  , Dice score=%f , avg fscore=%f' % (epoch, scores['dice_score'], scores['avg_fscore']))
            else:
                logging.info('>>>> Epoch:%d  , Dice score=%f' % (epoch, scores['dice_score']))

            if scores['avg_fscore'] is not None and scores['avg_fscore'] > avg_fscore_best:
                avg_fscore_best = scores['avg_fscore']
                # Save best checkpoint (only on rank 0)
                torch.save(Gen.module.state_dict(), os.path.join(args.output_dir, 'Gen.pth'))
                torch.save(Seg.module.state_dict(), os.path.join(args.output_dir, 'Seg.pth'))
                torch.save(D1.module.state_dict(), os.path.join(args.output_dir, 'D1.pth'))
                torch.save(D2.module.state_dict(), os.path.join(args.output_dir, 'D2.pth'))
                logging.info('>>>> Save the model checkpoints to %s' % (os.path.join(args.output_dir)))
                nstop = 0
            elif scores['avg_fscore'] is not None and scores['avg_fscore'] <= avg_fscore_best:
                nstop += 1

            # Save periodic checkpoint every 4 hours (only on rank 0)
            current_time = time.time()
            if current_time - last_checkpoint_time >= 4 * 3600:
                print(f'INFO: Saving periodic checkpoint at epoch {epoch} (after {(current_time - start_time)/3600:.1f} hours)')
                torch.save(Gen.module.state_dict(), os.path.join(args.output_dir, f'Gen_epoch_{epoch}.pth'))
                torch.save(Seg.module.state_dict(), os.path.join(args.output_dir, f'Seg_epoch_{epoch}.pth'))
                torch.save(D1.module.state_dict(), os.path.join(args.output_dir, f'D1_epoch_{epoch}.pth'))
                torch.save(D2.module.state_dict(), os.path.join(args.output_dir, f'D2_epoch_{epoch}.pth'))
                last_checkpoint_time = current_time
                logging.info(f'>>>> Saved periodic checkpoint at epoch {epoch}')

            if nstop == args.patience:
                print('INFO: Early Stopping met ...')
                print('INFO: Finish training process')
                break

        # Synchronize all processes
        dist.barrier()

    cleanup_distributed()

def main():
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set_dir", required=True, type=str, help="path for the train dataset")
    ap.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    ap.add_argument("--max_epoch", default=2500, type=int, help="maximum epoch to train model")
    ap.add_argument("--batch_size", default=2, type=int, help="train batch size per GPU")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and best checkpoint")
    ap.add_argument("--p_vanilla", default=0.2, type=float, help="probability value of vanilla augmentation")
    ap.add_argument("--p_diff", default=0.2, type=float, help="probability value of differential augmentation")
    ap.add_argument("--patience", default=200, type=int, help="Number of epochs without improvement for early stopping")
    ap.add_argument("--gen_nc", default=3, type=int, help="number of channels of the generated images")
    ap.add_argument("--seg_model", default='UNET', type=str, help="type of the segmentation model")

    args = ap.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get world size and rank from environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    # Run distributed training
    train_distributed(local_rank, world_size, args)

if __name__ == "__main__":
    main()
