"""
Training script using configuration file.
This script demonstrates how to use the config.py file for easy parameter management.
"""

import torch
from torch.utils.data import DataLoader, random_split
import itertools

from preprocess import collect_data_paths, SARToEODataset
from train_cyclegan import (
    ResnetGenerator, NLayerDiscriminator, 
    CycleGANTrainer, ssim_metric
)
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG, DEVICE_CONFIG


def setup_device():
    """Setup device based on configuration."""
    if DEVICE_CONFIG['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = DEVICE_CONFIG['device']
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def setup_data():
    """Setup dataset and dataloaders."""
    print("Setting up data...")
    
    # Collect data paths
    sar_paths, eo_paths = collect_data_paths(
        DATA_CONFIG['sar_dir'], 
        DATA_CONFIG['eo_dir'], 
        max_samples=DATA_CONFIG['max_samples']
    )
    
    # Create dataset
    dataset = SARToEODataset(
        sar_paths, eo_paths,
        patch_size=DATA_CONFIG['patch_size'],
        output_mode=DATA_CONFIG['output_mode']
    )
    
    # Split dataset
    train_size = int(TRAIN_CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=TRAIN_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False
    )
    
    return train_loader, val_loader


def setup_models(device):
    """Setup CycleGAN models."""
    print("Setting up models...")
    
    # Generators
    G_AB = ResnetGenerator(
        input_nc=MODEL_CONFIG['input_nc'],
        output_nc=MODEL_CONFIG['output_nc'],
        ngf=MODEL_CONFIG['ngf'],
        n_blocks=MODEL_CONFIG['n_blocks']
    )
    
    G_BA = ResnetGenerator(
        input_nc=MODEL_CONFIG['output_nc'],
        output_nc=MODEL_CONFIG['input_nc'],
        ngf=MODEL_CONFIG['ngf'],
        n_blocks=MODEL_CONFIG['n_blocks']
    )
    
    # Discriminators
    D_A = NLayerDiscriminator(
        input_nc=MODEL_CONFIG['input_nc'],
        ndf=MODEL_CONFIG['ndf'],
        n_layers=MODEL_CONFIG['n_layers']
    )
    
    D_B = NLayerDiscriminator(
        input_nc=MODEL_CONFIG['output_nc'],
        ndf=MODEL_CONFIG['ndf'],
        n_layers=MODEL_CONFIG['n_layers']
    )
    
    return G_AB, G_BA, D_A, D_B


def setup_optimizers(G_AB, G_BA, D_A, D_B):
    """Setup optimizers."""
    print("Setting up optimizers...")
    
    # Generator parameters
    gen_params = itertools.chain(G_AB.parameters(), G_BA.parameters())
    
    # Discriminator parameters
    disc_params = itertools.chain(D_A.parameters(), D_B.parameters())
    
    # Optimizers
    optimizer_G = torch.optim.AdamW(
        gen_params,
        lr=TRAIN_CONFIG['learning_rate'],
        betas=TRAIN_CONFIG['betas'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    optimizer_D = torch.optim.AdamW(
        disc_params,
        lr=TRAIN_CONFIG['learning_rate'],
        betas=TRAIN_CONFIG['betas'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    return optimizer_G, optimizer_D


def main():
    """Main training function."""
    print("Starting SAR to EO CycleGAN Training")
    print("=" * 50)
    
    # Setup
    device = setup_device()
    train_loader, val_loader = setup_data()
    G_AB, G_BA, D_A, D_B = setup_models(device)
    optimizer_G, optimizer_D = setup_optimizers(G_AB, G_BA, D_A, D_B)
    
    # Initialize trainer
    trainer = CycleGANTrainer(
        G_AB, G_BA, D_A, D_B,
        dataloaders=(train_loader, val_loader),
        optimizers=(optimizer_G, optimizer_D),
        pool_size=TRAIN_CONFIG['pool_size'],
        device=device,
        output_dir=OUTPUT_CONFIG['output_dir'],
        img_save_epoch=OUTPUT_CONFIG['img_save_epoch']
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 50)
    
    loss_history = trainer.train(
        n_epochs=TRAIN_CONFIG['n_epochs'],
        metrics_fn=ssim_metric
    )
    
    print("\nTraining completed!")
    print("=" * 50)
    print(f"Results saved to: {OUTPUT_CONFIG['output_dir']}")
    
    return loss_history


if __name__ == "__main__":
    main()
