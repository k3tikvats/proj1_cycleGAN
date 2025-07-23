"""
Training script for CycleGAN SAR to EO conversion.
Handles the complete training loop with progressive learning and checkpointing.
"""

import torch
import torch.optim as optim
import numpy as np
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import os
import warnings

from config import *
from preprocess import create_dataloaders, setup_data_environment
from models import create_models
from losses import CycleGANLosses

warnings.filterwarnings('ignore')


def get_optimizers(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                  lr, betas, weight_decay, resnet_factor=1.0, critic_factor=0.5):
    """
    Create optimizers for all models with different learning rates for ResNet layers.
    
    Args:
        sar_to_eo: SAR to EO generator
        eo_to_sar: EO to SAR generator
        sar_critic: SAR critic
        eo_critic: EO critic
        lr: Base learning rate
        betas: Beta parameters for Adam optimizer
        weight_decay: Weight decay values for each model
        resnet_factor: Learning rate multiplier for ResNet layers
        critic_factor: Learning rate multiplier for critics
        
    Returns:
        Dictionary of optimizers
    """
    optimizers = {}
    lrs = [lr, lr * 0.8, lr * critic_factor * 0.8, lr * critic_factor]
    models = [('sar_to_eo', sar_to_eo), ('eo_to_sar', eo_to_sar), 
              ('sar_critic', sar_critic), ('eo_critic', eo_critic)]
    
    for i in range(len(lrs)):
        lr_current = lrs[i]
        name, model = models[i]
        
        # Separate ResNet parameters for different learning rates
        resnet_params = [param for pname, param in model.named_parameters() 
                        if 'resnet' in pname and param.requires_grad]
        resnet_ids = set(id(p) for p in resnet_params)
        other_params = [p for p in model.parameters() 
                       if id(p) not in resnet_ids and p.requires_grad]
        
        # Create parameter groups with different learning rates
        param_groups = []
        if resnet_params:
            param_groups.append({'params': resnet_params, 'lr': lr_current * resnet_factor})
        if other_params:
            param_groups.append({'params': other_params, 'lr': lr_current})
        
        optimizers[name] = optim.AdamW(
            param_groups, 
            weight_decay=weight_decay[i], 
            betas=betas, 
            fused=True, 
            eps=1e-8
        )
    
    return optimizers


def freeze_layers(model, layers=[]):
    """
    Freeze or unfreeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layers: List of layer name patterns to freeze
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers):
            param.requires_grad = False
        else:
            param.requires_grad = True


def get_progressive_optimizers(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                             epoch, betas=(0.5, 0.999), weight_decay=[1e-4, 1e-4, 0, 0]):
    """
    Progressive learning strategy with different freezing and learning rates per epoch.
    
    Args:
        sar_to_eo: SAR to EO generator
        eo_to_sar: EO to SAR generator
        sar_critic: SAR critic
        eo_critic: EO critic
        epoch: Current epoch
        betas: Beta parameters for optimizers
        weight_decay: Weight decay values
        
    Returns:
        Dictionary of optimizers for current epoch
    """
    if epoch in range(3):
        # Early epochs: freeze ResNet, low learning rate
        lr = 3e-4
        freeze_layers(sar_to_eo, ['resnet'])
        freeze_layers(eo_to_sar, ['resnet'])
        freeze_layers(sar_critic, ['resnet'])
        freeze_layers(eo_critic, ['resnet'])
        return get_optimizers(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                            lr, betas, weight_decay, resnet_factor=0.01, critic_factor=0.2)
        
    elif epoch in range(3, 6):
        # Mid-early epochs: unfreeze all, moderate learning rate
        lr = 1e-4
        freeze_layers(sar_to_eo)
        freeze_layers(eo_to_sar)
        freeze_layers(sar_critic)
        freeze_layers(eo_critic)
        return get_optimizers(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                            lr, betas, weight_decay, resnet_factor=0.05, critic_factor=0.2)
        
    elif epoch in range(6, 10):
        # Mid epochs: balanced learning
        lr = 5e-5
        return get_optimizers(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                            lr, betas, weight_decay, resnet_factor=0.1, critic_factor=0.4)
        
    else:
        # Late epochs: fine-tuning with low learning rate
        lr = 1e-5
        return get_optimizers(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                            lr, betas, weight_decay)


def save_checkpoint(sar_to_eo, eo_to_sar, sar_critic, eo_critic, epoch, 
                   checkpoint_dir=CHECKPOINT_DIR):
    """
    Save model checkpoints.
    
    Args:
        sar_to_eo: SAR to EO generator
        eo_to_sar: EO to SAR generator
        sar_critic: SAR critic
        eo_critic: EO critic
        epoch: Current epoch
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'sar_to_eo': sar_to_eo.state_dict(),
        'eo_to_sar': eo_to_sar.state_dict(),
        'sar_critic': sar_critic.state_dict(),
        'eo_critic': eo_critic.state_dict(),
        'epoch': epoch
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, sar_to_eo, eo_to_sar, sar_critic, eo_critic):
    """
    Load model checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint file
        sar_to_eo: SAR to EO generator
        eo_to_sar: EO to SAR generator
        sar_critic: SAR critic
        eo_critic: EO critic
        
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    sar_to_eo.load_state_dict(checkpoint['sar_to_eo'])
    eo_to_sar.load_state_dict(checkpoint['eo_to_sar'])
    sar_critic.load_state_dict(checkpoint['sar_critic'])
    eo_critic.load_state_dict(checkpoint['eo_critic'])
    
    starting_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Checkpoint loaded: {checkpoint_path}, resuming from epoch {starting_epoch}")
    
    return starting_epoch


def train_cyclegan(sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                  train_dataloader, val_dataloader=None,
                  crit_repeats=CRITIC_REPEATS, gen_repeats=GENERATOR_REPEATS, 
                  epochs=EPOCHS, device=DEVICE, starting_epoch=0,
                  checkpoint_path=None):
    """
    Main training loop for CycleGAN.
    
    Args:
        sar_to_eo: SAR to EO generator
        eo_to_sar: EO to SAR generator
        sar_critic: SAR critic
        eo_critic: EO critic
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        crit_repeats: Number of critic updates per batch
        gen_repeats: Number of generator updates per batch
        epochs: Total number of epochs
        device: Device for training
        starting_epoch: Starting epoch (for resuming training)
        checkpoint_path: Path to load checkpoint from
    """
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        starting_epoch = load_checkpoint(checkpoint_path, sar_to_eo, eo_to_sar, 
                                       sar_critic, eo_critic)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(device)
    
    # Training loop
    for epoch in range(starting_epoch, epochs):
        # Set models to training mode
        sar_to_eo.train()
        eo_to_sar.train()
        sar_critic.train()
        eo_critic.train()
        
        # Update optimizers based on progressive learning schedule
        if epoch in [0, 3, 6, 10, starting_epoch]:
            optimizers = get_progressive_optimizers(sar_to_eo, eo_to_sar, 
                                                  sar_critic, eo_critic, epoch)
            # Clear gradients
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            
            # Reduce critic repeats as training progresses
            crit_repeats = max(crit_repeats - 1, 1) if crit_repeats > 1 else 1
        
        # Initialize epoch statistics
        epoch_stats = {
            'sar_critic_loss': [], 'sar_adv': [], 'sar_cycle': [], 
            'eo_critic_loss': [], 'eo_adv': [], 'eo_cycle': [], 
            'perceptual': [], 'ssim': []
        }

        # Initialize loss calculator
        losses = CycleGANLosses(device=device)
        
        # Training loop for current epoch
        for batch_idx, (sar_batch, eo_batch) in enumerate(tqdm(train_dataloader, 
                                                             desc=f"Epoch: {epoch}/{epochs}")):
            # Move data to device
            sar_batch = sar_batch.to(device=device, memory_format=torch.channels_last)
            eo_batch = eo_batch.to(device=device, memory_format=torch.channels_last)
            
            # Train critics multiple times per batch
            for crit_step in range(crit_repeats):
                with autocast(device):
                    # Generate fake images (no gradients needed for generator updates)
                    with torch.no_grad():
                        fake_sar = eo_to_sar(eo_batch)
                        fake_eo = sar_to_eo(sar_batch)
                    
                    # Calculate critic losses
                    sar_loss = losses.critic_loss(sar_critic, sar_batch, fake_sar, 
                                                 losses.lambdas['gp'], device)
                    eo_loss = losses.critic_loss(eo_critic, eo_batch, fake_eo, 
                                               losses.lambdas['gp'], device)
                
                # Backward pass with gradient scaling
                scaler.scale(sar_loss).backward()
                scaler.scale(eo_loss).backward()

                # Unscale gradients for clipping
                scaler.unscale_(optimizers['sar_critic'])
                scaler.unscale_(optimizers['eo_critic'])

                # Handle NaN/Inf gradients
                for model in [sar_critic, eo_critic]:
                    for param in model.parameters():
                        if param.grad is not None:
                            mask = torch.isnan(param.grad.data) | torch.isinf(param.grad.data)
                            if mask.any():
                                param.grad.data[mask] = 0.0
                
                # Update critic optimizers
                scaler.step(optimizers['sar_critic'])
                scaler.step(optimizers['eo_critic'])
                scaler.update()

                # Clear gradients
                optimizers['sar_critic'].zero_grad(set_to_none=True)
                optimizers['eo_critic'].zero_grad(set_to_none=True)
                
                # Record losses
                epoch_stats['sar_critic_loss'].append(sar_loss.item())
                epoch_stats['eo_critic_loss'].append(eo_loss.item())
                
            # Train generators
            for gen_step in range(gen_repeats):
                with autocast(device):
                    # Calculate generator losses
                    g_loss, g_stats = losses.generator_loss_focused(
                        sar_to_eo, eo_to_sar, sar_critic, eo_critic,
                        sar_batch, eo_batch, device
                    )
                
                # Backward pass
                scaler.scale(g_loss).backward()
            
                # Unscale gradients
                scaler.unscale_(optimizers['sar_to_eo'])
                scaler.unscale_(optimizers['eo_to_sar'])
                
                # Handle NaN/Inf gradients
                for model in [sar_to_eo, eo_to_sar]:
                    for param in model.parameters():
                        if param.grad is not None:
                            mask = torch.isnan(param.grad.data) | torch.isinf(param.grad.data)
                            if mask.any():
                                param.grad.data[mask] = 0.0
                
                # Update generator optimizers
                scaler.step(optimizers['sar_to_eo'])
                scaler.step(optimizers['eo_to_sar'])
                scaler.update()
                
                # Clear gradients
                optimizers['sar_to_eo'].zero_grad(set_to_none=True)
                optimizers['eo_to_sar'].zero_grad(set_to_none=True)
                
                # Record generator losses
                for key, value in g_stats.items():
                    epoch_stats[key].append(value)
                
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: "
                      f"EO Adv: {g_stats['eo_adv']:.4f}, "
                      f"EO Cycle: {g_stats['eo_cycle']:.4f}, "
                      f"EO Critic: {eo_loss.item():.4f}, "
                      f"SAR Adv: {g_stats['sar_adv']:.4f}, "
                      f"SAR Cycle: {g_stats['sar_cycle']:.4f}, "
                      f"SAR Critic: {sar_loss.item():.4f}, "
                      f"Perceptual: {g_stats['perc']:.4f}, "
                      f"SSIM: {g_stats['ssim']:.4f}")

        # Save checkpoint every epoch
        save_checkpoint(sar_to_eo, eo_to_sar, sar_critic, eo_critic, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  EO Critic Loss: {np.mean(epoch_stats['eo_critic_loss']):.4f}")
        print(f"  EO Adversarial Loss: {np.mean(epoch_stats['eo_adv']):.4f}")
        print(f"  EO Cycle Consistency Loss: {np.mean(epoch_stats['eo_cycle']):.4f}")
        print(f"  SAR Critic Loss: {np.mean(epoch_stats['sar_critic_loss']):.4f}")
        print(f"  SAR Adversarial Loss: {np.mean(epoch_stats['sar_adv']):.4f}")
        print(f"  SAR Cycle Consistency Loss: {np.mean(epoch_stats['sar_cycle']):.4f}")
        print(f"  Perceptual Loss: {np.mean(epoch_stats['perceptual']):.4f}")
        print(f"  SSIM: {np.mean(epoch_stats['ssim']):.4f}")
        print("-" * 50)


def main():
    """Main training function."""
    print("Setting up training environment...")
    
    # Setup environment
    setup_data_environment()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(DATA_PATH)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create models
    print("Creating models...")
    sar_to_eo, eo_to_sar, sar_critic, eo_critic = create_models()
    
    # Start training
    print("Starting training...")
    train_cyclegan(
        sar_to_eo=sar_to_eo,
        eo_to_sar=eo_to_sar, 
        sar_critic=sar_critic,
        eo_critic=eo_critic, 
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        crit_repeats=CRITIC_REPEATS,
        gen_repeats=GENERATOR_REPEATS,
        epochs=EPOCHS,
        device=DEVICE,
        starting_epoch=0
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
