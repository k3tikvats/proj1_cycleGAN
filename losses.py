"""
Loss functions for CycleGAN training.
Includes adversarial losses, cycle consistency losses, perceptual losses, and SSIM.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from config import *


def critic_wgan_loss(real_logits, fake_logits):
    """
    Wasserstein GAN loss for critic/discriminator.
    
    Args:
        real_logits: Critic output for real images
        fake_logits: Critic output for fake images
        
    Returns:
        WGAN critic loss
    """
    return (fake_logits - real_logits).mean()


def generator_wgan_loss(fake_logits):
    """
    Wasserstein GAN loss for generator.
    
    Args:
        fake_logits: Critic output for fake images
        
    Returns:
        WGAN generator loss
    """
    return -fake_logits.mean()


def gradient_penalty(critic, real_img, fake_img, device=DEVICE):
    """
    Gradient penalty for WGAN-GP training.
    
    Args:
        critic: Critic network
        real_img: Real images
        fake_img: Generated images
        device: Device to run computation on
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_img.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake images
    interp = alpha * real_img + (1 - alpha) * fake_img
    interp.requires_grad_(True)
    
    # Get critic output for interpolated images
    interp_logits = critic(interp)
    
    # Calculate gradients
    grads = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_logits),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Calculate gradient penalty
    grad_norm = grads.reshape(batch_size, -1).norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    
    return penalty


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features for better image quality.
    Compares high-level features between generated and target images.
    """
    
    def __init__(self, device=DEVICE):
        super().__init__()

        # Load pretrained VGG19 and extract features
        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Use up to conv4_4 layer (index 27)
        self.feature_extractor = nn.Sequential(*vgg[:27])
        self.device = device
        self.criterion = nn.MSELoss()

    def preprocess(self, x):
        """
        Preprocess images for VGG input.
        Convert from [-1, 1] to [0, 1] and apply ImageNet normalization.
        """
        x = (x + 1) / 2.0  # Convert to [0, 1]
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        return (x - mean) / std
        
    def forward(self, preds, targets):
        """
        Calculate perceptual loss between predictions and targets.
        Uses RGB and NIR-SWIR-RE combinations for multispectral images.
        """
        losses = []
        
        # Define band combinations for perceptual loss
        indices = [RGB_INDICES, [3, 7, 11]]  # RGB and NIR-SWIR-RE

        for i in indices:
            pred = preds[:, i]
            target = targets[:, i]
            
            # Preprocess for VGG
            pred_processed = self.preprocess(pred)
            target_processed = self.preprocess(target)
    
            # Extract features and calculate loss
            pred_features = self.feature_extractor(pred_processed)
            target_features = self.feature_extractor(target_processed)
            
            losses.append(self.criterion(pred_features, target_features))
        
        return torch.mean(torch.tensor(losses).to(self.device))


def ssim_loss(X, Y, sensor_type):
    """
    Structural Similarity Index loss for image quality assessment.
    
    Args:
        X: First image tensor
        Y: Second image tensor
        sensor_type: 's1' for SAR or 's2' for EO
        
    Returns:
        List of SSIM values for different band combinations
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    ssim_values = []
    
    if sensor_type == 's2':
        # For EO images, use RGB and NIR-SWIR-RE combinations
        indices = [RGB_INDICES, NIR_SWIR_RE_INDICES]
        for i in indices:
            x = X[:, i]
            y = Y[:, i]
            # Convert from [-1, 1] to [0, 1] for SSIM calculation
            x = (x + 1.0) / 2.0
            y = (y + 1.0) / 2.0
            ssim_values.append(ssim(x, y))
    else:
        # For SAR images, use all channels
        X = (X + 1.0) / 2.0
        Y = (Y + 1.0) / 2.0
        ssim_values.append(ssim(X, Y))
        
    return ssim_values


class CycleGANLosses:
    """
    Combined loss functions for CycleGAN training.
    Manages all loss components with configurable weights.
    """
    
    def __init__(self,
                 lambda_gp=LAMBDA_GP,
                 lambda_perc=LAMBDA_PERCEPTUAL,
                 lambda_ssim=LAMBDA_SSIM,
                 lambda_eo_adv=LAMBDA_EO_ADV,
                 lambda_eo_cycle=LAMBDA_EO_CYCLE,
                 lambda_sar_adv=LAMBDA_SAR_ADV,
                 lambda_sar_cycle=LAMBDA_SAR_CYCLE,
                 device=DEVICE):
        
        self.perc_loss = PerceptualLoss(device)
        self.lambdas = {
            'gp': lambda_gp,
            'perc': lambda_perc,
            'ssim': lambda_ssim,
            'eo_adv': lambda_eo_adv,
            'eo_cycle': lambda_eo_cycle,
            'sar_adv': lambda_sar_adv,
            'sar_cycle': lambda_sar_cycle
        }
        self.device = device
        self.l1loss = nn.L1Loss()

    def generator_loss_focused(self, sar_to_eo, eo_to_sar, sar_critic, eo_critic, 
                             real_sar, real_eo, device=DEVICE):
        """
        Calculate combined generator loss including adversarial, cycle consistency,
        perceptual, and SSIM losses.
        
        Args:
            sar_to_eo: SAR to EO generator
            eo_to_sar: EO to SAR generator
            sar_critic: SAR critic
            eo_critic: EO critic
            real_sar: Real SAR images
            real_eo: Real EO images
            device: Device for computation
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Generate fake images
        fake_eo = sar_to_eo(real_sar)
        fake_sar = eo_to_sar(real_eo)

        # Adversarial losses
        adv_sar = generator_wgan_loss(sar_critic(fake_sar))
        adv_eo = generator_wgan_loss(eo_critic(fake_eo))

        # Cycle consistency losses
        rec_eo = sar_to_eo(fake_sar)
        rec_sar = eo_to_sar(fake_eo)
        
        cycle_eo = self.l1loss(rec_eo, real_eo)
        cycle_sar = self.l1loss(rec_sar, real_sar)

        # Perceptual loss (only for EO images)
        perc = self.perc_loss(fake_eo, real_eo)
        
        # SSIM losses
        ssim_l = ssim_loss(fake_eo, real_eo, 's2')
        ssim_l_sar = ssim_loss(fake_sar, real_sar, 's1')

        # Combine all losses
        total = (
            self.lambdas['eo_adv'] * adv_eo +         
            self.lambdas['eo_cycle'] * cycle_eo + 
            self.lambdas['sar_cycle'] * cycle_sar +
            self.lambdas['sar_adv'] * adv_sar +     
            self.lambdas['perc'] * perc +
            self.lambdas['ssim'] * 0.2 * (1 - torch.mean(torch.tensor(ssim_l_sar).to(device))) + 
            self.lambdas['ssim'] * (1 - torch.mean(torch.tensor(ssim_l).to(device)))
        )
        
        # Return loss components for monitoring
        loss_dict = {
            'eo_adv': adv_eo.item(), 
            'eo_cycle': cycle_eo.item(), 
            'perc': perc.item(), 
            'ssim': ssim_l[0].item(),
            'sar_adv': adv_sar.item(),
            'sar_cycle': cycle_sar.item()
        }
        
        return total, loss_dict

    def critic_loss(self, critic, real, fake, lambda_gp=LAMBDA_GP, device=DEVICE):
        """
        Calculate critic/discriminator loss.
        
        Args:
            critic: Critic network
            real: Real images
            fake: Generated images
            lambda_gp: Gradient penalty weight
            device: Device for computation
            
        Returns:
            Critic loss
        """
        real_logits = critic(real)
        fake_logits = critic(fake.detach())
        adv = critic_wgan_loss(real_logits, fake_logits)
        
        # Note: Gradient penalty disabled in current configuration
        # Uncomment the following lines to enable gradient penalty
        # gp = gradient_penalty(critic, real, fake, device) * lambda_gp
        # return adv + gp
        
        return adv


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 2
    sar_data = torch.randn(batch_size, SAR_CHANNELS, 256, 256).to(DEVICE)
    eo_data = torch.randn(batch_size, EO_CHANNELS, 256, 256).to(DEVICE)
    
    # Test perceptual loss
    perc_loss = PerceptualLoss()
    loss = perc_loss(eo_data, eo_data)
    print(f"Perceptual loss (self): {loss.item():.4f}")
    
    # Test SSIM
    ssim_vals = ssim_loss(eo_data, eo_data, 's2')
    print(f"SSIM values: {[s.item() for s in ssim_vals]}")
    
    print("Loss functions working correctly!")
