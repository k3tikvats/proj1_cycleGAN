"""
CycleGAN model architectures for SAR to EO conversion.
Contains Generator and Critic (Discriminator) networks with ResNet backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init
import torch.nn.utils.spectral_norm as spectral_norm

from config import *


class UpProjection(nn.Module):
    """
    Custom upsampling block using multiple convolutions and pixel shuffling.
    Provides stable upsampling for generator networks.
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
        # Multiple convolution paths for robust upsampling
        self.a = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1))
        self.b = nn.Conv2d(in_channels, out_channels, (2, 3), padding=(0, 1))  
        self.c = nn.Conv2d(in_channels, out_channels, (3, 2), padding=(1, 0))
        self.d = nn.Conv2d(in_channels, out_channels, (2, 2), padding=(0, 0))
        
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.residual_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.final_conv = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Apply different convolution patterns
        a = self.a(x)
        b = self.b(x)
        c = self.c(x)
        d = self.d(x)
        
        # Pad to ensure consistent dimensions
        b = F.pad(b, (0, 0, 0, 1), mode='reflect')
        c = F.pad(c, (0, 1, 0, 0), mode='reflect')  
        d = F.pad(d, (0, 1, 0, 1), mode='reflect')
        
        # Combine into upsampled output
        out = torch.zeros(B, a.size(1), H*2, W*2, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b  
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d
        
        # Add residual connection
        residual = self.residual_conv(self.lrelu(out))
        out = torch.cat([residual, out], dim=1)
        out = self.dropout(out)
        out = self.final_conv(self.lrelu(out))
        
        return self.norm(out)


class ResidualBlock(nn.Module):
    """
    Residual block with instance normalization for stable GAN training.
    """
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.lrelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual


class Generator(nn.Module):
    """
    U-Net style generator with ResNet-18 encoder and custom decoder.
    Converts SAR images to EO images or vice versa.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # --- ENCODER (ResNet-18 backbone) ---
        self.resnet = models.resnet18(weights='DEFAULT')
        # Replace BatchNorm with InstanceNorm for GAN stability
        self._replace_norm(self.resnet)
        
        # Initial convolution to handle variable input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet.bn1,
            self.resnet.relu
        )
        
        # Encoder layers from ResNet
        self.encoder_layer1 = self.resnet.layer1  # 64 channels
        self.encoder_layer2 = self.resnet.layer2  # 128 channels
        self.encoder_layer3 = self.resnet.layer3  # 256 channels
        self.bottleneck = self.resnet.layer4      # 512 channels

        # --- DECODER (Custom architecture with skip connections) ---
        self.up_block1 = UpProjection(512, 256, 0.4)
        self.res_block1 = ResidualBlock(256 + 256)  # After skip connection
        
        self.up_block2 = UpProjection(256 + 256, 128)
        self.res_block2 = ResidualBlock(128 + 128)
        
        self.up_block3 = UpProjection(128 + 128, 64, 0.2)
        self.res_block3 = ResidualBlock(64 + 64)
        
        self.up_block4 = UpProjection(64 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def _replace_norm(self, model):
        """Replace BatchNorm with InstanceNorm for improved GAN stability."""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, nn.InstanceNorm2d(module.num_features, affine=True))
            else:
                self._replace_norm(module)
                
    def forward(self, x):
        # Encoder path
        e1 = self.initial_conv(x)
        e2 = self.encoder_layer1(e1)
        e3 = self.encoder_layer2(e2)
        e4 = self.encoder_layer3(e3)
        b = self.bottleneck(e4)
        
        # Decoder path with skip connections
        d1 = self.up_block1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.res_block1(d1)
        
        d2 = self.up_block2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.res_block2(d2)

        d3 = self.up_block3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.res_block3(d3)
        
        d4 = self.up_block4(d3)
        
        return self.final_conv(d4)


class Critic(nn.Module):
    """
    PatchGAN critic/discriminator with ResNet backbone.
    Determines if input images are real or generated.
    
    Args:
        in_channels: Number of input channels
    """
    
    def __init__(self, in_channels):
        super().__init__()
        
        # Use ResNet-18 as backbone
        self.resnet = models.resnet18(weights='DEFAULT')
        self._replace_norm(self.resnet)
        
        # Initial layers
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_norm = self.resnet.bn1
        self.initial_relu = self.resnet.relu
        
        # Main body with dropout for regularization
        self.body = nn.Sequential(
            self.resnet.layer1,
            nn.Dropout(0.4),
            self.resnet.layer2,
            self.resnet.layer3,
            nn.Dropout(0.2),
            self.resnet.layer4
        )
        
        # Final convolutional layer for patch-based output
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
    
    def _replace_norm(self, model):
        """Replace BatchNorm with InstanceNorm for improved GAN stability."""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, nn.InstanceNorm2d(module.num_features, affine=True))
            else:
                self._replace_norm(module)
                
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = self.initial_relu(x)
        x = self.body(x)
        return self.final_conv(x)


def initialize_cyclegan_models(sar_to_eo_gen, eo_to_sar_gen, sar_critic, eo_critic):
    """
    Initialize CycleGAN model weights for stable training.
    
    Args:
        sar_to_eo_gen: SAR to EO generator
        eo_to_sar_gen: EO to SAR generator
        sar_critic: SAR critic/discriminator
        eo_critic: EO critic/discriminator
        
    Returns:
        Tuple of initialized models
    """
    
    def init_weights(model):
        """Initialize weights using Kaiming normal initialization."""
        for name, module in model.named_modules():
            # Skip ResNet pretrained weights
            if 'resnet' in name:
                continue
                
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
    
    # Initialize all models
    init_weights(sar_to_eo_gen)
    init_weights(eo_to_sar_gen)
    init_weights(sar_critic)
    init_weights(eo_critic)

    def apply_spectral_norm(model):
        """Apply spectral normalization to critic networks for training stability."""
        for name, child in model.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                if not hasattr(child, 'weight_orig'):
                    setattr(model, name, spectral_norm(child))
            else:
                apply_spectral_norm(child)
        return model

    # Apply spectral normalization to critics
    sar_critic = apply_spectral_norm(sar_critic)
    eo_critic = apply_spectral_norm(eo_critic)
    
    print("Model initialization complete!")
    
    return sar_to_eo_gen, eo_to_sar_gen, sar_critic, eo_critic


def create_models():
    """
    Create and initialize all CycleGAN models.
    
    Returns:
        Tuple of (sar_to_eo_gen, eo_to_sar_gen, sar_critic, eo_critic)
    """
    
    # Create generators
    sar_to_eo_gen = Generator(SAR_CHANNELS, EO_CHANNELS).to(
        device=DEVICE, memory_format=torch.channels_last
    )
    eo_to_sar_gen = Generator(EO_CHANNELS, SAR_CHANNELS).to(
        device=DEVICE, memory_format=torch.channels_last
    )
    
    # Create critics
    sar_critic = Critic(SAR_CHANNELS).to(
        device=DEVICE, memory_format=torch.channels_last
    )
    eo_critic = Critic(EO_CHANNELS).to(
        device=DEVICE, memory_format=torch.channels_last
    )
    
    # Initialize models
    sar_to_eo_gen, eo_to_sar_gen, sar_critic, eo_critic = initialize_cyclegan_models(
        sar_to_eo_gen, eo_to_sar_gen, sar_critic, eo_critic
    )
    
    return sar_to_eo_gen, eo_to_sar_gen, sar_critic, eo_critic


if __name__ == "__main__":
    # Test model creation
    models = create_models()
    print("Models created successfully!")
    
    # Print model sizes
    for i, (name, model) in enumerate([
        ("SAR to EO Generator", models[0]),
        ("EO to SAR Generator", models[1]),
        ("SAR Critic", models[2]),
        ("EO Critic", models[3])
    ]):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {total_params:,} total parameters, {trainable_params:,} trainable")
