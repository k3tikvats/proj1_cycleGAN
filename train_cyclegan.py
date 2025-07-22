import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import itertools
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess import SARToEODataset, ImagePool, collect_data_paths, show_tensor_image


class ResnetBlock(nn.Module):
    """Residual block for ResNet generator."""
    
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d):
        """
        Initialize ResNet block.
        
        Args:
            dim (int): Number of input/output channels
            norm_layer: Normalization layer to use
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN."""
    
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        """
        Initialize ResNet generator.
        
        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            ngf (int): Number of generator filters in first conv layer
            n_blocks (int): Number of ResNet blocks
            norm_layer: Normalization layer to use
        """
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        # Residual blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(inplace=True)
            ]

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """N-layer discriminator for CycleGAN."""
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """
        Initialize N-layer discriminator.
        
        Args:
            input_nc (int): Number of input channels
            ndf (int): Number of discriminator filters in first conv layer
            n_layers (int): Number of conv layers in discriminator
            norm_layer: Normalization layer to use
        """
        super().__init__()
        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class CycleGANTrainer:
    """Trainer class for CycleGAN model."""
    
    def __init__(self, G_AB, G_BA, D_A, D_B,
                 dataloaders, optimizers,
                 pool_size=50, device='cuda',
                 output_dir='./outputs', img_save_epoch=5):
        """
        Initialize CycleGAN trainer.
        
        Args:
            G_AB: Generator A to B
            G_BA: Generator B to A
            D_A: Discriminator for domain A
            D_B: Discriminator for domain B
            dataloaders: Tuple of (train_loader, val_loader)
            optimizers: Tuple of (optimizer_G, optimizer_D)
            pool_size (int): Size of image buffer
            device (str): Device to use for training
            output_dir (str): Directory to save outputs
            img_save_epoch (int): Frequency of saving sample images
        """
        self.G_AB, self.G_BA = G_AB, G_BA
        self.D_A, self.D_B = D_A, D_B
        self.train_loader, self.val_loader = dataloaders
        self.opt_G, self.opt_D = optimizers
        self.device = device
        self.best_ssim = -float('inf')
        self.output_dir = output_dir
        self.img_save_epoch = img_save_epoch
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)

        self.loss_history = {'G': [], 'D': [], 'cycle': [], 'ssim': []}
        self.G_AB = self.G_AB.to(self.device)
        self.G_BA = self.G_BA.to(self.device)
        self.D_A = self.D_A.to(self.device)
        self.D_B = self.D_B.to(self.device)

    def save_sample_images(self, epoch, num_samples=3):
        """Save sample images during training."""
        self.G_AB.eval()
        with torch.no_grad():
            val_iter = iter(self.val_loader)
            for i in range(num_samples):
                try:
                    real_A, real_B = next(val_iter)
                except StopIteration:
                    break
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                fake_B = self.G_AB(real_A)

                plt.figure(figsize=(12, 4))
                # SAR input (assume 2 or 3 channels: VV, VH, VV/VH)
                plt.subplot(1, 3, 1)
                show_tensor_image(real_A[0], 'Input SAR', cmap='gray')

                # Real EO (assume first 3 bands are RGB)
                plt.subplot(1, 3, 2)
                show_tensor_image(real_B[0], 'Real EO', bands=[0, 1, 2])

                # Fake EO
                plt.subplot(1, 3, 3)
                show_tensor_image(fake_B[0], 'Generated EO', bands=[0, 1, 2])

                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/images/epoch_{epoch}_sample_{i}.png")
                plt.close()

    def train(self, n_epochs, metrics_fn):
        """
        Train the CycleGAN model.
        
        Args:
            n_epochs (int): Number of epochs to train
            metrics_fn: Function to compute validation metrics
            
        Returns:
            dict: Training loss history
        """
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
        
        for epoch in range(1, n_epochs+1):
            epoch_losses = {'G': 0., 'D': 0., 'cycle': 0., 'ssim': 0.}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{n_epochs}")
            
            for real_A, real_B in pbar:
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                # ------------------
                #  Train Generators
                # ------------------
                self.opt_G.zero_grad()
                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)

                rec_A = self.G_BA(fake_B)
                rec_B = self.G_AB(fake_A)

                # GAN losses
                loss_GAN_AB = F.mse_loss(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
                loss_GAN_BA = F.mse_loss(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))

                # Cycle consistency
                loss_cycle = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)

                # Combined generator loss (no identity term)
                loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * loss_cycle
                loss_G.backward()
                self.opt_G.step()

                # -----------------------
                #  Train Discriminators
                # -----------------------
                self.opt_D.zero_grad()
                fake_B_ = self.fake_B_pool.query(fake_B.detach())
                fake_A_ = self.fake_A_pool.query(fake_A.detach())

                loss_D_B = (F.mse_loss(self.D_B(real_B), torch.ones_like(self.D_B(real_B))) +
                                  F.mse_loss(self.D_B(fake_B_), torch.zeros_like(self.D_B(fake_B_))))
                loss_D_A = (F.mse_loss(self.D_A(real_A), torch.ones_like(self.D_A(real_A))) +
                                  F.mse_loss(self.D_A(fake_A_), torch.zeros_like(self.D_A(fake_A_))))
                loss_D = loss_D_A + loss_D_B
                loss_D.backward()
                self.opt_D.step()

                # Logging
                epoch_losses['G'] += loss_G.item()
                epoch_losses['D'] += loss_D.item()
                epoch_losses['cycle'] += loss_cycle.item()
                pbar.set_postfix(G=loss_G.item(), D=loss_D.item())

            # Average losses
            for k in ['G', 'D', 'cycle']:
                epoch_losses[k] /= len(self.train_loader)

            # Validation: compute SSIM on a small batch
            val_real_A, val_real_B = next(iter(self.val_loader))
            val_real_A, val_real_B = val_real_A.to(self.device), val_real_B.to(self.device)
            val_fake_B = self.G_AB(val_real_A)
            ssim_val = metrics_fn(val_fake_B, val_real_B)
            epoch_losses['ssim'] = ssim_val

            # Save best checkpoint
            if ssim_val > self.best_ssim:
                self.best_ssim = ssim_val
                torch.save({
                    'epoch': epoch,
                    'G_AB': self.G_AB.state_dict(),
                    'G_BA': self.G_BA.state_dict(),
                    'D_A': self.D_A.state_dict(),
                    'D_B': self.D_B.state_dict(),
                    'opt_G': self.opt_G.state_dict(),
                    'opt_D': self.opt_D.state_dict(),
                    'best_ssim': self.best_ssim
                }, f"{self.output_dir}/checkpoints/best.pth")

            # Save epoch checkpoint
            torch.save({
                'epoch': epoch,
                'G_AB': self.G_AB.state_dict(),
                'G_BA': self.G_BA.state_dict(),
                'D_A': self.D_A.state_dict(),
                'D_B': self.D_B.state_dict(),
                'opt_G': self.opt_G.state_dict(),
                'opt_D': self.opt_D.state_dict(),
                'best_ssim': self.best_ssim
            }, f"{self.output_dir}/checkpoints/epoch_{epoch}.pth")

            # Save sample images periodically
            if epoch % self.img_save_epoch == 0:
                self.save_sample_images(epoch, num_samples=3)

            # Record loss history
            for k in ['G', 'D', 'cycle', 'ssim']:
                self.loss_history[k].append(epoch_losses[k])

            print(f"Epoch {epoch} | SSIM: {ssim_val:.4f}")

        return self.loss_history


def ssim_metric(pred, target):
    """
    Calculate SSIM metric between predicted and target images.
    
    Args:
        pred: Predicted images tensor
        target: Target images tensor
        
    Returns:
        float: SSIM value
    """
    try:
        import piq
        # pred and target are in [–1,1] range; data_range=2.0 covers that span
        p = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
        t = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        return piq.ssim(p, t, data_range=1.0, reduction='mean').item()
    except ImportError:
        print("Warning: piq library not available, using L1 loss as proxy metric")
        return -F.l1_loss(pred, target).item()


def main():
    """Main training function."""
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data paths (modify these according to your data location)
    sar_dir = '/kaggle/input/sar-images/ROIs2017_winter_s1/ROIs2017_winter'
    eo_dir = '/kaggle/input/sar-images/ROIs2017_winter_s2/ROIs2017_winter'
    
    # Collect data paths
    sar_paths, eo_paths = collect_data_paths(sar_dir, eo_dir, max_samples=5000)
    
    # Create dataset and dataloaders
    dataset = SARToEODataset(sar_paths, eo_paths, output_mode='RGB')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # Initialize models
    G_sar2rgb = ResnetGenerator(input_nc=3, output_nc=3)
    G_rgb2sar = ResnetGenerator(input_nc=3, output_nc=3)
    D_sar = NLayerDiscriminator(input_nc=3)
    D_rgb = NLayerDiscriminator(input_nc=3)
    
    # Initialize optimizers
    gen_params = itertools.chain(G_sar2rgb.parameters(), G_rgb2sar.parameters())
    disc_params = itertools.chain(D_sar.parameters(), D_rgb.parameters())
    
    optimizer_G = torch.optim.AdamW(gen_params, lr=2e-4,
                                    betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = torch.optim.AdamW(disc_params, lr=2e-4,
                                    betas=(0.5, 0.999), weight_decay=1e-4)
    
    # Initialize trainer
    trainer = CycleGANTrainer(
        G_sar2rgb, G_rgb2sar, D_sar, D_rgb,
        dataloaders=(train_loader, val_loader),
        optimizers=(optimizer_G, optimizer_D),
        pool_size=50,
        device=device,
        output_dir='./runs/exp1',
        img_save_epoch=5
    )
    
    # Train the model
    print("Starting training...")
    loss_history = trainer.train(15, ssim_metric)
    print("Training completed!")
    
    return loss_history


if __name__ == "__main__":
    main()