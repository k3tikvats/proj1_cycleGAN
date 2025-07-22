"""
Configuration file for SAR to EO CycleGAN training.
Modify these parameters according to your setup and requirements.
"""

# Data Configuration
DATA_CONFIG = {
    'sar_dir': '/kaggle/input/sar-images/ROIs2017_winter_s1/ROIs2017_winter',
    'eo_dir': '/kaggle/input/sar-images/ROIs2017_winter_s2/ROIs2017_winter',
    'max_samples': 5000,  # Set to None to use all available data
    'patch_size': 256,
    'output_mode': 'RGB',  # Options: 'RGB', 'NIR_SWIR', 'RGB_NIR'
}

# Model Configuration
MODEL_CONFIG = {
    'input_nc': 3,          # Number of input channels
    'output_nc': 3,         # Number of output channels
    'ngf': 64,              # Generator filters
    'ndf': 64,              # Discriminator filters
    'n_blocks': 9,          # Number of ResNet blocks
    'n_layers': 3,          # Number of discriminator layers
}

# Training Configuration
TRAIN_CONFIG = {
    'n_epochs': 15,
    'batch_size': 8,
    'learning_rate': 2e-4,
    'betas': (0.5, 0.999),
    'weight_decay': 1e-4,
    'cycle_loss_weight': 10.0,
    'pool_size': 50,
    'num_workers': 2,
    'train_split': 0.8,
}

# Output Configuration
OUTPUT_CONFIG = {
    'output_dir': './runs/exp1',
    'img_save_epoch': 5,
    'checkpoint_epoch': 1,
}

# Device Configuration
DEVICE_CONFIG = {
    'device': 'auto',  # 'auto', 'cuda', or 'cpu'
    'mixed_precision': False,  # Enable for faster training (experimental)
}
