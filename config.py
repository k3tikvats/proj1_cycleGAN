"""
Configuration file for SAR to EO CycleGAN project.
Contains all hyperparameters, paths, and model configurations.
"""

import torch
import os

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data paths (relative to project root)
DATA_PATH = "/kaggle/input/sar-images"  # Update this path as needed
CHECKPOINT_DIR = "./checkpoints"
GENERATED_SAMPLES_DIR = "./generated_samples"

# Training hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE = 1e-4

# Model hyperparameters
CRITIC_REPEATS = 4
GENERATOR_REPEATS = 1

# Loss function weights
LAMBDA_EO_ADV = 2.0
LAMBDA_EO_CYCLE = 10.0
LAMBDA_SAR_CYCLE = 7.5
LAMBDA_SAR_ADV = 1.5
LAMBDA_PERCEPTUAL = 0.25
LAMBDA_SSIM = 0.25
LAMBDA_GP = 0.0

# Data split
TRAIN_SPLIT = 0.1  # Note: This is set to 0.1 as in the original notebook

# Model architecture parameters
SAR_CHANNELS = 2  # VV, VH
EO_CHANNELS = 13  # All S2 bands

# Normalization ranges
SAR_NORM_RANGE = (-31.9, -1.3)
EO_NORM_RANGE = (7.0, 5356.0)

# Optimizer parameters
BETAS = (0.5, 0.999)
WEIGHT_DECAY = [1e-4, 1e-4, 0, 0]

# Visualization parameters
RGB_INDICES = [3, 2, 1]  # S2 bands for RGB visualization
NIR_SWIR_RE_INDICES = [7, 11, 4]  # NIR, SWIR, Red Edge for false color

# Random seed for reproducibility
RANDOM_SEED = 42

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GENERATED_SAMPLES_DIR, exist_ok=True)
