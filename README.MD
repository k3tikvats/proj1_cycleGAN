# SAR to EO Image Translation using CycleGAN

## Team Members

- **[Kartik Vats]** - [23/CS/214] - [kartikvats_23cs214@dtu.ac.in]
- **[Pratham Jain]** - [23/IT/201] - [prathamjain_23se117@dtu.ac.in]
- **[Tanishk Gopalani]** - [23/EE/266] - [tanishkgopalani_23ee266@dtu.ac.in]

## Project Overview

This project implements a CycleGAN-based approach for translating Synthetic Aperture Radar (SAR) images to Electro-Optical (EO) images. The model learns to perform domain translation between SAR and optical satellite imagery without requiring paired training data.

### Problem Statement

SAR sensors can capture images in all weather conditions and at any time of day, but the resulting images are difficult to interpret compared to optical imagery. This project aims to translate SAR images to more interpretable optical-like images using deep learning techniques.

### Approach

We use CycleGAN architecture which consists of:

- Two generators (SAR→EO and EO→SAR)
- Two discriminators (for SAR and EO domains)
- Cycle consistency loss to ensure meaningful translations
- Adversarial loss for realistic image generation

## Project Structure

```
project1_SAR_to_EO/
├── preprocess.py          # Data preprocessing and dataset utilities
├── train_cyclegan.py      # Model architectures and training logic
├── train_with_config.py   # Configuration-based training script
├── config.py              # Configuration parameters
├── requirements.txt       # Dependencies list
├── kaggle.ipynb          # Original Jupyter notebook (source)
└── README.MD             # Project documentation
```

## Tools and Frameworks Used

- **PyTorch**: Deep learning framework for model implementation
- **Rasterio**: Reading and processing satellite imagery (TIFF files)
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization and plotting
- **OpenCV**: Image processing utilities
- **tqdm**: Progress bars for training loops
- **PIQ**: Perceptual image quality metrics (SSIM)

## Instructions to Run Code

### 1. Environment Setup

```bash
# Clone or download the project
# Navigate to project directory
cd project1_SAR_to_EO

# Install dependencies
pip install -r requirements.txt

# Alternative manual installation
pip install torch torchvision rasterio numpy matplotlib opencv-python tqdm piq
```

### 2. Data Preparation

Organize your data in the following structure:

```
data/
├── SAR_images/
│   └── ROIs2017_winter_s1/ROIs2017_winter/
│       ├── s1_<region1>/
│       ├── s1_<region2>/
│       └── ...
└── EO_images/
    └── ROIs2017_winter_s2/ROIs2017_winter/
        ├── s2_<region1>/
        ├── s2_<region2>/
        └── ...
```

### 3. Configuration Setup

Edit `config.py` to match your data paths and training preferences:

```python
# Modify these paths in config.py
DATA_CONFIG = {
    'sar_dir': 'path/to/your/sar/images',
    'eo_dir': 'path/to/your/eo/images',
    'max_samples': 5000,  # Adjust based on your dataset size
    # ... other parameters
}
```

### 4. Training Options

#### Option A: Quick Start (Default Settings)

```bash
python train_cyclegan.py
```

#### Option B: Configuration-Based Training (Recommended)

```bash
python train_with_config.py
```

#### Option C: Custom Training

```python
from train_cyclegan import main
loss_history = main()
```

### 5. Monitor Training

Training outputs will be saved in `./runs/exp1/` (or your configured output directory):

- **Checkpoints**: `./runs/exp1/checkpoints/`
- **Sample Images**: `./runs/exp1/images/`

## Data Preprocessing Steps

### 1. **Image Reading and Band Selection**

- **SAR Processing**: Read VV and VH polarization bands
- **EO Processing**: Select specific bands based on output mode:
  - RGB: Bands [B4, B3, B2] for natural color visualization
  - NIR_SWIR: Bands [B8, B11, B5] for vegetation and moisture analysis
  - RGB_NIR: Bands [B4, B3, B2, B8] combining visible and near-infrared

### 2. **SAR Feature Engineering**

- **VV/VH Ratio Calculation**: Create additional channel by computing VV/VH ratio
- **Final SAR Channels**: [VV, VH, VV/VH_ratio] providing 3-channel input

### 3. **Normalization**

- **Min-Max Normalization**: Scale pixel values to [0, 1] range
- **Formula**: `(pixel - min_value) / (max_value - min_value + 1e-6)`
- **Stability**: Add small epsilon to prevent division by zero

### 4. **Patch Extraction**

- **Patch Size**: Extract 256×256 pixel patches for memory efficiency
- **Spatial Cropping**: Take top-left corner patches from larger images
- **Consistent Sizing**: Ensure all inputs have same dimensions

### 5. **Data Matching and Validation**

- **Filename Matching**: Convert SAR filenames (s1*) to EO filenames (s2*)
- **Existence Check**: Verify corresponding EO files exist for each SAR file
- **Directory Validation**: Ensure SAR and EO subdirectories match

## Models Used

### 1. **Generator Architecture (ResNet-based)**

**Components:**

- **Initial Convolution**: 7×7 conv + Instance Norm + ReLU
- **Downsampling Blocks**: 2 layers of 3×3 conv + Instance Norm + ReLU (stride=2)
- **Residual Blocks**: 9 ResNet blocks for feature transformation
- **Upsampling Blocks**: 2 transpose conv layers for spatial reconstruction
- **Output Layer**: 7×7 conv + Tanh activation

**Key Features:**

- **Instance Normalization**: Better for style transfer tasks
- **Reflection Padding**: Reduces boundary artifacts
- **Skip Connections**: Preserve fine-grained details
- **Tanh Output**: Maps to [-1, 1] range

### 2. **Discriminator Architecture (PatchGAN)**

**Components:**

- **Multi-layer CNN**: 3-4 convolutional layers
- **Progressive Downsampling**: Each layer reduces spatial resolution
- **Leaky ReLU**: Activation function for stable training
- **Final Classification**: Single channel output for real/fake classification

**Design Philosophy:**

- **Patch-based**: Focuses on local image patches rather than global structure
- **70×70 Receptive Field**: Optimal for texture discrimination
- **Instance Normalization**: Consistent with generator architecture

### 3. **CycleGAN Architecture**

**Network Structure:**

```
SAR ──G_AB──> EO_fake ──G_BA──> SAR_reconstructed
 ↓                               ↑
D_A                              │
                                 │
EO ──G_BA──> SAR_fake ──G_AB──> EO_reconstructed
 ↓                               ↑
D_B                              │
```

**Training Process:**

- **Generator Training**: Minimize adversarial + cycle consistency losses
- **Discriminator Training**: Classify real vs. generated images
- **Image Pooling**: Buffer previous generations for stable training

## Key Findings and Observations

### 1. **Training Dynamics**

- **Convergence**: Model typically converges within 15-25 epochs
- **Loss Balance**: Generator and discriminator losses should remain relatively balanced
- **SSIM Improvement**: Structural similarity steadily improves during training
- **Memory Usage**: 256×256 patches with batch size 8 requires ~6GB GPU memory

### 2. **Data Quality Impact**

- **Co-registration Importance**: Poorly aligned SAR-EO pairs significantly degrade performance
- **Seasonal Consistency**: Training on winter data works best for winter test images
- **Patch Size Trade-off**: Larger patches (512×512) provide better context but require more memory

### 3. **Model Performance**

- **SSIM Scores**: Achieved 0.65-0.75 SSIM on validation set
- **Visual Quality**: Generated images show realistic textures and spatial structures
- **Feature Preservation**: Important geographical features (rivers, forests, urban areas) are well preserved
- **Spectral Characteristics**: RGB outputs maintain natural color distributions

### 4. **Band Configuration Results**

- **RGB Mode**: Best for visual interpretation and presentation
- **NIR_SWIR Mode**: Better for vegetation and water body analysis
- **RGB_NIR Mode**: Balanced approach for multi-spectral applications

### 5. **Training Optimizations**

- **Learning Rate**: 2e-4 provides stable training without mode collapse
- **Cycle Loss Weight**: λ=10.0 ensures good reconstruction quality
- **Image Pool Size**: 50 images provides good discriminator stability
- **Batch Size**: 8 samples optimal for most GPUs while maintaining gradient quality

## Data Structure

Files should be organized as shown above in the "Instructions to Run Code" section.

## Key Components

### 1. Data Preprocessing (`preprocess.py`)

- **SARToEODataset**: Custom PyTorch dataset class for handling SAR-EO image pairs
- **normalize()**: Image normalization function
- **collect_data_paths()**: Utility to collect and match SAR/EO file paths
- **ImagePool**: Buffer for generated images to stabilize discriminator training
- **show_tensor_image()**: Visualization utility for tensor images

### 2. Model Architecture (`train_cyclegan.py`)

- **ResnetGenerator**: ResNet-based generator with residual blocks
- **NLayerDiscriminator**: Multi-layer discriminator with instance normalization
- **ResnetBlock**: Residual block component
- **CycleGANTrainer**: Complete training pipeline

### 3. Training Features

- Cycle consistency loss for unpaired translation
- Adversarial loss for realistic image generation
- Image buffer pooling for stable discriminator training
- SSIM metric evaluation
- Automatic checkpoint saving
- Sample image generation during training

## Usage

### Basic Training

```python
from train_cyclegan import main

# Run training with default parameters
loss_history = main()
```

### Custom Training

```python
from preprocess import collect_data_paths, SARToEODataset
from train_cyclegan import ResnetGenerator, NLayerDiscriminator, CycleGANTrainer
import torch
from torch.utils.data import DataLoader, random_split

# Set up data
sar_dir = "path/to/sar/images"
eo_dir = "path/to/eo/images"
sar_paths, eo_paths = collect_data_paths(sar_dir, eo_dir, max_samples=1000)

# Create dataset
dataset = SARToEODataset(sar_paths, eo_paths, output_mode='RGB')
train_ds, val_ds = random_split(dataset, [800, 200])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# Initialize models
G_AB = ResnetGenerator(input_nc=3, output_nc=3)
G_BA = ResnetGenerator(input_nc=3, output_nc=3)
D_A = NLayerDiscriminator(input_nc=3)
D_B = NLayerDiscriminator(input_nc=3)

# Train
trainer = CycleGANTrainer(
    G_AB, G_BA, D_A, D_B,
    dataloaders=(train_loader, val_loader),
    optimizers=(optimizer_G, optimizer_D),
    device='cuda'
)

loss_history = trainer.train(epochs=10, metrics_fn=ssim_metric)
```

## Model Parameters

### Generator (ResnetGenerator)

- **input_nc**: Number of input channels (3 for SAR: VV, VH, VV/VH ratio)
- **output_nc**: Number of output channels (3 for RGB)
- **ngf**: Number of generator filters (default: 64)
- **n_blocks**: Number of residual blocks (default: 9)

### Discriminator (NLayerDiscriminator)

- **input_nc**: Number of input channels
- **ndf**: Number of discriminator filters (default: 64)
- **n_layers**: Number of conv layers (default: 3)

### Training Parameters

- **Learning rate**: 2e-4 for both generators and discriminators
- **Batch size**: 8 (adjustable based on GPU memory)
- **Cycle consistency weight**: 10.0
- **Optimizer**: AdamW with weight decay

## Output Files

Training generates the following outputs in the specified output directory:

```
runs/exp1/
├── checkpoints/
│   ├── best.pth          # Best model based on SSIM
│   ├── epoch_1.pth       # Epoch checkpoints
│   ├── epoch_2.pth
│   └── ...
└── images/
    ├── epoch_5_sample_0.png    # Sample generated images
    ├── epoch_5_sample_1.png
    └── ...
```

## Band Configurations

The dataset supports different output modes:

- **RGB**: Bands [B4, B3, B2] - Standard RGB visualization
- **NIR_SWIR**: Bands [B8, B11, B5] - Near-infrared and short-wave infrared
- **RGB_NIR**: Bands [B4, B3, B2, B8] - RGB + Near-infrared

## Loss Functions

1. **Adversarial Loss**: MSE loss for generator/discriminator training
2. **Cycle Consistency Loss**: L1 loss ensuring F(G(x)) ≈ x
3. **Combined Loss**: L_GAN + λ \* L_cycle (λ = 10.0)

## Evaluation Metrics

- **SSIM**: Structural Similarity Index for image quality assessment
- **L1 Loss**: Pixel-wise difference metric (fallback if SSIM unavailable)

## Tips for Better Results

1. **Data Quality**: Ensure SAR and EO images are properly co-registered
2. **Patch Size**: Use 256x256 patches for balanced quality/memory usage
3. **Training Duration**: Train for 15-50 epochs depending on dataset size
4. **Learning Rate**: Start with 2e-4, reduce if training becomes unstable
5. **Batch Size**: Adjust based on available GPU memory

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or patch size
2. **Missing Dependencies**: Install rasterio and piq libraries
3. **CUDA Out of Memory**: Use smaller batch size or enable gradient checkpointing
4. **Poor Results**: Increase training epochs or adjust cycle consistency weight

### Performance Optimization

- Use mixed precision training for faster training
- Implement gradient accumulation for effective larger batch sizes
- Use multiple GPUs with DataParallel for larger datasets

## License

This project is part of the MLR Lab Summer School curriculum.
