# SAR to EO Image Translation using CycleGAN

This project implements a CycleGAN-based approach for translating Synthetic Aperture Radar (SAR) images to Electro-Optical (EO) images using the SEN12MS dataset.

## Team Members

- **[Kartik Vats]** - [23/CS/214] - [kartikvats_23cs214@dtu.ac.in]
- **[Pratham Jain]** - [23/IT/201] - [prathamjain_23se117@dtu.ac.in]
- **[Tanishk Gopalani]** - [23/EE/266] - [tanishkgopalani_23ee266@dtu.ac.in]

*Note: Please update with actual team member information*

## Project Overview

This project addresses the challenge of converting SAR satellite imagery to EO imagery using deep learning. SAR images provide all-weather, day-night imaging capabilities but are difficult to interpret. Our CycleGAN-based approach learns to translate SAR images to more interpretable EO images while maintaining semantic content.

### Key Features

- **CycleGAN Architecture**: Bidirectional translation between SAR and EO domains
- **ResNet-18 Backbone**: Pretrained encoder for robust feature extraction
- **Custom Decoder**: Specialized upsampling blocks for high-quality image generation
- **Progressive Training**: Learning rate scheduling and layer freezing strategies
- **Multi-loss Optimization**: Combines adversarial, cycle consistency, perceptual, and SSIM losses
- **Comprehensive Evaluation**: Multiple metrics including SSIM, PSNR, and L1 loss

## Instructions to Run Code

### Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (recommended)
3. SEN12MS dataset (download separately)

### Installation

1. Clone the repository:
```bash
git clone [repository_url]
cd project1_SAR_to_EO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update data path in `config.py`:
```python
DATA_PATH = "/path/to/your/sen12ms/dataset"
```

### Training

To test the environment setup:
```bash
python utils.py
```

To start training from scratch:
```bash
python train_cycleGAN.py
```

To start training with the main entry point:
```bash
python run.py train
```

### Evaluation

To evaluate a trained model:
```bash
python evaluate_results.py --checkpoint ./checkpoints/checkpoint_10.pt --num_samples 20
```

To generate sample results:
```bash
python evaluate_results.py --checkpoint ./checkpoints/checkpoint_best.pt --output_dir ./my_results
```

Using the main entry point:
```bash
python run.py eval --checkpoint ./checkpoints/checkpoint_10.pt --num_samples 20
```

### Project Structure

```
project1_SAR_to_EO\
├── README.md                    # Project summary, team info, instructions
├── requirements.txt             # List of Python libraries used
├── preprocess.py               # Preprocessing & normalization code
├── train_cycleGAN.py           # SAR → EO training scripts
├── evaluate_results.py         # Visualizations and metrics
├── generated_samples/          # Sample generated EO images
├── config.py                   # Configuration parameters
├── models.py                   # CycleGAN model architectures
├── losses.py                   # Loss functions
├── utils.py                    # Utility functions and testing
├── run.py                      # Main entry point script
└── checkpoints/                # Model checkpoints (created during training)
```

## Data Preprocessing Steps

### 1. Data Loading
- **SEN12MS Dataset**: Uses winter season ROIs from 2017
- **SAR Data**: Sentinel-1 VV and VH polarizations
- **EO Data**: Sentinel-2 13-band multispectral imagery
- **Patch Matching**: Ensures spatial correspondence between SAR and EO patches

### 2. Normalization
- **SAR Normalization**: Clips values to [-31.9, -1.3] dB range, then normalizes to [-1, 1]
- **EO Normalization**: Clips values to [7.0, 5356.0] range, then normalizes to [-1, 1]
- **GAN-Compatible**: Ensures proper input range for stable GAN training

### 3. Data Augmentation
- **Quality Control**: Automatically filters out patches with NaN or infinite values
- **Random Sampling**: Implements robust sampling to handle corrupted data
- **Memory Optimization**: Uses channels_last memory format for improved performance

## Models Used

### Generator Architecture
- **Encoder**: ResNet-18 backbone with pretrained ImageNet weights
- **Decoder**: Custom upsampling blocks using UpProjection layers
- **Skip Connections**: U-Net style connections for detail preservation
- **Normalization**: Instance normalization for improved GAN stability
- **Output**: Tanh activation for [-1, 1] output range

### Discriminator/Critic Architecture
- **Backbone**: ResNet-18 with modified normalization
- **PatchGAN**: Outputs spatial predictions rather than single values
- **Spectral Normalization**: Applied for training stability
- **Dropout**: Regularization to prevent overfitting

### Training Strategy
- **Progressive Learning**: Different learning rates and freezing schedules across epochs
- **Mixed Precision**: Automatic mixed precision for memory efficiency
- **Gradient Handling**: Robust NaN/Inf gradient detection and correction
- **Checkpointing**: Regular model saves for recovery and evaluation

## Key Findings and Observations

### Training Insights
1. **Progressive Learning**: Early epochs benefit from frozen ResNet layers, allowing custom components to adapt first
2. **Loss Balancing**: Careful weighting of adversarial, cycle consistency, and perceptual losses is crucial
3. **Stability**: Instance normalization and spectral normalization significantly improve training stability
4. **Memory Efficiency**: Channels_last memory format and gradient scaling enable larger batch sizes

### Model Performance
1. **Visual Quality**: Generated EO images show good visual correspondence with ground truth
2. **Spectral Fidelity**: RGB composites maintain realistic color distributions
3. **Detail Preservation**: High-frequency details are reasonably preserved across translation
4. **Semantic Consistency**: Land cover types are generally maintained in translation

### Challenges Addressed
1. **Domain Gap**: Large difference between SAR and EO modalities
2. **Training Stability**: GAN training instability mitigated through careful architecture choices
3. **Memory Constraints**: Efficient implementation allows training on standard hardware
4. **Data Quality**: Robust handling of real-world dataset inconsistencies

## Tools and Frameworks Used

### Core Dependencies
- **PyTorch**: Deep learning framework (v1.9+)
- **torchvision**: Computer vision utilities and pretrained models
- **CUDA**: GPU acceleration for training and inference

### Data Processing
- **rasterio**: Geospatial raster data I/O
- **numpy**: Numerical computing
- **PIL/Pillow**: Image processing and saving

### Evaluation and Visualization
- **matplotlib**: Plotting and visualization
- **torchmetrics**: Standardized metric calculations (SSIM, PSNR)
- **tqdm**: Progress bars for training monitoring

### Development Tools
- **scikit-image**: Additional image processing utilities
- **opencv-python**: Computer vision operations

## Evaluation Metrics

- **SSIM**: Structural Similarity Index for perceptual quality
- **PSNR**: Peak Signal-to-Noise Ratio for signal fidelity
- **L1 Loss**: Mean absolute error between generated and target images
- **Perceptual Loss**: VGG-based feature matching for semantic similarity

## Future Improvements

1. **Architecture Enhancements**: Explore attention mechanisms and newer GAN architectures
2. **Multi-Scale Training**: Implement progressive growing or multi-resolution training
3. **Dataset Expansion**: Include more seasons and geographic regions
4. **Application-Specific Metrics**: Develop domain-specific evaluation criteria
5. **Real-Time Inference**: Optimize models for deployment scenarios

## Acknowledgments

- SEN12MS dataset creators for providing high-quality SAR-EO image pairs
- PyTorch community for robust deep learning framework
- Research community for CycleGAN and related architectural innovations

## License

This project is developed for educational purposes. Please respect the licensing terms of the SEN12MS dataset and other dependencies.

---

*For questions or issues, please contact the team members listed above.*
