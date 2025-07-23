"""
Data preprocessing and dataset handling for SAR to EO conversion.
Handles loading, normalization, and data augmentation for SEN12MS dataset.
"""

import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from enum import Enum
from glob import glob
import random
from typing import Tuple, List, Optional
import warnings

from config import *

warnings.filterwarnings('ignore')

class S1Bands(Enum):
    """Sentinel-1 SAR band definitions"""
    VV = 1
    VH = 2
    ALL = [VV, VH]

class S2Bands(Enum):
    """Sentinel-2 EO band definitions"""
    B01 = 1   # Aerosol
    B02 = 2   # Blue
    B03 = 3   # Green  
    B04 = 4   # Red
    B05 = 5   # Red Edge 1
    B06 = 6   # Red Edge 2
    B07 = 7   # Red Edge 3
    B08 = 8   # NIR
    B08A = 9  # NIR Narrow
    B09 = 10  # Water Vapor
    B10 = 11  # Cirrus
    B11 = 12  # SWIR 1
    B12 = 13  # SWIR 2
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    # Project specific configurations
    RGB = [B04, B03, B02]  # B4, B3, B2
    NIR_SWIR_RE = [B08, B11, B05]  # B8, B11, B5  
    RGB_NIR = [B04, B03, B02, B08]  # B4, B3, B2, B8


class SARToEODataset(Dataset):
    """
    Dataset class for SAR to EO image pairs from SEN12MS dataset.
    
    Args:
        base_dir (str): Path to the dataset root directory
        s1_bands (List[S1Bands]): SAR bands to load
        s2_bands (List[S2Bands]): EO bands to load
        normalize (bool): Whether to normalize data to [-1, 1] range
        transform: Optional transforms to apply
    """
    
    def __init__(self, 
                 base_dir: str, 
                 s1_bands: List[S1Bands] = S1Bands.ALL,
                 s2_bands: List[S2Bands] = S2Bands.ALL,
                 normalize: bool = True,
                 transform=None):
        self.base_dir = base_dir
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands  
        self.normalize = normalize
        self.transform = transform
        
        # Define paths to S1 and S2 data
        self.s1_dir = os.path.join(base_dir, "ROIs2017_winter_s1", "ROIs2017_winter")
        self.s2_dir = os.path.join(base_dir, "ROIs2017_winter_s2", "ROIs2017_winter")
        
        # Validate paths exist
        if not os.path.exists(self.s1_dir):
            raise FileNotFoundError(f"S1 directory not found: {self.s1_dir}")
        if not os.path.exists(self.s2_dir):
            raise FileNotFoundError(f"S2 directory not found: {self.s2_dir}")
            
        self.patch_list = self._get_patch_list()
    
    def _get_patch_list(self) -> List[Tuple[str, int, int]]:
        """
        Generate list of valid SAR-EO patch pairs.
        
        Returns:
            List of tuples containing (scene_name, scene_id, patch_id)
        """
        patch_list = []
        
        s1_scenes = glob(os.path.join(self.s1_dir, "s1_*"))
        
        for s1_scene_path in s1_scenes:
            scene_name = os.path.basename(s1_scene_path)
            scene_id = int(scene_name.split('_')[1])
            
            s2_scene_path = os.path.join(self.s2_dir, f"s2_{scene_id}")
            if not os.path.exists(s2_scene_path):
                continue
                
            s1_patches = glob(os.path.join(s1_scene_path, "*.tif"))
            
            for s1_patch_path in s1_patches:
                filename = os.path.basename(s1_patch_path)
                patch_id = int(filename.split('_p')[1].split('.')[0])
                
                s2_filename = f"ROIs2017_winter_s2_{scene_id}_p{patch_id}.tif"
                s2_patch_path = os.path.join(s2_scene_path, s2_filename)
                
                if os.path.exists(s2_patch_path):
                    patch_list.append((scene_name, scene_id, patch_id))
        
        return patch_list
    
    def _load_patch(self, sensor_dir: str, scene_id: int, patch_id: int, 
                   bands: List, sensor_prefix: str) -> np.ndarray:
        """
        Load a specific patch from raster file.
        
        Args:
            sensor_dir: Directory containing sensor data
            scene_id: Scene identifier
            patch_id: Patch identifier
            bands: List of bands to load
            sensor_prefix: Sensor prefix ('s1' or 's2')
            
        Returns:
            Numpy array of shape (bands, height, width)
        """
        scene_dir = os.path.join(sensor_dir, f"{sensor_prefix}_{scene_id}")
        filename = f"ROIs2017_winter_{sensor_prefix}_{scene_id}_p{patch_id}.tif"
        patch_path = os.path.join(scene_dir, filename)
        
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"Patch not found: {patch_path}")
        
        if isinstance(bands[0], (S1Bands, S2Bands)):
            band_indices = [b.value for b in bands]
        else:
            band_indices = bands
            
        with rasterio.open(patch_path) as src:
            data = src.read(band_indices)
            
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
            
        return data
    
    def _normalize_to_gan_range(self, data: np.ndarray, sensor_type: str) -> np.ndarray:
        """
        Normalize data to [-1, 1] range suitable for GAN training.
        
        Args:
            data: Input data array
            sensor_type: 's1' for SAR or 's2' for EO
            
        Returns:
            Normalized data in range [-1, 1]
        """
        if not self.normalize:
            return data

        if sensor_type == 's1':
            # SAR data normalization
            data = np.clip(data, SAR_NORM_RANGE[0], SAR_NORM_RANGE[1])
            data = 2.0 * ((data + abs(SAR_NORM_RANGE[0])) / (abs(SAR_NORM_RANGE[0]) - SAR_NORM_RANGE[1])) - 1.0
            
        elif sensor_type == 's2':
            # EO data normalization
            data = np.clip(data, EO_NORM_RANGE[0], EO_NORM_RANGE[1])
            data = 2.0 * ((data - EO_NORM_RANGE[0]) / (EO_NORM_RANGE[1] - EO_NORM_RANGE[0])) - 1.0
            
        return data.astype(np.float32)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.patch_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a SAR-EO image pair.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (sar_tensor, eo_tensor)
        """
        scene_name, scene_id, patch_id = self.patch_list[idx]
        
        # Load SAR data
        sar_data = self._load_patch(self.s1_dir, scene_id, patch_id, 
                                   self.s1_bands, "s1")
        sar_data = self._normalize_to_gan_range(sar_data, 's1')
        
        # Load EO data
        eo_data = self._load_patch(self.s2_dir, scene_id, patch_id,
                                  self.s2_bands, "s2")
        eo_data = self._normalize_to_gan_range(eo_data, 's2')
        
        # Convert to tensors
        sar_tensor = torch.from_numpy(sar_data).float()
        eo_tensor = torch.from_numpy(eo_data).float()
        
        # Apply transforms if specified
        if self.transform:
            sar_tensor = self.transform(sar_tensor)
            eo_tensor = self.transform(eo_tensor)

        # Handle NaN/Inf values by returning a random sample
        if (torch.isnan(sar_tensor).any() or torch.isnan(eo_tensor).any() or 
            torch.isinf(sar_tensor).any() or torch.isinf(eo_tensor).any()):
            print(f'NAN/INF detected in batch: {idx}')
            return self.__getitem__(random.randint(0, len(self.patch_list) - 1))
            
        return sar_tensor, eo_tensor
    
    def get_band_configs(self) -> dict:
        """
        Get predefined band configurations for visualization.
        
        Returns:
            Dictionary with RGB, NIR_SWIR_RE, and RGB_NIR band indices
        """
        return {
            'RGB': [b.value for b in S2Bands.RGB.value],
            'NIR_SWIR_RE': [b.value for b in S2Bands.NIR_SWIR_RE.value], 
            'RGB_NIR': [b.value for b in S2Bands.RGB_NIR.value]
        }


def create_dataloaders(dataset_path: str, 
                      batch_size: int = BATCH_SIZE,
                      train_split: float = TRAIN_SPLIT,
                      num_workers: int = NUM_WORKERS,
                      shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for data loading
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create full dataset
    full_dataset = SARToEODataset(
        base_dir=dataset_path,
        s1_bands=S1Bands.ALL.value,  # VV, VH
        s2_bands=S2Bands.ALL.value,  # All 13 bands
        normalize=True
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader


def setup_data_environment():
    """
    Set up the data environment with proper seeding and configurations.
    """
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Configure PyTorch for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    # Example usage
    setup_data_environment()
    
    # Test dataset loading
    dataset = SARToEODataset(
        base_dir=DATA_PATH,
        s1_bands=S1Bands.ALL.value,
        s2_bands=S2Bands.ALL.value,
        normalize=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sar_patch, eo_patch = dataset[0]
    print(f"SAR range: [{sar_patch.min():.3f}, {sar_patch.max():.3f}]")
    print(f"EO range: [{eo_patch.min():.3f}, {eo_patch.max():.3f}]")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(DATA_PATH)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
