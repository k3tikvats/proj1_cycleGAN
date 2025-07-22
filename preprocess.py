import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import torch
from torch.utils.data import Dataset
import random


def normalize(img):
    """Normalize image to [0, 1] range."""
    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-6)
    return img


def show_tensor_image(img_tensor, title='', cmap=None, bands=None):
    """
    Convert CHW tensor to HWC image and display/save with proper handling.
    """
    img = img_tensor.detach().cpu().numpy()
    
    if len(img.shape) == 3:
        if bands is not None:
            img = img[bands]
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)

    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')


def collect_data_paths(sar_dir, eo_dir, max_samples=None):
    """
    Collect and match SAR and EO image paths from directories.
    
    Args:
        sar_dir (str): Path to SAR images directory
        eo_dir (str): Path to EO images directory
        max_samples (int, optional): Maximum number of samples to collect
        
    Returns:
        tuple: (sar_paths, eo_paths) lists of matched file paths
    """
    sar_subdirs = sorted(os.listdir(sar_dir))
    eo_subdirs = sorted(os.listdir(eo_dir))
    assert len(sar_subdirs) == len(eo_subdirs), "SAR and EO folder counts do not match"
    
    sar_paths = []
    eo_paths = []
    
    for s_sub, e_sub in zip(sar_subdirs, eo_subdirs):
        # Verify directory matching (remove s1/s2 prefix for comparison)
        assert s_sub.replace("s1_", "") == e_sub.replace("s2_", ""), f"Unmatched subdirs: {s_sub}, {e_sub}"

        sar_sub_path = os.path.join(sar_dir, s_sub)
        eo_sub_path = os.path.join(eo_dir, e_sub)

        sar_files = sorted(os.listdir(sar_sub_path))
        eo_files = sorted(os.listdir(eo_sub_path))

        # Proper filename matching
        for sar_fname in sar_files:
            # Convert SAR filename to corresponding EO filename
            eo_fname = sar_fname.replace('_s1_', '_s2_')  # s1 → s2 conversion
            
            # Verify the EO file actually exists
            if eo_fname in eo_files:
                sar_paths.append(os.path.join(sar_sub_path, sar_fname))
                eo_paths.append(os.path.join(eo_sub_path, eo_fname))
            else:
                print(f"Warning: No matching EO file for {sar_fname}")
    
    print(f"Length of whole dataset is {len(sar_paths)} pairs")
    
    # Limit dataset size if specified
    if max_samples and len(sar_paths) > max_samples:
        sar_paths = sar_paths[:max_samples]
        eo_paths = eo_paths[:max_samples]
        print(f"Limited dataset to {max_samples} pairs")
    
    return sar_paths, eo_paths


class SARToEODataset(Dataset):
    """Dataset class for SAR to EO image translation."""
    
    def __init__(self, sar_paths, eo_paths, patch_size=256, output_mode='RGB'):
        """
        Initialize dataset.
        
        Args:
            sar_paths (list): List of SAR image file paths
            eo_paths (list): List of EO image file paths
            patch_size (int): Size of image patches
            output_mode (str): Output mode for EO bands ('RGB', 'NIR_SWIR', 'RGB_NIR')
        """
        # Define band indices for Sentinel-2 (0-based)
        self.bands = {
            'RGB': [3, 2, 1],           # B4, B3, B2
            'NIR_SWIR': [7, 10, 4],     # B8, B11, B5
            'RGB_NIR': [3, 2, 1, 7]     # B4, B3, B2, B8
        }
        self.sar_paths = sar_paths
        self.eo_paths = eo_paths
        self.patch_size = patch_size
        self.output_mode = output_mode

    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):
        sar = self.read_image(self.sar_paths[idx], bands=[0, 1])  # VV, VH
        eo_bands = self.bands[self.output_mode]
        eo = self.read_image(self.eo_paths[idx], bands=eo_bands)
        
        sar = torch.from_numpy(sar).float()
        eo = torch.from_numpy(eo).float()
        
        return sar, eo

    def read_image(self, path, bands):
        """
        Read and process image from file.
        
        Args:
            path (str): Path to image file
            bands (list): List of band indices to read
            
        Returns:
            np.ndarray: Processed image array
        """
        with rasterio.open(path) as src:
            img = []
            raw_band_data = []
            for b in bands:
                band_data = normalize(src.read(b + 1))  # rasterio bands start at 1
                raw_band_data.append(band_data)
            
            if bands == [0, 1]:  # SAR processing
                vv, vh = raw_band_data
                vv_vh_ratio = np.divide(vv, vh + 1e-6)
                img = [vv, vh, vv_vh_ratio]
            else:  # EO processing
                img = raw_band_data

            img = np.stack(img, axis=0)
            img = img[:, :self.patch_size, :self.patch_size]
            return img


class ImagePool:
    """Image buffer that stores previously generated images to stabilize training.
    
    This buffer enables us to update discriminators using a history of generated images
    rather than only the most recently generated images.
    """
    
    def __init__(self, pool_size):
        """
        Initialize image pool.
        
        Args:
            pool_size (int): Size of the image buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        """Return images from the pool.
        
        Parameters:
            images: the latest generated images from the generator
        Returns:
            images from the buffer.
        
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if buffer size is 0, do nothing
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if buffer not full
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # 50% chance to return a previously stored image
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # 50% chance to return the current image
                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)
        return return_images