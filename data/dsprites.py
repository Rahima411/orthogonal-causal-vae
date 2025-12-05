"""
dSprites dataset loader
Dataset of 2D shapes with controlled factors of variation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import urllib.request
from PIL import Image


class DSpritesDataset(Dataset):
    """
    dSprites dataset
    
    Factors of variation:
    - shape: 3 values (square, ellipse, heart)
    - scale: 6 values
    - orientation: 40 values
    - posX: 32 values
    - posY: 32 values
    
    Total: 737,280 images (64x64 binary)
    """
    
    urls = [
        'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    ]
    
    def __init__(self, root='./data', train=True, transform=None, download=True):
        """
        Args:
            root: Root directory for dataset
            train: If True, use training set; else test set
            transform: Optional transforms
            download: If True, download dataset if not present
        """
        self.root = root
        self.train = train
        self.transform = transform
        
        # Create directory
        os.makedirs(root, exist_ok=True)
        
        # Download if needed
        if download:
            self.download()
        
        # Load dataset
        self.data_path = os.path.join(root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        
        if not os.path.exists(self.data_path):
            raise RuntimeError('Dataset not found. Set download=True to download it.')
        
        # Load data
        dataset_zip = np.load(self.data_path, allow_pickle=True, encoding='bytes')
        
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        
        # Metadata
        self.metadata = {
            'latents_names': ['color', 'shape', 'scale', 'orientation', 'posX', 'posY'],
            'latents_sizes': np.array([1, 3, 6, 40, 32, 32])
        }
        
        # Train/test split (80/20)
        total_samples = len(self.imgs)
        split_idx = int(0.8 * total_samples)
        
        if self.train:
            self.imgs = self.imgs[:split_idx]
            self.latents_values = self.latents_values[:split_idx]
            self.latents_classes = self.latents_classes[:split_idx]
        else:
            self.imgs = self.imgs[split_idx:]
            self.latents_values = self.latents_values[split_idx:]
            self.latents_classes = self.latents_classes[split_idx:]
    
    def download(self):
        """Download dataset"""
        filepath = os.path.join(self.root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        
        if os.path.exists(filepath):
            print(f'Dataset already downloaded: {filepath}')
            return
        
        print(f'Downloading dSprites dataset...')
        url = self.urls[0]
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f'Downloaded to {filepath}')
        except Exception as e:
            print(f'Error downloading: {e}')
            raise
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor of shape (1, 64, 64)
            factors: Tensor of factor values (excluding color which is constant)
        """
        img = self.imgs[idx]
        factors = self.latents_classes[idx, 1:]  # Exclude color (constant)
        
        # Convert to tensor
        img = torch.from_numpy(img).float().unsqueeze(0)  # (1, 64, 64)
        factors = torch.from_numpy(factors).long()
        
        if self.transform:
            img = self.transform(img)
        
        return img, factors
    
    def get_img_by_latent(self, shape=0, scale=0, orientation=0, posX=0, posY=0):
        """
        Get image by specifying latent factor values
        
        Args:
            shape: 0-2
            scale: 0-5
            orientation: 0-39
            posX: 0-31
            posY: 0-31
            
        Returns:
            Image tensor
        """
        # Compute flat index
        latent_sizes = self.metadata['latents_sizes']
        idx = (shape * latent_sizes[2] * latent_sizes[3] * latent_sizes[4] * latent_sizes[5] +
               scale * latent_sizes[3] * latent_sizes[4] * latent_sizes[5] +
               orientation * latent_sizes[4] * latent_sizes[5] +
               posX * latent_sizes[5] +
               posY)
        
        return self.__getitem__(idx)[0]


def get_dsprites_dataloader(root='./data', batch_size=128, train=True, shuffle=None, num_workers=4):
    """
    Convenience function to get dSprites dataloader
    
    Args:
        root: Root directory
        batch_size: Batch size
        train: Training or test set
        shuffle: Whether to shuffle (defaults to train=True)
        num_workers: Number of workers for loading
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    if shuffle is None:
        shuffle = train
    
    dataset = DSpritesDataset(root=root, train=train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader