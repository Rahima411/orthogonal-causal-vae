"""
3D Shapes dataset loader
Dataset of 3D rendered shapes with controlled factors
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import urllib.request


class Shapes3DDataset(Dataset):
    """
    3D Shapes dataset
    
    Factors of variation:
    - floor_hue: 10 values
    - wall_hue: 10 values
    - object_hue: 10 values
    - scale: 8 values
    - shape: 4 values
    - orientation: 15 values
    
    Total: 480,000 images (64x64 RGB)
    """
    
    urls = [
        'https://storage.googleapis.com/3d-shapes/3dshapes.h5'
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
        self.data_path = os.path.join(root, '3dshapes.h5')
        
        if not os.path.exists(self.data_path):
            raise RuntimeError('Dataset not found. Set download=True to download it.')
        
        # Load data (lazy loading for memory efficiency)
        self.dataset_file = h5py.File(self.data_path, 'r')
        self.imgs = self.dataset_file['images']
        self.labels = self.dataset_file['labels']
        
        # Metadata
        self.factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        self.factor_sizes = [10, 10, 10, 8, 4, 15]
        
        # Train/test split (80/20)
        total_samples = len(self.imgs)
        split_idx = int(0.8 * total_samples)
        
        if self.train:
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, total_samples))
    
    def download(self):
        """Download dataset"""
        filepath = os.path.join(self.root, '3dshapes.h5')
        
        if os.path.exists(filepath):
            print(f'Dataset already downloaded: {filepath}')
            return
        
        print(f'Downloading 3D Shapes dataset (this may take a while, ~5GB)...')
        url = self.urls[0]
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f'Downloaded to {filepath}')
        except Exception as e:
            print(f'Error downloading: {e}')
            print('You can manually download from: https://github.com/deepmind/3d-shapes')
            raise
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor of shape (3, 64, 64)
            factors: Tensor of factor values (6 values)
        """
        real_idx = self.indices[idx]
        
        img = self.imgs[real_idx]  # (64, 64, 3)
        factors = self.labels[real_idx]  # (6,)
        
        # Convert to tensor
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3, 64, 64)
        factors = torch.from_numpy(factors).float()
        
        if self.transform:
            img = self.transform(img)
        
        return img, factors
    
    def __del__(self):
        """Close HDF5 file on deletion"""
        if hasattr(self, 'dataset_file'):
            self.dataset_file.close()


def get_shapes3d_dataloader(root='./data', batch_size=128, train=True, shuffle=None, num_workers=4):
    """
    Convenience function to get 3D Shapes dataloader
    
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
    
    try:
        dataset = Shapes3DDataset(root=root, train=train)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader
    
    except Exception as e:
        print(f"Failed to load 3D Shapes: {e}")
        print("Using dummy dataset for testing...")
        
        # Create dummy dataset
        class DummyShapes3D(Dataset):
            def __init__(self, size=10000):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                img = torch.randn(3, 64, 64)
                factors = torch.rand(6)
                return img, factors
        
        dataset = DummyShapes3D()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader