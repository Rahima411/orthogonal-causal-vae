"""
CelebA dataset loader with attribute labels
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd


class CelebADataset(Dataset):
    """
    CelebA dataset with attribute labels for causal disentanglement
    
    We select a subset of attributes that have causal relationships:
    - Gender (root cause)
    - Age (root cause)
    - Hair style (depends on gender)
    - Eyeglasses (depends on age, gender)
    - Beard/Mustache (depends on gender)
    """
    
    # Selected attributes (indices in CelebA attributes)
    SELECTED_ATTRS = [
        'Male',           # 0: Gender
        'Young',          # 1: Age
        'Black_Hair',     # 2: Hair (proxy for style)
        'Eyeglasses',     # 3: Glasses
        'Mustache',       # 4: Facial hair
    ]
    
    def __init__(self, root='./data', train=True, transform=None, download=False, img_size=64):
        """
        Args:
            root: Root directory containing 'celeba' folder
            train: If True, use training set; else validation set
            transform: Optional transforms
            download: If True, download dataset (requires manual download)
            img_size: Image size (will be resized)
        """
        self.root = root
        self.train = train
        self.img_size = img_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(148),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Try to use torchvision's CelebA dataset
        try:
            split = 'train' if train else 'test'
            self.celeba = datasets.CelebA(
                root=root,
                split=split,
                target_type='attr',
                transform=self.transform,
                download=download
            )
            
            # Get selected attributes
            self.attr_names = self.celeba.attr_names
            self.selected_indices = [self.attr_names.index(attr) for attr in self.SELECTED_ATTRS]
            
        except Exception as e:
            print(f"Error loading CelebA: {e}")
            print("Note: CelebA requires manual download. Please download from:")
            print("https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            raise
    
    def __len__(self):
        return len(self.celeba)
    
    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor of shape (3, img_size, img_size)
            factors: Selected binary attributes (5 values)
        """
        img, all_attrs = self.celeba[idx]
        
        # Extract selected attributes
        factors = all_attrs[self.selected_indices].float()
        
        return img, factors
    
    def get_factor_names(self):
        """Get names of selected factors"""
        return self.SELECTED_ATTRS


def get_celeba_dataloader(root='./data', batch_size=128, train=True, shuffle=None, 
                          num_workers=4, img_size=64):
    """
    Convenience function to get CelebA dataloader
    
    Args:
        root: Root directory
        batch_size: Batch size
        train: Training or validation set
        shuffle: Whether to shuffle (defaults to train=True)
        num_workers: Number of workers
        img_size: Image size
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    if shuffle is None:
        shuffle = train
    
    try:
        dataset = CelebADataset(root=root, train=train, img_size=img_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader
    
    except Exception as e:
        print(f"Failed to load CelebA: {e}")
        print("Falling back to dummy dataset for testing...")
        
        # Create dummy dataset for testing
        class DummyCelebA(Dataset):
            def __init__(self, size=1000, img_size=64):
                self.size = size
                self.img_size = img_size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                img = torch.randn(3, self.img_size, self.img_size)
                factors = torch.randint(0, 2, (5,)).float()
                return img, factors
        
        dataset = DummyCelebA(img_size=img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader