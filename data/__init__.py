"""
Data module for Orthogonal Causal β-VAE
Provides dataset loaders for disentanglement benchmarks
"""

from .dsprites import DSpritesDataset
from .celeba import CelebADataset
from .shapes3d import Shapes3DDataset

__all__ = [
    'DSpritesDataset',
    'CelebADataset',
    'Shapes3DDataset',
]


def get_dataset(name, root='./data', train=True, **kwargs):
    """
    Factory function to get dataset by name
    
    Args:
        name: Dataset name ('dsprites', 'celeba', 'shapes3d')
        root: Root directory for data
        train: Whether to load training or test set
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dataset object
    """
    name = name.lower()
    
    if name == 'dsprites':
        return DSpritesDataset(root=root, train=train, **kwargs)
    elif name == 'celeba':
        return CelebADataset(root=root, train=train, **kwargs)
    elif name == 'shapes3d':
        return Shapes3DDataset(root=root, train=train, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: dsprites, celeba, shapes3d")


def get_causal_graph(dataset_name):
    """
    Get predefined causal graph for dataset
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dictionary mapping latent index to parent indices
    """
    if dataset_name == 'dsprites':
        # Factors: shape (3), scale (6), rotation (40), posX (32), posY (32)
        # Causal structure: shape and scale are roots
        # Position depends on shape and scale
        # Rotation depends on shape
        return {
            0: [],      # shape (root)
            1: [],      # scale (root)
            2: [0],     # rotation (depends on shape)
            3: [0, 1],  # posX (depends on shape, scale)
            4: [0, 1],  # posY (depends on shape, scale)
        }
    
    elif dataset_name == 'shapes3d':
        # Factors: floor_hue, wall_hue, object_hue, scale, shape, orientation
        # Causal structure: hues and shape are roots
        # Scale and orientation may depend on object properties
        return {
            0: [],      # floor_hue (root)
            1: [],      # wall_hue (root)
            2: [],      # object_hue (root)
            3: [2, 4],  # scale (depends on object_hue, shape)
            4: [],      # shape (root)
            5: [4],     # orientation (depends on shape)
        }
    
    elif dataset_name == 'celeba':
        # Simplified causal structure for face attributes
        # gender, age are roots; others may depend on these
        return {
            0: [],      # gender (root)
            1: [],      # age (root)
            2: [0],     # hair style (depends on gender)
            3: [0, 1],  # glasses (depends on gender, age)
            4: [0],     # facial hair (depends on gender)
        }
    
    else:
        return {}  # No causal structure