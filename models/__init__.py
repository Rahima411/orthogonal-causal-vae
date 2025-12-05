"""
Models module for Orthogonal Causal β-VAE
"""

from .vae import VAE, Encoder, Decoder
from .beta_vae import BetaVAE
from .ortho_vae import OrthoVAE
from .causal_vae import CausalVAE
from .ortho_causal_vae import OrthoCausalVAE

__all__ = [
    'VAE',
    'Encoder',
    'Decoder',
    'BetaVAE',
    'OrthoVAE',
    'CausalVAE',
    'OrthoCausalVAE',
]


def get_model(name, latent_dim=10, img_channels=1, causal_graph=None, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        name: Model name ('vae', 'beta_vae', 'ortho_vae', 'causal_vae', 'ortho_causal_vae')
        latent_dim: Latent dimension size
        img_channels: Number of image channels
        causal_graph: Causal graph structure (for causal models)
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    name = name.lower()
    
    if name == 'vae':
        return VAE(latent_dim=latent_dim, img_channels=img_channels, **kwargs)
    elif name == 'beta_vae':
        return BetaVAE(latent_dim=latent_dim, img_channels=img_channels, **kwargs)
    elif name == 'ortho_vae':
        return OrthoVAE(latent_dim=latent_dim, img_channels=img_channels, **kwargs)
    elif name == 'causal_vae':
        return CausalVAE(latent_dim=latent_dim, img_channels=img_channels, 
                        causal_graph=causal_graph, **kwargs)
    elif name == 'ortho_causal_vae':
        return OrthoCausalVAE(latent_dim=latent_dim, img_channels=img_channels,
                             causal_graph=causal_graph, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")