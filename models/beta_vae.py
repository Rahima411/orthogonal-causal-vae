"""
β-VAE: Variational Autoencoder with controllable disentanglement
Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (ICLR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE


class BetaVAE(VAE):
    """
    β-VAE with controllable beta parameter
    
    The beta parameter controls the trade-off between:
    - Reconstruction quality (low beta)
    - Disentanglement (high beta)
    
    Typical values: beta ∈ [1, 10]
    """
    
    def __init__(self, latent_dim=10, img_channels=1, hidden_dims=None, beta=4.0):
        """
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of image channels
            hidden_dims: Hidden dimensions for encoder/decoder
            beta: Weight for KL divergence term (>1 for disentanglement)
        """
        super().__init__(latent_dim, img_channels, hidden_dims)
        
        self.beta = beta
    
    def loss_function(self, x, x_recon, mu, logvar):
        """
        Compute β-VAE loss: Reconstruction + β * KL
        
        Args:
            x: Input images
            x_recon: Reconstructed images
            mu: Latent means
            logvar: Latent log-variances
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (binary cross-entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        # Per-sample losses for logging
        loss_dict = {
            'loss': total_loss.item() / batch_size,
            'recon': recon_loss.item() / batch_size,
            'kl': kl_loss.item() / batch_size,
            'beta': self.beta
        }
        
        return total_loss, loss_dict
    
    def set_beta(self, beta):
        """Update beta parameter"""
        self.beta = beta


def test_beta_vae():
    """Test β-VAE architecture"""
    batch_size = 8
    img_channels = 1
    img_size = 64
    latent_dim = 10
    beta = 4.0
    
    # Create model
    model = BetaVAE(latent_dim=latent_dim, img_channels=img_channels, beta=beta)
    
    # Test forward pass
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    x = torch.sigmoid(x)  # Ensure in [0, 1]
    
    x_recon, mu, logvar, z = model(x)
    
    # Test loss
    loss, loss_dict = model.loss_function(x, x_recon, mu, logvar)
    
    print(f"β-VAE test (beta={beta}):")
    print(f"  Total loss: {loss_dict['loss']:.4f}")
    print(f"  Recon loss: {loss_dict['recon']:.4f}")
    print(f"  KL loss: {loss_dict['kl']:.4f}")
    
    # Test beta update
    model.set_beta(10.0)
    loss, loss_dict = model.loss_function(x, x_recon, mu, logvar)
    print(f"\nAfter beta=10.0:")
    print(f"  Total loss: {loss_dict['loss']:.4f}")
    
    print("\nβ-VAE test passed!")


if __name__ == '__main__':
    test_beta_vae()