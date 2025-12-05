"""
Orthogonal VAE: VAE with orthogonality constraint on latent dimensions
Based on: Cha & Thiyagalingam, "Orthogonality-Enforced Latent Space in Autoencoders" (ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .beta_vae import BetaVAE


class OrthoVAE(BetaVAE):
    """
    VAE with orthogonality constraint on latent representations
    
    Enforces latent dimensions to be orthogonal for better disentanglement
    """
    
    def __init__(self, latent_dim=10, img_channels=1, hidden_dims=None, 
                 beta=4.0, lambda_ortho=1.0):
        """
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of image channels
            hidden_dims: Hidden dimensions for encoder/decoder
            beta: Weight for KL divergence
            lambda_ortho: Weight for orthogonality constraint
        """
        super().__init__(latent_dim, img_channels, hidden_dims, beta)
        
        self.lambda_ortho = lambda_ortho
    
    def orthogonality_loss(self, mu):
        """
        Compute orthogonality loss on latent means
        
        Encourages latent dimensions to be orthogonal:
        mu @ mu^T should be close to identity matrix
        
        Args:
            mu: Latent means [batch_size, latent_dim]
            
        Returns:
            ortho_loss: Orthogonality loss (scalar)
        """
        # Normalize each latent dimension across batch
        mu_normalized = F.normalize(mu, p=2, dim=0, eps=1e-8)
        
        # Compute gram matrix: should be identity
        gram = torch.mm(mu_normalized.T, mu_normalized)
        identity = torch.eye(mu.size(1), device=mu.device)
        
        # Frobenius norm of (gram - identity)
        ortho_loss = torch.norm(gram - identity, p='fro') ** 2
        
        return ortho_loss
    
    def loss_function(self, x, x_recon, mu, logvar):
        """
        Compute Orthogonal VAE loss: Reconstruction + β * KL + λ * Orthogonality
        
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
        
        # Standard β-VAE losses
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Orthogonality loss
        ortho_loss = self.orthogonality_loss(mu)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.lambda_ortho * ortho_loss
        
        # Per-sample losses for logging
        loss_dict = {
            'loss': total_loss.item() / batch_size,
            'recon': recon_loss.item() / batch_size,
            'kl': kl_loss.item() / batch_size,
            'ortho': ortho_loss.item() / batch_size,
            'beta': self.beta,
            'lambda_ortho': self.lambda_ortho
        }
        
        return total_loss, loss_dict
    
    def set_lambda_ortho(self, lambda_ortho):
        """Update orthogonality weight"""
        self.lambda_ortho = lambda_ortho


def test_ortho_vae():
    """Test Orthogonal VAE architecture"""
    batch_size = 8
    img_channels = 1
    img_size = 64
    latent_dim = 10
    beta = 4.0
    lambda_ortho = 1.0
    
    # Create model
    model = OrthoVAE(
        latent_dim=latent_dim,
        img_channels=img_channels,
        beta=beta,
        lambda_ortho=lambda_ortho
    )
    
    # Test forward pass
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    x = torch.sigmoid(x)
    
    x_recon, mu, logvar, z = model(x)
    
    # Test loss
    loss, loss_dict = model.loss_function(x, x_recon, mu, logvar)
    
    print(f"Orthogonal VAE test (beta={beta}, lambda_ortho={lambda_ortho}):")
    print(f"  Total loss: {loss_dict['loss']:.4f}")
    print(f"  Recon loss: {loss_dict['recon']:.4f}")
    print(f"  KL loss: {loss_dict['kl']:.4f}")
    print(f"  Ortho loss: {loss_dict['ortho']:.4f}")
    
    # Check orthogonality
    mu_norm = F.normalize(mu, p=2, dim=0)
    gram = torch.mm(mu_norm.T, mu_norm)
    print(f"\nGram matrix diagonal: {torch.diag(gram).mean().item():.4f} (should be ~1)")
    print(f"Gram matrix off-diag: {(gram.sum() - torch.diag(gram).sum()).abs().item() / (latent_dim * (latent_dim - 1)):.4f} (should be ~0)")
    
    print("\nOrthogonal VAE test passed!")


if __name__ == '__main__':
    test_ortho_vae()