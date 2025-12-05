"""
Causal VAE: VAE with causal structure awareness
Simplified version inspired by ICM-VAE (IJCAI 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .beta_vae import BetaVAE


class CausalVAE(BetaVAE):
    """
    VAE that respects known causal structure between latent factors
    
    Enforces independence between root causal variables
    """
    
    def __init__(self, latent_dim=10, img_channels=1, hidden_dims=None,
                 beta=4.0, lambda_causal=0.5, causal_graph=None):
        """
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of image channels
            hidden_dims: Hidden dimensions for encoder/decoder
            beta: Weight for KL divergence
            lambda_causal: Weight for causal independence loss
            causal_graph: Dictionary mapping latent index to parent indices
                         e.g., {0: [], 1: [], 2: [0], 3: [0, 1]}
                         means 0, 1 are roots; 2 depends on 0; 3 depends on 0, 1
        """
        super().__init__(latent_dim, img_channels, hidden_dims, beta)
        
        self.lambda_causal = lambda_causal
        self.causal_graph = causal_graph or {}
        
        # Identify root nodes (no parents)
        self.root_indices = [i for i in range(latent_dim) 
                            if i not in self.causal_graph or len(self.causal_graph.get(i, [])) == 0]
    
    def causal_independence_loss(self, mu):
        """
        Encourage causal root factors to be independent
        
        Penalizes correlation between root latent dimensions
        
        Args:
            mu: Latent means [batch_size, latent_dim]
            
        Returns:
            causal_loss: Independence loss for root factors
        """
        if len(self.root_indices) < 2:
            # Need at least 2 roots for independence
            return torch.tensor(0.0, device=mu.device)
        
        # Extract root latents
        root_latents = mu[:, self.root_indices]  # [batch_size, num_roots]
        
        # Compute correlation matrix
        # Center the data
        root_centered = root_latents - root_latents.mean(dim=0, keepdim=True)
        
        # Covariance matrix
        cov = torch.mm(root_centered.T, root_centered) / (root_latents.size(0) - 1)
        
        # Standard deviations
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        
        # Correlation matrix
        corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-8)
        
        # Penalize off-diagonal correlations
        # Create mask for off-diagonal elements
        num_roots = len(self.root_indices)
        mask = 1 - torch.eye(num_roots, device=mu.device)
        
        # Sum of absolute correlations (off-diagonal)
        causal_loss = torch.sum(torch.abs(corr * mask))
        
        return causal_loss
    
    def loss_function(self, x, x_recon, mu, logvar):
        """
        Compute Causal VAE loss: Reconstruction + β * KL + λ * Causal Independence
        
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
        
        # Causal independence loss
        causal_loss = self.causal_independence_loss(mu)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.lambda_causal * causal_loss
        
        # Per-sample losses for logging
        loss_dict = {
            'loss': total_loss.item() / batch_size,
            'recon': recon_loss.item() / batch_size,
            'kl': kl_loss.item() / batch_size,
            'causal': causal_loss.item() / batch_size,
            'beta': self.beta,
            'lambda_causal': self.lambda_causal
        }
        
        return total_loss, loss_dict
    
    def set_lambda_causal(self, lambda_causal):
        """Update causal independence weight"""
        self.lambda_causal = lambda_causal


def test_causal_vae():
    """Test Causal VAE architecture"""
    batch_size = 8
    img_channels = 1
    img_size = 64
    latent_dim = 5
    beta = 4.0
    lambda_causal = 0.5
    
    # Define simple causal graph
    # Factors: 0, 1 are roots; 2 depends on 0; 3, 4 depend on 0, 1
    causal_graph = {
        0: [],
        1: [],
        2: [0],
        3: [0, 1],
        4: [0, 1]
    }
    
    # Create model
    model = CausalVAE(
        latent_dim=latent_dim,
        img_channels=img_channels,
        beta=beta,
        lambda_causal=lambda_causal,
        causal_graph=causal_graph
    )
    
    print(f"Causal graph: {causal_graph}")
    print(f"Root indices: {model.root_indices}")
    
    # Test forward pass
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    x = torch.sigmoid(x)
    
    x_recon, mu, logvar, z = model(x)
    
    # Test loss
    loss, loss_dict = model.loss_function(x, x_recon, mu, logvar)
    
    print(f"\nCausal VAE test (beta={beta}, lambda_causal={lambda_causal}):")
    print(f"  Total loss: {loss_dict['loss']:.4f}")
    print(f"  Recon loss: {loss_dict['recon']:.4f}")
    print(f"  KL loss: {loss_dict['kl']:.4f}")
    print(f"  Causal loss: {loss_dict['causal']:.4f}")
    
    # Check correlation of root factors
    root_latents = mu[:, model.root_indices]
    corr = torch.corrcoef(root_latents.T)
    print(f"\nRoot factor correlations:")
    print(corr)
    
    print("\nCausal VAE test passed!")


if __name__ == '__main__':
    test_causal_vae()