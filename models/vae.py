"""
Base Variational Autoencoder (VAE) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional encoder for images
    Maps image to latent mean and log-variance
    """
    
    def __init__(self, latent_dim=10, img_channels=1, hidden_dims=None):
        """
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            hidden_dims: List of hidden dimensions for conv layers
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 32, 64, 64]
        
        # Build encoder layers
        modules = []
        in_channels = img_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size after convolutions
        # Assuming input image is 64x64, after 4 stride-2 convs: 64 -> 32 -> 16 -> 8 -> 4
        self.flatten_size = hidden_dims[-1] * 4 * 4
        
        # Latent projection layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log-variance [batch_size, latent_dim]
        """
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        # Project to latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Convolutional decoder for images
    Maps latent code to reconstructed image
    """
    
    def __init__(self, latent_dim=10, img_channels=1, hidden_dims=None):
        """
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of output image channels
            hidden_dims: List of hidden dimensions for deconv layers (reversed from encoder)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 32, 32]
        
        # Project latent to spatial feature map
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        
        # Build decoder layers
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU()
                )
            )
        
        # Final layer to image
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], img_channels,
                                 kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        """
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed images [batch_size, channels, height, width]
        """
        # Project and reshape
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        
        # Decode
        x_recon = self.decoder(h)
        
        return x_recon


class VAE(nn.Module):
    """
    Vanilla Variational Autoencoder
    """
    
    def __init__(self, latent_dim=10, img_channels=1, hidden_dims=None):
        """
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of image channels
            hidden_dims: Hidden dimensions for encoder/decoder
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        
        # Encoder and decoder
        self.encoder = Encoder(latent_dim, img_channels, hidden_dims)
        
        # Decoder uses reversed hidden dims
        decoder_hidden_dims = hidden_dims[::-1] if hidden_dims else None
        self.decoder = Decoder(latent_dim, img_channels, decoder_hidden_dims)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            z: Sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x: Input images
            
        Returns:
            x_recon: Reconstructed images
            mu: Latent means
            logvar: Latent log-variances
            z: Sampled latent codes
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar, z
    
    def encode(self, x):
        """
        Encode images to latent mean (deterministic)
        
        Args:
            x: Input images
            
        Returns:
            mu: Latent means
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """
        Decode latent codes to images
        
        Args:
            z: Latent codes
            
        Returns:
            x_recon: Reconstructed images
        """
        return self.decoder(z)
    
    def sample(self, num_samples, device):
        """
        Sample from prior and decode
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            x_samples: Generated images
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        x_samples = self.decode(z)
        return x_samples


def test_vae():
    """Test VAE architecture"""
    batch_size = 8
    img_channels = 1
    img_size = 64
    latent_dim = 10
    
    # Create model
    model = VAE(latent_dim=latent_dim, img_channels=img_channels)
    
    # Test forward pass
    x = torch.randn(batch_size, img_channels, img_size, img_size)
    x_recon, mu, logvar, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Latent z shape: {z.shape}")
    
    # Test sampling
    samples = model.sample(num_samples=4, device='cpu')
    print(f"Generated samples shape: {samples.shape}")
    
    print("VAE test passed!")


if __name__ == '__main__':
    test_vae()