# models/ortho_causal_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim=10, img_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=10, img_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, 2, 1)

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        return torch.sigmoid(self.deconv4(h))


class OrthoCausalVAE(nn.Module):
    def __init__(self, latent_dim=10, img_channels=1, causal_graph=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.causal_graph = causal_graph
        self.encoder = Encoder(latent_dim, img_channels)
        self.decoder = Decoder(latent_dim, img_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)