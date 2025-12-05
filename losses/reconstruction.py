# losses/reconstruction.py

import torch
import torch.nn.functional as F


def mse_loss(x_recon, x):
    """
    Mean Squared Error reconstruction loss.
    Suitable for continuous-valued images (e.g., CelebA, Shapes3D).
    """
    return F.mse_loss(x_recon, x, reduction="sum")


def bce_loss(x_recon, x):
    """
    Binary Cross-Entropy reconstruction loss.
    Suitable for binary or near-binary datasets (e.g., dSprites).
    """
    return F.binary_cross_entropy(x_recon, x, reduction="sum")
