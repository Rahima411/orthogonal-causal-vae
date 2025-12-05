import torch
import torch.nn.functional as F

def mse_reconstruction(x_recon, x):
    return F.mse_loss(x_recon, x, reduction='mean')

def bce_reconstruction(x_recon, x):
    return F.binary_cross_entropy(x_recon, x, reduction='mean')
