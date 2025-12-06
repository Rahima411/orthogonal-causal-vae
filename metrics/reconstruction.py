import torch
import torch.nn.functional as F

def mse_reconstruction(x_recon, x):
    return F.mse_loss(x_recon, x, reduction='mean')

def bce_reconstruction(x_recon, x):
    return F.binary_cross_entropy(x_recon, x, reduction='mean')

import numpy as np

def compute_ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    img1, img2: torch tensors (B, C, H, W) normalized to [0, 1]
    
    Simplified implementation suitable for VAE evaluation.
    """
    channel = img1.size(1)
    
    # Gaussian window generation
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    if img1.size(1) == 1: # Grayscale
        window = create_window(window_size, 1).to(img1.device)
    else:
        window = create_window(window_size, channel).to(img1.device)
        
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
