import numpy as np
import matplotlib.pyplot as plt

def latent_covariance(latents, save_path=None):
    cov = np.cov(latents.T)
    if save_path:
        plt.figure(figsize=(8, 6))
        plt.imshow(cov, cmap="coolwarm")
        plt.colorbar()
        plt.title("Latent Covariance Matrix")
        plt.savefig(save_path, dpi=300)
        plt.close()
    return cov


def latent_svd(latents, save_path=None):
    U, S, Vt = np.linalg.svd(latents - latents.mean(0), full_matrices=False)
    if save_path:
        plt.figure()
        plt.plot(S, marker='o')
        plt.title("Latent Singular Values")
        plt.savefig(save_path, dpi=300)
        plt.close()
    return S

import torch
from sklearn.decomposition import PCA

def compute_pca_explained_variance(latents, n_components=None):
    """
    Compute cumulative explained variance ratio by principal components of the latent space.
    latents: (N, L) numpy array
    """
    if n_components is None:
        n_components = min(latents.shape)
        
    pca = PCA(n_components=n_components)
    pca.fit(latents)
    
    # Return array of explained variance ratios
    return pca.explained_variance_ratio_


def compute_correlation_score(latents):
    """
    Compute the average absolute off-diagonal correlation.
    Lower is better (more disentangled/independent).
    latents: (N, L) numpy array
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
        
    corr_matrix = np.corrcoef(latents, rowvar=False)
    
    # Mask diagonal
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    
    # Average absolute correlation of off-diagonal elements
    if mask.sum() == 0:
        return 0.0
        
    avg_abs_corr = np.mean(np.abs(corr_matrix[mask]))
    return float(avg_abs_corr)
