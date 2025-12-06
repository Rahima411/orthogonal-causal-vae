# import torch
# import numpy as np
# import logging
# from tqdm import tqdm

# from metrics.disentanglement import compute_mig, compute_sap, compute_dci
# from metrics.reconstruction import mse_reconstruction, bce_reconstruction, compute_ssim
# from metrics.latent_analysis import compute_pca_explained_variance, compute_correlation_score

# def evaluate_model(model, loader, device, compute_disentanglement=True):
#     """
#     Evaluate model on full dataset and return dictionary of metrics.
#     """
#     model.eval()
    
#     # Containers
#     all_mu = []
#     all_factors = []
    
#     # Running metrics
#     total_mse = 0.0
#     total_ssim = 0.0
#     total_samples = 0
    
#     # 1. Collect Data & Compute Batch Metrics
#     print("Evaluating (streaming metrics)...")
#     with torch.no_grad():
#         for x, factors in tqdm(loader, desc="Evaluating"):
#             x = x.to(device)
#             batch_size = x.size(0)
            
#             # BetaVAE/OrthoVAE return: x_recon, mu, logvar, z
#             x_recon, mu, _, _ = model(x)
            
#             # Compute Reconstruction Metrics Batch-wise
#             # MSE
#             batch_mse = mse_reconstruction(x_recon, x).item()
#             total_mse += batch_mse * batch_size
            
#             # SSIM
#             batch_ssim = compute_ssim(x_recon, x, size_average=True)
#             total_ssim += batch_ssim * batch_size
            
#             total_samples += batch_size
            
#             # Store latents for disentanglement (lightweight)
#             all_mu.append(mu.cpu())
#             all_factors.append(factors)
            
#     # Concatenate latents
#     all_mu = torch.cat(all_mu, dim=0)
#     all_factors = torch.cat(all_factors, dim=0)
    
#     # Convert to numpy for certain metrics
#     latents_np = all_mu.numpy()
#     factors_np = all_factors.numpy()
    
#     metrics = {}
    
#     # 2. Average Reconstruction Metrics
#     metrics['mse'] = total_mse / total_samples
#     metrics['ssim'] = total_ssim / total_samples
#     print(f"Reconstruction Metrics Computed: MSE={metrics['mse']:.4f}, SSIM={metrics['ssim']:.4f}")
    
#     # 3. Latent Space Analysis
#     print("Computing Latent Analysis metrics...")
#     metrics['correlation_score'] = compute_correlation_score(latents_np)
    
#     # PCA: explained variance by top 5 components
#     pca_vars = compute_pca_explained_variance(latents_np)
#     metrics['pca_var_top3'] = float(np.sum(pca_vars[:3]))
#     metrics['pca_var_top5'] = float(np.sum(pca_vars[:5]))
    
#     # 4. Disentanglement Metrics (Expensive)
#     if compute_disentanglement:
#         print("Computing Disentanglement metrics (MIG, SAP, DCI)... this may take awhile")
#         try:
#             metrics['mig'] = compute_mig(latents_np, factors_np)
#             metrics['sap'] = compute_sap(latents_np, factors_np)
#             metrics['dci'] = compute_dci(latents_np, factors_np)
#         except Exception as e:
#             print(f"Error computing disentanglement metrics: {e}")
#             metrics['mig'] = -1.0
#             metrics['sap'] = -1.0
#             metrics['dci'] = -1.0
            
#     return metrics

# experiments/evaluate.py
import torch
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # Kill all sklearn warnings
import os

from metrics.disentanglement import compute_mig, compute_sap, compute_dci
from metrics.reconstruction import mse_reconstruction, compute_ssim
from metrics.latent_analysis import (
    compute_pca_explained_variance,
    compute_correlation_score,
    latent_covariance,
    latent_svd
)

def evaluate_model(model, loader, device, compute_disentanglement=True, save_figures_dir=None):
    """
    Full evaluation with clear printouts and optional figure saving
    """
    model.eval()
    
    all_mu = []
    all_factors = []
    
    total_mse = 0.0
    total_ssim = 0.0
    total_samples = 0

    print("Evaluating model and collecting latents...")
    with torch.no_grad():
        for x, factors in tqdm(loader, desc="Collecting latents", leave=False):
            x = x.to(device)
            batch_size = x.size(0)
            
            x_recon, mu, _, _ = model(x)
            
            # Reconstruction metrics
            total_mse += mse_reconstruction(x_recon, x).item() * batch_size
            total_ssim += compute_ssim(x_recon, x, size_average=True) * batch_size
            total_samples += batch_size
            
            # Store for analysis (lightweight only)
            all_mu.append(mu.cpu())
            all_factors.append(factors)

    # Concatenate
    all_mu = torch.cat(all_mu, dim=0)
    all_factors = torch.cat(all_factors, dim=0)
    
    latents_np = all_mu.numpy()
    factors_np = all_factors.numpy()

    metrics = {}
    
    # === Reconstruction ===
    metrics['mse'] = total_mse / total_samples
    metrics['ssim'] = total_ssim / total_samples
    print(f"Reconstruction → MSE: {metrics['mse']:.6f} | SSIM: {metrics['ssim']:.4f}")

    # === Latent Analysis ===
    print("Computing latent space analysis...")
    metrics['correlation_score'] = compute_correlation_score(latents_np)
    print(f"   Avg off-diagonal correlation: {metrics['correlation_score']:.4f}")

    pca_vars = compute_pca_explained_variance(latents_np)
    metrics['pca_var_top3'] = float(np.sum(pca_vars[:3]))
    metrics['pca_var_top5'] = float(np.sum(pca_vars[:5]))
    print(f"   PCA Top-3 explained variance: {metrics['pca_var_top3']:.4f}")
    print(f"   PCA Top-5 explained variance: {metrics['pca_var_top5']:.4f}")

    # Optional: save covariance & SVD plots
    if save_figures_dir:
        os.makedirs(save_figures_dir, exist_ok=True)
        latent_covariance(latents_np, save_path=os.path.join(save_figures_dir, "covariance.png"))
        latent_svd(latents_np, save_path=os.path.join(save_figures_dir, "svd.png"))

    # === Disentanglement ===
    if compute_disentanglement:
        print("Computing disentanglement metrics (MIG, SAP, DCI)...")
        try:
            metrics['mig'] = compute_mig(latents_np, factors_np)
            metrics['sap'] = compute_sap(latents_np, factors_np)
            metrics['dci'] = compute_dci(latents_np, factors_np)
            print(f"Disentanglement → MIG: {metrics['mig']:.4f} | SAP: {metrics['sap']:.4f} | DCI: {metrics['dci']:.4f}")
        except Exception as e:
            print(f"Disentanglement failed: {e}")
            metrics['mig'] = metrics['sap'] = metrics['dci'] = -1.0

    print("\nEvaluation Complete!\n")
    return metrics