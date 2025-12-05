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
