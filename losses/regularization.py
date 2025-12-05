# losses/regularization.py

import torch
import torch.nn.functional as F


def kl_divergence(mu, logvar):
    """
    KL divergence between posterior N(mu, sigma) and prior N(0, 1).
    Standard VAE regularization.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def orthogonality_loss(mu):
    """
    Enforce independence by pushing latent dimensions to be orthogonal.
    Based on Cha et al., ICML 2023.

    Uses Gram matrix penalty:
        || G - I ||_F^2
    where G = normalized(mu)^T normalized(mu)
    """
    # Normalize each latent dimension over the batch
    mu_norm = F.normalize(mu, dim=0, eps=1e-8)

    # Gram matrix
    gram = torch.mm(mu_norm.T, mu_norm)

    # Identity matrix
    identity = torch.eye(mu.size(1), device=mu.device)

    # Frobenius norm penalty
    return torch.norm(gram - identity, p="fro") ** 2


def causal_independence_loss(mu, causal_graph):
    """
    Penalize correlation between latent *root* variables.
    Encourages causal separation.

    causal_graph example:
        {0: [], 1: [], 2: [0], 3: [0,1]}
    roots => 0,1
    """
    if causal_graph is None or len(causal_graph) == 0:
        return torch.tensor(0.0, device=mu.device)

    # Identify root nodes (corrected implementation)
    roots = [k for k, parents in causal_graph.items() if len(parents) == 0]

    if len(roots) < 2:
        return torch.tensor(0.0, device=mu.device)

    # Extract only the root latent dimensions
    root_latents = mu[:, roots]

    # Compute correlation matrix
    corr = torch.corrcoef(root_latents.T)

    # Penalize non-zero off-diagonal correlations
    mask = 1 - torch.eye(len(roots), device=mu.device)

    return torch.sum(torch.abs(corr * mask))
