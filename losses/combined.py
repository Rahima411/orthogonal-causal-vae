import torch

from .reconstruction import mse_loss, bce_loss
from .regularization import (
    kl_divergence,
    orthogonality_loss,
    causal_independence_loss
)


def reconstruction_loss(x_recon, x, loss_type="bce"):

    if loss_type == "bce":
        return bce_loss(x_recon, x)
    elif loss_type == "mse":
        return mse_loss(x_recon, x)
    else:
        raise ValueError("Unknown loss type.")


def total_loss(
    x_recon, x, mu, logvar, causal_graph=None,
    beta=4.0, lambda_ortho=1e-3, lambda_causal=1e-3
):

    # AUTO-SELECT RECON LOSS
    if x.max() > 1.0:
        recon = reconstruction_loss(x_recon, x, "mse")
    else:
        recon = reconstruction_loss(x_recon, x, "bce")

    kl = kl_divergence(mu, logvar)
    if lambda_ortho > 0:
        ortho = orthogonality_loss(mu)
    else:
        ortho = torch.tensor(0.0, device=mu.device)

    if lambda_causal > 0:
        causal = causal_independence_loss(mu, causal_graph)
    else:
        causal = torch.tensor(0.0, device=mu.device)

    loss = recon + beta * kl + lambda_ortho * ortho + lambda_causal * causal

    metrics = {
        "recon": recon.item(),
        "kl": kl.item(),
        "ortho": ortho.item(),
        "causal": causal.item(),
        "total": loss.item()
    }

    return loss, metrics
