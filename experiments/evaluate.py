# experiments/evaluate.py

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

from data import get_dataset, get_causal_graph
from metrics.disentanglement import compute_mig, compute_sap
from models.ortho_causal_vae import OrthoCausalVAE


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dsprites")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔍 Evaluating model on {args.dataset}")

    dataset = get_dataset(args.dataset, train=False)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = OrthoCausalVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_latents, all_recons, all_imgs, all_factors = [], [], [], []

    with torch.no_grad():
        for x, factors in loader:
            x = x.to(device)

            x_recon, mu, _, _ = model(x)
            all_latents.append(mu.cpu().numpy())
            all_recons.append(x_recon.cpu().numpy())
            all_imgs.append(x.cpu().numpy())
            all_factors.append(factors.numpy())

    latents = np.concatenate(all_latents)
    recons = np.concatenate(all_recons)
    imgs = np.concatenate(all_imgs)
    factors = np.concatenate(all_factors)

    # Reconstruction MSE
    mse = np.mean((imgs - recons) ** 2)

    # MIG / SAP
    try:
        mig = compute_mig(latents, factors)
        sap = compute_sap(latents, factors)
    except Exception:
        mig, sap = None, None

    # PCA visualization
    pca = PCA(n_components=2)
    l2d = pca.fit_transform(latents)

    plt.scatter(l2d[:, 0], l2d[:, 1], s=1)
    plt.title("Latent PCA")
    plt.savefig("./figures/latent_pca.png", dpi=300)

    # Correlation matrix
    corr = np.corrcoef(latents.T)
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Latent Correlation Matrix")
    plt.colorbar()
    plt.savefig("./figures/latent_corr.png", dpi=300)

    results = {
        "mse": float(mse),
        "mig": float(mig) if mig is not None else None,
        "sap": float(sap) if sap is not None else None,
        "latent_variance_pca": pca.explained_variance_ratio_[:2].tolist(),
    }

    with open("./figures/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n Evaluation Results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
