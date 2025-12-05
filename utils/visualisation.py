import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def latent_traversal(model, device, latent_dim, steps=10, std=2.0,
                     save_dir="../figures/traversals"):
    os.makedirs(save_dir, exist_ok=True)

    base = torch.zeros((1, latent_dim)).to(device)

    for d in range(latent_dim):
        imgs = []
        for s in np.linspace(-std, std, steps):
            z = base.clone()
            z[0, d] = s
            img = model.decode(z).cpu().detach().numpy()[0, 0]
            imgs.append(img)

        grid = np.hstack(imgs)
        plt.imshow(grid, cmap="gray")
        plt.axis("off")
        plt.title(f"Latent Dimension {d} Traversal")
        plt.savefig(os.path.join(save_dir, f"latent_{d}.png"))
        plt.close()


def causal_intervention(model, x, dim, shifts=[-2, -1, 0, 1, 2],
                        device="cpu", save_dir="../figures/interventions"):

    os.makedirs(save_dir, exist_ok=True)

    x = x.to(device)
    with torch.no_grad():
        mu, _ = model.encoder(x)

    imgs = []
    for s in shifts:
        z = mu.clone()
        z[:, dim] += s
        img = model.decode(z).cpu()[0, 0]
        imgs.append(img)

    grid = np.hstack(imgs)
    plt.imshow(grid, cmap='gray')
    plt.title(f"Causal Intervention on dim {dim}")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"intervention_dim_{dim}.png"))
    plt.close()
