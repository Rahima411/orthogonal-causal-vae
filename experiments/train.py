# experiments/train.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import random
import numpy as np
import csv
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import time

from config import get_config
from data import get_dataset, get_causal_graph
from losses.combined import total_loss

# Import models
from models.vae import VAE
from models.beta_vae import BetaVAE
from models.ortho_vae import OrthoVAE
from models.causal_vae import CausalVAE
from models.ortho_causal_vae import OrthoCausalVAE


MODEL_REGISTRY = {
    "vae": VAE,
    "beta_vae": BetaVAE,
    "ortho_vae": OrthoVAE,
    "causal_vae": CausalVAE,
    "ortho_causal_vae": OrthoCausalVAE,
}


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_model(name, latent_dim, causal_graph=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](latent_dim=latent_dim, causal_graph=causal_graph)


def train_epoch(model, loader, optimizer, cfg, causal_graph):
    model.train()
    total = 0
    metrics_sum = {"recon": 0, "kl": 0, "ortho": 0, "causal": 0}

    for x, factors in loader:
        x = x.to(cfg.training.device)

        x_recon, mu, logvar, z = model(x)

        loss, metrics = total_loss(
            x_recon, x, mu, logvar,
            causal_graph=causal_graph,
            beta=cfg.model.beta,
            lambda_ortho=cfg.model.lambda_ortho,
            lambda_causal=cfg.model.lambda_causal
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        for k in metrics:
            if k != "total":
                metrics_sum[k] += metrics[k]

    avg_loss = total / len(loader.dataset)
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dsprites")
    parser.add_argument("--model", type=str, default="ortho_causal_vae")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Get configuration based on dataset
    try:
        cfg = get_config(args.dataset)
    except ValueError:
        # Fallback to dsprites config but override name if dataset is not in presets
        cfg = get_config('dsprites')
        cfg.data.name = args.dataset

    # Override config with command line arguments
    if args.model:
        cfg.model.name = args.model
    if args.epochs:
        cfg.training.epochs = args.epochs
    
    # Ensure device is set correcty
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed
    set_seed(cfg.training.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print(f"\n Training {cfg.model.name} on {cfg.data.name} for {cfg.training.epochs} epochs\n")

    # Save Config
    config_path = os.path.join(cfg.log_dir, f"config_{cfg.model.name}_{cfg.data.name}.json")
    with open(config_path, 'w') as f:
        # Simple helper to convert config to dict for JSON serialization
        # This is a basic serialization, for complex objects use a library
        import dataclasses
        def asdict_factory(data):
            # Filtering out callables/lambdas if any (though we removed them)
            return {k: v for k, v in data}
        json.dump(dataclasses.asdict(cfg, dict_factory=asdict_factory), f, indent=4)
        print(f"Config saved to {config_path}")

    # Initialize Logger (CSV)
    log_csv_path = os.path.join(cfg.log_dir, f"log_{cfg.model.name}_{cfg.data.name}.csv")
    csv_headers = ["Epoch", "Loss", "Recon", "KL", "Ortho", "Causal", "Time"]
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # Initialize WandB
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=dataclasses.asdict(cfg),
                name=f"{cfg.model.name}_{cfg.data.name}"
            )
            print("WandB initialized.")
        except ImportError:
            print("WandB not installed. Skipping.")
            cfg.use_wandb = False
    
    # Dataset
    dataset = get_dataset(cfg.data.name, root="./data", train=True)
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers
    )

    # Causal graph
    causal_graph = get_causal_graph(cfg.data.name)

    # Model
    model = get_model(cfg.model.name, cfg.model.latent_dim, causal_graph).to(cfg.training.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    start_time = time.time()

    for epoch in range(1, cfg.training.epochs + 1):

        epoch_start = time.time()

        avg_loss, metrics = train_epoch(model, loader, optimizer, cfg, causal_graph)

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        epochs_left = cfg.training.epochs - epoch
        eta = epochs_left * epoch_time

        print(
            f"Epoch {epoch}/{cfg.training.epochs} "
            f"| Time: {epoch_time:.2f}s "
            f"| ETA: {eta/60:.1f} min "
            f"| Loss={avg_loss:.4f} "
            f"| Recon={metrics['recon']:.2f} KL={metrics['kl']:.2f} "
            f"| Ortho={metrics['ortho']:.4f} Causal={metrics['causal']:.4f}"
        )
        
        # Log to CSV
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_loss, metrics['recon'], metrics['kl'], 
                metrics['ortho'], metrics['causal'], epoch_time
            ])

        # Log to WandB
        if cfg.use_wandb:
            wandb.log({
                "epoch": epoch,
                "loss": avg_loss,
                "recon_loss": metrics['recon'],
                "kl_loss": metrics['kl'],
                "ortho_loss": metrics['ortho'],
                "causal_loss": metrics['causal'],
                "epoch_time": epoch_time
            })

        # Save checkpoint
        if epoch % cfg.training.save_interval == 0:
            path = f"{cfg.checkpoint_dir}/{cfg.model.name}_{cfg.data.name}_epoch{epoch}.pt"
            torch.save(model.state_dict(), path)
            print(f" Saved: {path}")

if __name__ == "__main__":
    main()
