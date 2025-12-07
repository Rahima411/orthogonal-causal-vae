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
import dataclasses

from config import get_config
from data import get_dataset, get_causal_graph
from losses.combined import total_loss

# Evaluation
from experiments.evaluate import evaluate_model

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


# def get_model(name, latent_dim, causal_graph=None):
#     if name not in MODEL_REGISTRY:
#         raise ValueError(f"Unknown model: {name}")
    
#     # Only pass causal_graph to models that support it
#     if name in ['causal_vae', 'ortho_causal_vae']:
#         return MODEL_REGISTRY[name](latent_dim=latent_dim, causal_graph=causal_graph)
#     else:
#         return MODEL_REGISTRY[name](latent_dim=latent_dim)

def get_model(name, latent_dim, causal_graph=None, img_channels=1):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    ModelClass = MODEL_REGISTRY[name]

    if name in ['causal_vae', 'ortho_causal_vae']:
        return ModelClass(latent_dim=latent_dim, img_channels=img_channels, causal_graph=causal_graph)
    else:
        return ModelClass(latent_dim=latent_dim, img_channels=img_channels)



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

        if cfg.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.gradient_clip
            )

        optimizer.step()

        total += loss.item()
        for k in metrics:
            if k != "total":
                metrics_sum[k] += metrics[k]

    avg_loss = total / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dsprites")
    parser.add_argument("--model", type=str, default="ortho_causal_vae")
    parser.add_argument("--epochs", type=int, default=50)
    # Multi-seed support
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="List of seeds to run")
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

    # Enforce pure baseline for non-hybrid models and constraints
    if cfg.model.name in ['vae', 'beta_vae']:
        print(f"\n[INFO] Enforcing pure baseline for {cfg.model.name}: Setting lambda_ortho=0, lambda_causal=0")
        cfg.model.lambda_ortho = 0.0
        cfg.model.lambda_causal = 0.0
    elif cfg.model.name == 'ortho_vae':
        print(f"\n[INFO] Enforcing constraint for {cfg.model.name}: Setting lambda_causal=0")
        cfg.model.lambda_causal = 0.0
    elif cfg.model.name == 'causal_vae':
        print(f"\n[INFO] Enforcing constraint for {cfg.model.name}: Setting lambda_ortho=0")
        cfg.model.lambda_ortho = 0.0

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Initialize WandB (if enabled) - Global init usually for single run, 
    # but for multi-seed we might want one run per seed or grouped.
    # Here we will init inside the loop for per-seed tracking or group usages.
    # Actually, standard practice for multi-seed is one wandb run per seed with a group tag.


    # -------------------------------------------------------------------------
    # Multi-Seed Loop
    # -------------------------------------------------------------------------
    all_results = []
    
    total_seeds = len(args.seeds)
    print(f"\n Starting Multi-Seed Training for {cfg.model.name} on {cfg.data.name}")
    print(f" Seeds: {args.seeds}")
    
    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'='*60}")
        print(f" Run {seed_idx+1}/{total_seeds} | Seed: {seed}")
        print(f"{'='*60}")
        
        # 1. Setup for this run
        cfg.training.seed = seed
        set_seed(seed)
        
        # Save per-seed config
        seed_log_dir = os.path.join(cfg.log_dir, f"seed_{seed}")
        os.makedirs(seed_log_dir, exist_ok=True)
        
        # Dataset
        dataset = get_dataset(cfg.data.name, root="./data", train=True)
        loader = DataLoader(
            dataset, batch_size=cfg.data.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers
        )
        causal_graph = get_causal_graph(cfg.data.name)

        # Model
        # model = get_model(cfg.model.name, cfg.model.latent_dim, causal_graph).to(cfg.training.device)
        model = get_model(cfg.model.name, cfg.model.latent_dim, causal_graph, cfg.model.img_channels).to(cfg.training.device)

        # Optimizer: AdamW
        # Using typical defaults for VAEs on modern hardware
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=1e-5,  # ← changed from 1e-2
            betas=(0.9, 0.999)
        )

        # Initialize WandB for this seed
        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    config=dataclasses.asdict(cfg),
                    name=f"{cfg.model.name}_{cfg.data.name}_seed{seed}",
                    group=f"{cfg.model.name}_{cfg.data.name}",
                    reinit=True
                )
                print(f"WandB initialized for seed {seed}")
            except ImportError:
                print("WandB not installed. Skipping.")
                cfg.use_wandb = False

        # Logger for this seed
        log_csv_path = os.path.join(seed_log_dir, f"{cfg.model.name}_training_log.csv")
        csv_headers = ["Epoch", "Loss", "Recon", "KL", "Ortho", "Causal", "Time"]
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

        # 2. Training Loop
        start_time = time.time()
        for epoch in range(1, cfg.training.epochs + 1):
            epoch_start = time.time()
            avg_loss, metrics = train_epoch(model, loader, optimizer, cfg, causal_graph)
            epoch_time = time.time() - epoch_start
            
            # Print (reduced verborrhea for multi-seed)
            if epoch % 5 == 0 or epoch == 1 or epoch == cfg.training.epochs:
                print(
                    f" Ep {epoch}/{cfg.training.epochs} "
                    f"| L={avg_loss:.2f} R={metrics['recon']:.1f} KL={metrics['kl']:.1f} "
                    f"O={metrics['ortho']:.2f} C={metrics['causal']:.2f} "
                    f"| {epoch_time:.1f}s"
                )
            
            # Log
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

        # 3. Final Evaluation
        final_model_path = os.path.join(seed_log_dir, f"{cfg.model.name}_final_model.pt")
        torch.save(model.state_dict(), final_model_path)

        figures_dir = os.path.join("figures", f"{seed}_{cfg.model.name}")
        os.makedirs(figures_dir, exist_ok=True)
        print(f"Figures → {figures_dir}")
        
        print("\n Running Final Verification & Evaluation...")
        eval_metrics = evaluate_model(
                            model, loader, cfg.training.device,
                            compute_disentanglement=True,
                            save_figures_dir=figures_dir
                        )
        
        # Combine final train metrics with evaluation metrics
        result_entry = {
            "seed": seed,
            "final_train_loss": avg_loss,
            **eval_metrics
        }
        all_results.append(result_entry)
        
        # Save individual run results
        with open(os.path.join(seed_log_dir, f"{cfg.model.name}_eval_results.json"), 'w') as f:
            json.dump(result_entry, f, indent=4)
            
        print(f" Completed Seed {seed}. Results saved.")

        # Finish WandB run for this seed
        if cfg.use_wandb:
            import wandb
            wandb.log(eval_metrics)
            for fig_file in os.listdir(figures_dir):
                wandb.log({f"figures/{fig_file}": wandb.Image(os.path.join(figures_dir, fig_file))})
            wandb.finish()

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" MULTI-SEED AGGREGATION RESULTS")
    print(f"{'='*60}")
    
    summary = {}
    metric_keys = all_results[0].keys()
    
    print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 46)
    
    for k in metric_keys:
        if k == "seed": continue
        values = [r[k] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[k] = {"mean": float(mean_val), "std": float(std_val)}
        
        print(f"{k:<20} | {mean_val:<10.4f} | {std_val:<10.4f}")

    # Save summary
    summary_path = os.path.join(cfg.log_dir, f"summary_{cfg.model.name}_{cfg.data.name}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
