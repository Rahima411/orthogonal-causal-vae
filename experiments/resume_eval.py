# experiments/resume_eval.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json
import warnings
warnings.filterwarnings("ignore")

from config import get_config
from data import get_dataset, get_causal_graph
from experiments.train import get_model
from experiments.evaluate import evaluate_model

import dataclasses

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Resume evaluation from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="dsprites", help="Dataset name")
    parser.add_argument("--model", type=str, default="beta_vae", help="Model name")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    # 1. Config
    print(f"Loading config for {args.dataset}...")
    try:
        cfg = get_config(args.dataset)
    except ValueError:
        cfg = get_config('dsprites')
        cfg.data.name = args.dataset
        
    cfg.model.name = args.model
    cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Data
    print("Loading dataset...")
    dataset = get_dataset(cfg.data.name, root="./data", train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    causal_graph = get_causal_graph(cfg.data.name)

    # 3. Model
    print(f"Initializing {args.model}...")
    model = get_model(cfg.model.name, cfg.model.latent_dim, causal_graph, cfg.model.img_channels)
    model.to(cfg.training.device)
    
    # 4. Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=cfg.training.device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # 5. Figures directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    seed_name = os.path.basename(checkpoint_dir)
    figures_dir = os.path.join("figures", f"{seed_name}_{cfg.model.name}")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures → {figures_dir}")

    # 6. WandB Init (optional)
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=dataclasses.asdict(cfg),
                name=f"{cfg.model.name}_{cfg.data.name}_{seed_name}",
                group=f"{cfg.model.name}_{cfg.data.name}",
                reinit=True
            )
            print(f"WandB initialized for seed {seed_name}")
        except ImportError:
            print("WandB not installed. Skipping.")
            cfg.use_wandb = False

    # 7. Evaluate
    print("Starting evaluation...")
    metrics = evaluate_model(
        model, loader, cfg.training.device,
        compute_disentanglement=True,
        save_figures_dir=figures_dir
    )
    
    # 8. Log to WandB
    if cfg.use_wandb:
        wandb.log(metrics)
        # Upload figures
        for fig_file in os.listdir(figures_dir):
            wandb.log({f"figures/{fig_file}": wandb.Image(os.path.join(figures_dir, fig_file))})
        wandb.finish()

    # 9. Save JSON
    output_path = os.path.join(checkpoint_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nEVALUATION COMPLETE!")
    print(f"Results → {output_path}")
    print(f"Figures → {figures_dir}")
    print("\nFINAL METRICS:")
    for k, v in metrics.items():
        print(f"  {k.upper():<20}: {v:.6f}")

if __name__ == "__main__":
    main()