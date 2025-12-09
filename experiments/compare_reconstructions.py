"""
Compare reconstructions from different VAE variants side-by-side
Loads models from seed 42 checkpoints and visualizes reconstructions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import get_config
from data import get_dataset, get_causal_graph
from experiments.train import get_model

def load_model_from_checkpoint(model_name, checkpoint_path, cfg, causal_graph, device):
    """Load a trained model from checkpoint"""
    model = get_model(model_name, cfg.model.latent_dim, causal_graph, cfg.model.img_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def visualize_comparisons(models_dict, loader, device, num_samples=8, save_path="reconstruction_comparison.png"):
    """
    Create side-by-side comparison of reconstructions
    
    Args:
        models_dict: Dictionary of {model_name: model}
        loader: DataLoader for test images
        device: torch device
        num_samples: Number of images to visualize
        save_path: Where to save the figure
    """
    # Get a batch of images
    images, _ = next(iter(loader))
    images = images[:num_samples].to(device)
    
    # Generate reconstructions from each model
    reconstructions = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            recon, _, _, _ = model(images)
            reconstructions[name] = recon.cpu()
    
    images_cpu = images.cpu()
    
    # Create figure
    num_models = len(models_dict)
    fig, axes = plt.subplots(num_samples, num_models + 1, figsize=(3*(num_models + 1), 3*num_samples))
    
    model_names = list(models_dict.keys())
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images_cpu[i, 0], cmap='gray')
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=14, fontweight='bold')
        
        # Reconstructions from each model
        for j, model_name in enumerate(model_names):
            axes[i, j+1].imshow(reconstructions[model_name][i, 0], cmap='gray')
            axes[i, j+1].axis('off')
            if i == 0:
                # Format model name for title
                title = model_name.replace('_', '-').upper()
                axes[i, j+1].set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison figure to: {save_path}")
    plt.close()

def main():
    # Configuration
    seed = 42
    dataset_name = "dsprites"
    num_samples = 8
    
    # Model names and their checkpoint paths
    model_configs = {
        'beta_vae': f'logs/seed_{seed}/beta_vae_final_model.pt',
        'ortho_vae': f'logs/seed_{seed}/ortho_vae_final_model.pt',
        'causal_vae': f'logs/seed_{seed}/causal_vae_final_model.pt',
        'ortho_causal_vae': f'logs/seed_{seed}/ortho_causal_vae_final_model.pt'
    }
    
    # Setup
    cfg = get_config(dataset_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    causal_graph = get_causal_graph(dataset_name)
    
    print(f"Device: {device}")
    print(f"Loading models from seed {seed}...")
    
    # Load all models
    models = {}
    for model_name, checkpoint_path in model_configs.items():
        if os.path.exists(checkpoint_path):
            print(f"  Loading {model_name}...")
            models[model_name] = load_model_from_checkpoint(
                model_name, checkpoint_path, cfg, causal_graph, device
            )
        else:
            print(f"  Warning: Checkpoint not found for {model_name} at {checkpoint_path}")
    
    if not models:
        print("Error: No models loaded. Please check that checkpoints exist.")
        return
    
    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(dataset_name, root="./data", train=True)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    
    # Create visualization
    print(f"Generating comparison visualization...")
    output_path = f"figures/reconstruction_comparison_seed{seed}.png"
    os.makedirs("figures", exist_ok=True)
    
    visualize_comparisons(models, loader, device, num_samples=num_samples, save_path=output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
