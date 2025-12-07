"""
Configuration file for experiments
Centralized hyperparameter management
"""

import torch
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Dataset configuration"""
    name: str = 'dsprites'  # 'dsprites', 'celeba', 'shapes3d', 'mnist'
    root: str = './data'
    batch_size: int = 128
    # Disable multiprocessing on Windows to avoid OSError
    num_workers: int = 0 if os.name == 'nt' else 4
    img_size: int = 64
    train_split: float = 0.8


def default_hidden_dims():
    return [32, 32, 64, 64]


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = 'ortho_causal_vae'  # 'vae', 'beta_vae', 'ortho_vae', 'causal_vae', 'ortho_causal_vae'
    latent_dim: int = 10
    img_channels: int = 1  # 1 for grayscale (dsprites), 3 for RGB (celeba, shapes3d)
    hidden_dims: List[int] = field(default_factory=default_hidden_dims)
    
    # β-VAE parameters
    beta: float = 4.0
    
    # Orthogonal VAE parameters
    lambda_ortho: float = 1.0
    
    # Causal VAE parameters
    lambda_causal: float = 0.5
    use_causal_graph: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = 'cosine'  # 'none', 'step', 'cosine'
    scheduler_params: Dict = field(default_factory=dict)
    
    # Optimization
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    gradient_clip: float = 1.0  # 0 = no clipping
    
    # Logging
    log_interval: int = 100  # Log every N batches
    save_interval: int = 10  # Save checkpoint every N epochs
    
    # Evaluation
    eval_interval: int = 1  # Evaluate every N epochs
    num_samples_to_save: int = 8  # Number of reconstructions to save
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42



@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    compute_mig: bool = True
    compute_sap: bool = True
    compute_dci: bool = False  # Computationally expensive
    
    # Visualization
    save_reconstructions: bool = True
    save_samples: bool = True
    save_traversals: bool = True
    traversal_range: float = 3.0  # Range for latent traversal (-range, +range)
    traversal_steps: int = 10


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Experiment metadata
    name: str = 'ortho_causal_vae_dsprites'
    description: str = 'Orthogonal Causal β-VAE on dSprites'
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    fig_dir: str = './figures'
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Wandb logging (optional)
    use_wandb: bool = True
    wandb_project: str = 'ortho-causal-vae'
    wandb_entity: Optional[str] = None


# Predefined configurations for different experiments

def get_dsprites_config():
    """Configuration for dSprites experiments"""
    config = ExperimentConfig(
        name='ortho_causal_vae_dsprites',
        description='Orthogonal Causal β-VAE on dSprites dataset',
    )
    
    config.data.name = 'dsprites'
    config.data.batch_size = 128
    config.model.img_channels = 1
    config.model.latent_dim = 10
    config.model.beta = 4.0
    # config.model.lambda_ortho = 1.0
    # config.model.lambda_causal = 0.5
    config.model.lambda_ortho = 0.5 
    config.model.lambda_causal = 0.3 
    
    return config


def get_celeba_config():
    """Configuration for CelebA experiments"""
    config = ExperimentConfig(
        name='ortho_causal_vae_celeba',
        description='Orthogonal Causal β-VAE on CelebA dataset',
    )
    
    config.data.name = 'celeba'
    config.data.batch_size = 64  # Larger images, smaller batch
    config.model.img_channels = 3
    config.model.latent_dim = 10
    config.model.hidden_dims = [32, 64, 128, 256]  # Deeper for RGB
    config.model.beta = 4.0
    config.model.lambda_ortho = 1.0
    config.model.lambda_causal = 0.5
    
    config.training.epochs = 100  # More epochs for complex data
    config.training.learning_rate = 5e-4  # Lower LR
    
    return config


def get_shapes3d_config():
    """Configuration for 3D Shapes experiments"""
    config = ExperimentConfig(
        name='ortho_causal_vae_shapes3d',
        description='Orthogonal Causal β-VAE on 3D Shapes dataset',
    )
    
    config.data.name = 'shapes3d'
    config.data.batch_size = 64
    config.model.img_channels = 3
    config.model.latent_dim = 10
    config.model.hidden_dims = [32, 64, 128, 256]
    config.model.beta = 4.0
    config.model.lambda_ortho = 1.0
    config.model.lambda_causal = 0.5
    
    return config


def get_baseline_beta_vae_config():
    """Configuration for baseline β-VAE (no orthogonality or causality)"""
    config = get_dsprites_config()
    config.name = 'beta_vae_baseline_dsprites'
    config.model.name = 'beta_vae'
    config.model.lambda_ortho = 0.0
    config.model.lambda_causal = 0.0
    
    return config


def get_ablation_configs():
    """Get configurations for ablation study"""
    base_config = get_dsprites_config()
    
    configs = {}
    
    # 1. Vanilla β-VAE (no ortho, no causal)
    config_beta = base_config
    config_beta.name = 'ablation_beta_vae'
    config_beta.model.name = 'beta_vae'
    config_beta.model.lambda_ortho = 0.0
    config_beta.model.lambda_causal = 0.0
    configs['beta_vae'] = config_beta
    
    # 2. Orthogonal VAE only (no causal)
    config_ortho = get_dsprites_config()
    config_ortho.name = 'ablation_ortho_vae'
    config_ortho.model.name = 'ortho_vae'
    config_ortho.model.lambda_ortho = 1.0
    config_ortho.model.lambda_causal = 0.0
    configs['ortho_vae'] = config_ortho
    
    # 3. Causal VAE only (no ortho)
    config_causal = get_dsprites_config()
    config_causal.name = 'ablation_causal_vae'
    config_causal.model.name = 'causal_vae'
    config_causal.model.lambda_ortho = 0.0
    config_causal.model.lambda_causal = 0.5
    configs['causal_vae'] = config_causal
    
    # 4. Full model (ortho + causal)
    config_full = get_dsprites_config()
    config_full.name = 'ablation_ortho_causal_vae'
    config_full.model.name = 'ortho_causal_vae'
    config_full.model.lambda_ortho = 1.0
    config_full.model.lambda_causal = 0.5
    configs['ortho_causal_vae'] = config_full
    
    return configs


# Quick access to configs
CONFIGS = {
    'dsprites': get_dsprites_config,
    'celeba': get_celeba_config,
    'shapes3d': get_shapes3d_config,
    'baseline': get_baseline_beta_vae_config,
}


def get_config(name='dsprites'):
    """
    Get configuration by name
    
    Args:
        name: Configuration name
        
    Returns:
        ExperimentConfig object
    """
    if name in CONFIGS:
        return CONFIGS[name]()
    else:
        raise ValueError(f"Unknown config: {name}. Choose from: {list(CONFIGS.keys())}")


if __name__ == '__main__':
    # Test configurations
    print("Testing configurations...")
    
    config = get_dsprites_config()
    print(f"\ndSprites config:")
    print(f"  Dataset: {config.data.name}")
    print(f"  Model: {config.model.name}")
    print(f"  Latent dim: {config.model.latent_dim}")
    print(f"  Beta: {config.model.beta}")
    print(f"  Lambda ortho: {config.model.lambda_ortho}")
    print(f"  Lambda causal: {config.model.lambda_causal}")
    
    ablation_configs = get_ablation_configs()
    print(f"\nAblation study configs: {list(ablation_configs.keys())}")
    
    print("\nConfiguration test passed!")