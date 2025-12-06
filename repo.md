```
orthogonal-causal-vae/
│
├── data/
│   ├── __init__.py
│   ├── dsprites.py          # Dataset loader
│   ├── celeba.py
│   └── shapes3d.py
│
├── models/
│   ├── __init__.py
│   ├── vae.py               # Base VAE
│   ├── beta_vae.py          # β-VAE
│   ├── ortho_vae.py         # Orthogonal VAE
│   ├── causal_vae.py        # Causal VAE (simplified ICM)
│   └── ortho_causal_vae.py  # YOUR MODEL
│
├── losses/
│   ├── __init__.py
│   ├── reconstruction.py    # MSE, BCE
│   ├── regularization.py    # KL, orthogonality, causal
│   └── combined.py          # Total loss
│
├── metrics/
│   ├── __init__.py
│   ├── disentanglement.py   # MIG, SAP, DCI
│   ├── reconstruction.py    # MSE, SSIM
│   └── latent_analysis.py   # PCA, correlation
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py     # Latent traversal, PCA plots
│   ├── causal_graph.py      # Causal structure definitions
│   └── helpers.py
│
├── experiments/
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation script
│   └── config.py            # Hyperparameters
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
│
├── checkpoints/             # Saved models
├── logs/                    # Training logs
├── figures/                 # Generated plots
│
├── requirements.txt
├── README.md
```
