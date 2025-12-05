import torch
from models.ortho_causal_vae import OrthoCausalVAE
from losses.combined import total_loss

model = OrthoCausalVAE(latent_dim=10).cuda()
x = torch.randn(8, 1, 64, 64).cuda()

x_recon, mu, logvar, z = model(x)
loss, metrics = total_loss(x_recon, x, mu, logvar)

print("Forward OK. Loss:", metrics)
loss.backward()
print("Backward OK.")
