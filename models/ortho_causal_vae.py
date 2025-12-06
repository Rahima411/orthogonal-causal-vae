# models/ortho_causal_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=10, img_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)

        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=10, img_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, 2, 1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        x_recon = torch.sigmoid(self.deconv4(h))
        return x_recon


# ---------------------------------------------------------
# Ortho-Causal β-VAE (with orthogonalization + causal deconfounding)
# ---------------------------------------------------------
class OrthoCausalVAE(nn.Module):
    """
    VAE with optional batch-wise orthogonalization of latent dimensions
    and causal deconfounding (residualization of child latents w.r.t parents).

    - orthogonalize: if True, apply symmetric whitening on posterior means mu
      to produce z_ortho (decorrelates columns/latent dimensions across the batch).
    - causal_deconfound: if True, for each child latent that has parents in the
      causal_graph, replace its value with the residual of a linear regression
      on the parent latents (removes linear dependence on parents).
    """

    def __init__(
        self,
        latent_dim=10,
        img_channels=1,
        causal_graph=None,
        orthogonalize=True,
        causal_deconfound=True,
        whiten_eps=1e-6,
        ridge=1e-3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.causal_graph = causal_graph or {}
        self.orthogonalize_flag = orthogonalize
        self.causal_deconfound_flag = causal_deconfound
        self.whiten_eps = whiten_eps
        self.ridge = ridge

        self.encoder = Encoder(latent_dim, img_channels)
        self.decoder = Decoder(latent_dim, img_channels)

        # diagnostics (updated each forward)
        self.last_mu = None
        self.last_z = None
        self.last_z_ortho = None
        self.last_z_causal = None
        self.last_cov = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------------------------
    # Batch symmetric whitening (decorrelate latent dims across batch)
    # ---------------------------
    def _whiten_latents(self, mu):
        """
        Given mu (B, L), return mu_whitened (B, L) such that cov(mu_whitened) ~= I
        Uses eigen-decomposition of covariance: C = V S V^T, transform by C^{-1/2}.

        This is a differentiable linear transform (but note numerical issues for tiny eigenvalues).
        """
        # center
        mu_centered = mu - mu.mean(dim=0, keepdim=True)  # (B, L)
        B = mu_centered.shape[0]

        # covariance across batch (L x L)
        C = (mu_centered.t() @ mu_centered) / float(B)  # (L, L)
        # store for diagnostics
        self.last_cov = C.detach().cpu()

        # eigen-decomposition (symmetric)
        # Use torch.linalg.eigh for symmetric matrices
        try:
            s, V = torch.linalg.eigh(C)  # s: (L,), V: (L,L)
        except RuntimeError:
            # fallback to SVD if eigh fails
            U, S_svd, Vt = torch.linalg.svd(C)
            s = S_svd
            V = U

        # clamp eigenvalues for numerical stability
        s_clamped = torch.clamp(s, min=self.whiten_eps)

        # compute C^{-1/2} = V diag(1/sqrt(s)) V^T
        inv_sqrt = torch.diag(1.0 / torch.sqrt(s_clamped))
        C_inv_sqrt = (V @ inv_sqrt) @ V.t()  # (L, L)

        # whiten: mu_whitened = mu_centered @ C^{-1/2}
        mu_whitened = mu_centered @ C_inv_sqrt

        # add back the mean (optional) — here we keep zero-mean representation
        # but we also return a mean-centered version for diagnostics
        return mu_whitened

    # ---------------------------
    # Causal deconfounding (residualize child latents w.r.t parents)
    # ---------------------------
    def _causal_deconfound(self, z, causal_graph):
        """
        For each child c with parents P = causal_graph[c], compute linear regression:
            z_c = Z_p @ w + residual
        Replace z_c with residual = z_c - Z_p @ w
        Uses ridge regularization for stability.
        z: (B, L)
        """
        if causal_graph is None or len(causal_graph) == 0:
            return z

        z_res = z.clone()
        device = z.device
        B, L = z.shape

        for child, parents in causal_graph.items():
            if child < 0 or child >= L:
                continue
            if not parents:
                continue
            # ensure parents are valid indices
            parents = [p for p in parents if 0 <= p < L]
            if len(parents) == 0:
                continue

            Zp = z[:, parents]  # (B, P)
            zc = z[:, child].unsqueeze(1)  # (B, 1)

            # Solve (Zp^T Zp + ridge I) w = Zp^T zc
            # A = (P,P), b = (P,1)
            A = Zp.t() @ Zp
            diag = torch.eye(A.shape[0], device=device) * self.ridge
            A_reg = A + diag

            b = Zp.t() @ zc  # (P,1)

            # Solve for w
            try:
                w = torch.linalg.solve(A_reg, b)  # (P,1)
            except RuntimeError:
                # fallback to pinv if solve fails
                w = torch.linalg.pinv(A_reg) @ b

            # predicted part and residual
            pred = Zp @ w  # (B,1)
            residual = (zc - pred).squeeze(1)  # (B,)

            z_res[:, child] = residual

        return z_res

    # ---------------------------
    # Forward pass
    # ---------------------------
    def forward(self, x):
        """
        Returns:
            x_recon: reconstruction (B, C, H, W)
            mu: posterior mean (B, L)
            logvar: posterior log-variance (B, L)
            z_dec: latent used for decoding (B, L) after orthogonalization/causal steps
        """
        mu, logvar = self.encoder(x)  # (B, L), (B, L)
        z_sample = self.reparameterize(mu, logvar)  # (B, L)

        # diagnostics store
        self.last_mu = mu.detach()
        self.last_z = z_sample.detach()

        z_work = mu  # use posterior mean for structural transforms (stable)
        # Option: you may want to use sampled z instead; using mu is more stable for orthogonality

        # 1) orthogonalize (whiten) latent dims across the batch (optional)
        if self.orthogonalize_flag:
            z_ortho = self._whiten_latents(z_work)
            self.last_z_ortho = z_ortho.detach()
        else:
            z_ortho = z_work
            self.last_z_ortho = None

        # 2) causal deconfounding (optional)
        if self.causal_deconfound_flag and self.causal_graph:
            z_causal = self._causal_deconfound(z_ortho, self.causal_graph)
            self.last_z_causal = z_causal.detach()
        else:
            z_causal = z_ortho
            self.last_z_causal = None

        # Choose final latent for decoding
        z_dec = z_causal

        # decode (note: decoder expects batch of latent vectors)
        x_recon = self.decoder(z_dec)

        return x_recon, mu, logvar, z_dec

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)
