# ===============================================================
# ML-BASED LATENT SPACE HALO MOVIE PIPELINE 
# ===============================================================
# AUTHOR : Abhinav Vatsa
# ===============================================================

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from CenterOfMass2 import CenterOfMass

# ────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ────────────────────────────────────────────────────────────────

SNAPSHOTS   = np.arange(0, 800)
DATA_DIR    = "./data"
OUT_DIR     = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

LATENT_DIM  = 8
BATCH_SIZE  = 2048
EPOCHS      = 40
LR          = 3e-4
BETA_KL     = 1e-3
BINS        = 80
FPS         = 24
BITRATE     = 3000
N_CLUSTERS  = 7
R_MAX_KPC   = 300.0
N_RBINS     = 40
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {DEVICE}")

# ════════════════════════════════════════════════════════════════
# PHASE 1 │ DATA LOADING
# ════════════════════════════════════════════════════════════════

def _load_one(snap, data_dir):
    mw_file  = os.path.join(data_dir, f"MW_{snap:03d}.txt")
    m31_file = os.path.join(data_dir, f"M31_{snap:03d}.txt")

    MW  = CenterOfMass(mw_file,  1)
    M31 = CenterOfMass(m31_file, 1)

    x  = np.concatenate((MW.x,  M31.x))
    y  = np.concatenate((MW.y,  M31.y))
    z  = np.concatenate((MW.z,  M31.z))
    vx = np.concatenate((MW.vx, M31.vx))
    vy = np.concatenate((MW.vy, M31.vy))
    vz = np.concatenate((MW.vz, M31.vz))
    m  = np.concatenate((MW.m,  M31.m))

    xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
    vxcom, vycom, vzcom = MW.COMdefine(vx, vy, vz, m)

    x -= xcom; y -= ycom; z -= zcom
    vx -= vxcom; vy -= vycom; vz -= vzcom

    return np.vstack((x, y, z, vx, vy, vz, m)).T

def load_all_snapshots():
    all_parts = []
    snap_sizes = []

    for snap in SNAPSHOTS:
        p = _load_one(snap, DATA_DIR)
        all_parts.append(p)
        snap_sizes.append(len(p))

    return np.vstack(all_parts), snap_sizes

# Run Phase 1
X_raw7, snap_sizes = load_all_snapshots()
X_raw = X_raw7[:, :6]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))

# ════════════════════════════════════════════════════════════════
# PHASE 2 │ VAE
# ════════════════════════════════════════════════════════════════

class HaloVAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + BETA_KL * kl

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = HaloVAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
for epoch in range(EPOCHS):
    total_loss = 0
    for (xb,) in loader:
        xb = xb.to(DEVICE)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(xb)
        loss = vae_loss(x_hat, xb, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss/len(dataset):.6f}")

torch.save(model.state_dict(), os.path.join(OUT_DIR, "vae.pt"))

# ════════════════════════════════════════════════════════════════
# PHASE 3 │ LATENT SPACE
# ════════════════════════════════════════════════════════════════

model.eval()
Z = []

with torch.no_grad():
    for (xb,) in DataLoader(dataset, batch_size=8192):
        mu, _ = model.encode(xb.to(DEVICE))
        Z.append(mu.cpu().numpy())

Z = np.vstack(Z)

pca = PCA(n_components=3)
Z_pca = pca.fit_transform(Z)

print("PCA variance:", pca.explained_variance_ratio_)

# ════════════════════════════════════════════════════════════════
# SIMPLE DENSITY MOVIE (example)
# ════════════════════════════════════════════════════════════════

def generate_density_movie():
    fig, ax = plt.subplots()

    writer = FFMpegWriter(fps=FPS)

    with writer.saving(fig, os.path.join(OUT_DIR, "density.mp4"), dpi=150):
        for i in range(100):  # shortened for demo
            sample = Z[np.random.choice(len(Z), 50000)]

            ax.clear()
            ax.hist2d(sample[:,0], sample[:,1], bins=80)
            ax.set_title(f"Frame {i}")

            writer.grab_frame()

    plt.close()

generate_density_movie()

print("[DONE]")
