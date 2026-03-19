# ===============================================================
# ML-BASED LATENT SPACE HALO MOVIE PIPELINE
# ===============================================================
# AUTHOR: Abhinav Vatsa
# DATE: March 19, 2026
# ===============================================================
# PURPOSE:
# - Learn a latent representation of the MW–M31 halo N-body snapshots
# - Generate time-resolved movies of halo density, velocity anisotropy,
#   and rotation profiles directly from latent codes
# - Reduce repeated computation by decoding latent embeddings
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from CenterOfMass2 import CenterOfMass
import os

# ===============================================================
# 1. PARAMETERS
# ===============================================================

snapshots = np.arange(0, 800)
latent_dim = 5        # latent space dimensions
batch_size = 1024
epochs = 20
fps = 20
bitrate = 2000
device = "cuda" if torch.cuda.is_available() else "cpu"
bins = 50             # radial/density bin resolution

# ===============================================================
# 2. AUTOENCODER DEFINITION
# ===============================================================

class HaloAutoencoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=5):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ===============================================================
# 3. DATA LOADING AND PREPROCESSING
# ===============================================================

def load_snapshots(snapshots, data_dir="."):
    all_particles = []
    for snap in snapshots:
        mw_file = os.path.join(data_dir, f"MW_{snap:03d}.txt")
        m31_file = os.path.join(data_dir, f"M31_{snap:03d}.txt")
        MW = CenterOfMass(mw_file,1)
        M31 = CenterOfMass(m31_file,1)
        
        x = np.concatenate((MW.x, M31.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))
        vx = np.concatenate((MW.vx, M31.vx))
        vy = np.concatenate((MW.vy, M31.vy))
        vz = np.concatenate((MW.vz, M31.vz))
        m = np.concatenate((MW.m, M31.m))
        
        # Center system
        xcom, ycom, zcom = MW.COMdefine(x,y,z,m)
        vxcom, vycom, vzcom = MW.COMdefine(vx,vy,vz,m)
        x -= xcom; y -= ycom; z -= zcom
        vx -= vxcom; vy -= vycom; vz -= vzcom
        
        particles = np.vstack((x,y,z,vx,vy,vz)).T
        all_particles.append(particles)
        
    return np.vstack(all_particles)

# Load all snapshots into a single dataset for training
data_dir = "./data"  # adjust path
X_raw = load_snapshots(snapshots, data_dir)
print(f"Loaded {X_raw.shape[0]} particles across {len(snapshots)} snapshots.")

# Standardize features for ML
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Convert to PyTorch tensor
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===============================================================
# 4. MODEL INITIALIZATION
# ===============================================================

model = HaloAutoencoder(input_dim=6, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ===============================================================
# 5. TRAINING LOOP
# ===============================================================

for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        x_hat, _ = model(x_batch)
        loss = criterion(x_hat, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

# ===============================================================
# 6. LATENT SPACE PROJECTION
# ===============================================================

with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    _, Z = model(X_tensor)
    Z = Z.cpu().numpy()  # latent representation

# Optional: reduce to 3D for visualization
pca = PCA(n_components=3)
Z_vis = pca.fit_transform(Z)

# ===============================================================
# 7. CLUSTERING IN LATENT SPACE
# ===============================================================

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(Z)

# ===============================================================
# 8. MOVIE GENERATION FUNCTION
# ===============================================================

def generate_density_movie(Z, cluster_labels, scaler, filename="ml_density_movie.mp4"):
    fig, ax = plt.subplots(figsize=(7,7))
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    
    with writer.saving(fig, filename, dpi=200):
        n_particles = Z.shape[0]
        steps_per_snapshot = n_particles // len(snapshots)
        
        for i, snap in enumerate(snapshots):
            # Select particles corresponding to this snapshot
            start = i * steps_per_snapshot
            end = (i+1) * steps_per_snapshot
            Z_snap = Z[start:end]
            
            # Decode latent to original 6D space
            Z_tensor = torch.tensor(Z_snap, dtype=torch.float32).to(device)
            X_hat, _ = model.decoder(Z_tensor).cpu().detach().numpy()
            X_hat = scaler.inverse_transform(X_hat)
            x = X_hat[:,0]
            y = X_hat[:,1]
            
            # 2D histogram
            H, xedges, yedges = np.histogram2d(x, y, bins=bins)
            H = np.log10(H + 1)
            
            ax.cla()
            ax.imshow(H.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect='equal')
            ax.set_title(f"ML Density Evolution Snapshot {snap}")
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("y [kpc]")
            
            writer.grab_frame()
    plt.close()
    print(f"Density movie saved to {filename}")

# ===============================================================
# 9. GENERATE DENSITY MOVIE
# ===============================================================

generate_density_movie(Z, cluster_labels, scaler)

# ===============================================================
# FUTURE EXTENSIONS
# ===============================================================
# - Velocity anisotropy and rotation profiles can be similarly decoded
#   from latent space.
# - Interpolation between latent snapshots can generate smooth movies.
# - Latent clustering identifies tidal streams and substructures.
# - Could replace autoencoder with VAE for continuous generative modeling.
# ===============================================================
