import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

# ==========================================
# 1. Setup Paths
# ==========================================
results_dir = "model_vae/results_latest"
latent_path = os.path.join(results_dir, "latent_vectors_val.npy")
val_csv_path = "data/val_dataset.csv"

def plot_latent_space():
    if not os.path.exists(latent_path):
        print(f"Error: Could not find {latent_path}.")
        print("Make sure you let run.py finish completely so it saves the .npy file!")
        return

    print("Loading Latent Vectors and Validation Data...")
    Z = np.load(latent_path)          # Shape: (N_samples, 100)
    df = pd.read_csv(val_csv_path)    # Shape: (N_samples, ...)
    
    tfs = ['CTCF', 'REST', 'EP300']
    
    # ==========================================
    # 2. Dimensionality Reduction (UMAP)
    # ==========================================
    print("Running PCA to speed up UMAP...")
    Z_pca = PCA(n_components=50).fit_transform(Z)
    
    print("Running UMAP (compressing 50D -> 2D)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    Z_2d = reducer.fit_transform(Z_pca)
    
    # ==========================================
    # 3. Plotting
    # ==========================================
    print("Generating plots...")
    
    # Use a dark background style for better contrast (optional but looks great)
    plt.style.use('seaborn-v0_8-darkgrid') 
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Dual-Stream VAE Latent Space (UMAP)', fontsize=18, fontweight='bold', y=1.05)
    
    # Define our new, nicer color palette
    color_unbound = '#3498db'  # A nice, soft Peter River Blue
    color_bound = '#e74c3c'    # A vibrant Alizarin Crimson 
    
    for i, tf in enumerate(tfs):
        ax = axes[i]
        
        # Convert 'U'/'B' string labels to boolean mask
        is_bound = (df[tf] != 'U').values
        
        # 1. Plot Unbound (Background) 
        # Z-order=1 puts it at the back. High transparency so dense clusters don't look solid.
        ax.scatter(Z_2d[~is_bound, 0], Z_2d[~is_bound, 1], 
                   c=color_unbound, alpha=0.15, s=10, 
                   label='Unbound (U)', zorder=1)
        
        # 2. Plot Bound (Foreground)
        # Z-order=2 forces these to be drawn ON TOP. 
        # Added white edgecolors to make the individual red dots pop out of the blue sea.
        ax.scatter(Z_2d[is_bound, 0], Z_2d[is_bound, 1], 
                   c=color_bound, alpha=0.9, s=25, 
                   edgecolors='white', linewidths=0.5,
                   label='Bound (B)', zorder=2)
        
        ax.set_title(f'{tf} Binding', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Clean up the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Customizing the legend to have fully opaque markers so you can actually see them
        leg = ax.legend(loc='upper right', frameon=True, shadow=True)
        for lh in leg.legend_handles: 
            lh.set_alpha(1)
            lh.set_sizes([50])
        
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(results_dir, "latent_space_umap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDone! High-contrast plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_latent_space()