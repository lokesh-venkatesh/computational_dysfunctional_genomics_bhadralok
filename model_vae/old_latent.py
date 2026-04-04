
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    # 2. Dimensionality Reduction (t-SNE)
    # ==========================================
    print("Running PCA to speed up t-SNE...")
    # PCA first to reduce noise and speed up t-SNE calculation
    Z_pca = PCA(n_components=50).fit_transform(Z)
    
    print("Running t-SNE (compressing 50D -> 2D)... This might take a minute...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, n_jobs=1)
    Z_2d = tsne.fit_transform(Z_pca)
    
    # ==========================================
    # 3. Plotting
    # ==========================================
    print("Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Dual-Stream VAE Latent Space (t-SNE)', fontsize=16)
    
    for i, tf in enumerate(tfs):
        ax = axes[i]
        
        # Convert 'U'/'B' string labels to boolean mask
        # Assuming your CSV uses 'U' for Unbound, and anything else (like 'B') for Bound
        is_bound = (df[tf] != 'U').values
        
        # Plot Unbound (Background) - Grey, smaller, highly transparent
        ax.scatter(Z_2d[~is_bound, 0], Z_2d[~is_bound, 1], 
                   c='lightgray', alpha=0.3, s=10, label='Unbound (U)')
        
        # Plot Bound (Foreground) - Colored, slightly larger, opaque
        # Choosing distinct colors for the 3 TFs
        colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
        ax.scatter(Z_2d[is_bound, 0], Z_2d[is_bound, 1], 
                   c=colors[i], alpha=0.8, s=15, label='Bound (B)')
        
        ax.set_title(f'{tf} Binding')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        
        # Remove axes ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='best')
        
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(results_dir, "latent_space_tsne.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDone! Plot saved to: {save_path}")
    
    # Show the plot window if running interactively
    plt.show()

if __name__ == "__main__":
    plot_latent_space()