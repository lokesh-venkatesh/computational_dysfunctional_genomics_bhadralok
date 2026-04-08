import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge VAE latents with metadata and plot using PCA.")
    # Assuming you ran extract_latents.py on val_dataset.csv
    parser.add_argument("--latents", type=str, default="./data/val_dataset_latents.npy")
    parser.add_argument("--metadata", type=str, default="./data/val_dataset_metadata.csv")
    parser.add_argument("--output_csv", type=str, default="./data/val_dataset_combined_latents.csv")
    args = parser.parse_args()

    print(f"Loading latents from {args.latents}...")
    latents = np.load(args.latents)

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)

    # 1. Combine Latents with Original Metadata
    # Name the latent columns z_0, z_1, ..., z_255
    latent_cols = [f"z_{i}" for i in range(latents.shape[1])]
    latent_df = pd.DataFrame(latents, columns=latent_cols)
    
    combined_df = pd.concat([df, latent_df], axis=1)

    print(f"Saving combined dataset to {args.output_csv}...")
    combined_df.to_csv(args.output_csv, index=False)

    # 2. Dimensionality Reduction via PCA (256D -> 2D)
    print("Running PCA to reduce 256D space to 2D...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents)

    combined_df['pca_1'] = latents_2d[:, 0]
    combined_df['pca_2'] = latents_2d[:, 1]

    # Dynamically find the chromosome column name for coloring
    color_col = 'chrom'
    if 'chrom' not in combined_df.columns:
        if 'chr' in combined_df.columns: color_col = 'chr'
        elif 'chromosome' in combined_df.columns: color_col = 'chromosome'

    # 3. Plot the latent space
    print(f"Plotting latent space colored by {color_col}...")
    plt.figure(figsize=(12, 8))
    
    # We use seaborn for easy mapping of categories to colors
    sns.scatterplot(
        x='pca_1', y='pca_2',
        hue=color_col,
        palette='tab20', # Great palette for distinguishing multiple categories
        data=combined_df,
        alpha=0.6,
        s=20,
        edgecolor=None
    )
    
    plt.title('PCA of ProSNP-VAE Latent Space', fontsize=16, fontweight='bold')
    
    # Show the explained variance on the axes!
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
    
    # Move the legend outside the plot so it doesn't cover your data points
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Chromosome")
    plt.tight_layout()
    
    # Display the plot, nothing else!
    plt.show()

if __name__ == "__main__":
    main()