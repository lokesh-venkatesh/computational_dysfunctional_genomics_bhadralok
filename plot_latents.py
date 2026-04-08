import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def calculate_gc(seq):
    """Calculates GC percentage of a sequence."""
    seq = seq.upper()
    g_count = seq.count('G')
    c_count = seq.count('C')
    valid_len = len(seq.replace('N', ''))
    if valid_len == 0: return 0.0
    return (g_count + c_count) / valid_len

def calculate_kmer_oe(seq, kmer):
    """Calculates the Observed/Expected (O/E) ratio of a specific k-mer."""
    seq = seq.upper()
    kmer = kmer.upper()
    k = len(kmer)
    L = len(seq.replace('N', ''))
    
    if L < k: return 0.0
    
    # Observed count
    # Count overlapping occurrences as well (e.g., 'CGCG' in 'CGCGCG' is 2)
    obs = sum(1 for i in range(len(seq) - k + 1) if seq[i:i+k] == kmer)
    
    # Expected count assuming independent nucleotide probabilities
    prob = 1.0
    for nuc in kmer:
        nuc_count = seq.count(nuc)
        prob *= (nuc_count / L) if L > 0 else 0
        
    exp = (L - k + 1) * prob
    
    # Return O/E (add a tiny epsilon to avoid division by zero)
    if exp == 0: return 0.0
    return obs / exp

def plot_feature(df, x_col, y_col, feature, out_dir, pca_var):
    """Generates and saves a scatter plot for a given feature."""
    plt.figure(figsize=(10, 8))
    
    # Check if the feature is categorical or continuous
    is_categorical = df[feature].dtype == 'object' or df[feature].nunique() <= 10
    
    if is_categorical:
        # Categorical Plot (Discrete colors)
        sns.scatterplot(
            x=x_col, y=y_col,
            hue=feature,
            palette='tab10' if df[feature].nunique() <= 10 else 'tab20',
            data=df, alpha=0.6, s=15, edgecolor=None
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=feature)
    else:
        # Continuous Plot (Color gradient)
        scatter = plt.scatter(
            df[x_col], df[y_col],
            c=df[feature],
            cmap='viridis',
            alpha=0.6, s=15, edgecolors='none'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(feature, rotation=270, labelpad=15)

    plt.title(f'PCA of ProSNP-VAE Latent Space\nColored by {feature}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca_var[0]*100:.1f}% Var)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca_var[1]*100:.1f}% Var)', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    safe_name = feature.replace(":", "_")
    out_path = os.path.join(out_dir, f"pca_latent_{safe_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Saved plot for '{feature}' to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Map biological features onto VAE latents using PCA.")
    parser.add_argument("--latents", type=str, default="./data/val_dataset_latents.npy")
    # Provide the ORIGINAL CSV so we have the sequence column!
    parser.add_argument("--data_csv", type=str, default="./data/val_dataset.csv")
    parser.add_argument("--out_dir", type=str, default="./latent_plots")
    
    # Comma-separated list of features to compute and plot
    parser.add_argument("--features", type=str, 
                        default="chrom,ATAC,CTCF,gc,kmer:CG,kmer:GATA",
                        help="Comma-separated list of features to plot. Use 'kmer:SEQ' for k-mer O/E.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading latents from {args.latents}...")
    latents = np.load(args.latents)

    print(f"Loading original data from {args.data_csv}...")
    df = pd.read_csv(args.data_csv)
    
    # Standardize chromosome column just in case
    if 'chrom' not in df.columns and 'chr' in df.columns:
        df.rename(columns={'chr': 'chrom'}, inplace=True)

    if 'sequence' not in df.columns:
        print("ERROR: The provided data CSV does not have a 'sequence' column!")
        return

    # 1. Dimensionality Reduction via PCA
    print("Running PCA to reduce 256D space to 2D...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents)

    df['pca_1'] = latents_2d[:, 0]
    df['pca_2'] = latents_2d[:, 1]
    
    # 2. Process Requested Features
    features_to_plot = [f.strip() for f in args.features.split(",")]
    
    print("\nCalculating and Plotting features:")
    for feat in features_to_plot:
        # If it's a GC request
        if feat.lower() == 'gc':
            if 'gc' not in df.columns:
                df['gc'] = df['sequence'].apply(calculate_gc)
            plot_feature(df, 'pca_1', 'pca_2', 'gc', args.out_dir, pca.explained_variance_ratio_)
            
        # If it's a dynamic k-mer request
        elif feat.lower().startswith('kmer:'):
            kmer_seq = feat.split(":")[1].upper()
            df[feat] = df['sequence'].apply(lambda s: calculate_kmer_oe(s, kmer_seq))
            plot_feature(df, 'pca_1', 'pca_2', feat, args.out_dir, pca.explained_variance_ratio_)
            
        # If it's already a column in the dataset (chrom, ATAC, CTCF, etc.)
        elif feat in df.columns:
            plot_feature(df, 'pca_1', 'pca_2', feat, args.out_dir, pca.explained_variance_ratio_)
            
        else:
            print(f"  -> Skipping '{feat}': Not recognized and not found in dataset columns.")

    print(f"\nAll requested plots have been saved to the '{args.out_dir}' folder!")

if __name__ == "__main__":
    main()