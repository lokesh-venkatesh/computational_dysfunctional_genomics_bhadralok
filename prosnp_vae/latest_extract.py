import os
import torch
import argparse
import pandas as pd
import numpy as np
from prosnp_vae_model import ProSNPVAEWrapper, ProSNPVAE

def main():
    parser = argparse.ArgumentParser(description="Extract Latent Vectors (Mu) for downstream analysis.")
    parser.add_argument("--model_path", type=str, default="./prosnp_vae/checkpoints/best.pth")
    parser.add_argument("--data_csv", type=str, default="./data/val_dataset.csv")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_csv}...")
    df = pd.read_csv(args.data_csv)
    
    print(f"Initializing VAE on CPU...")
    wrapper = ProSNPVAEWrapper(epochs=1, batch_size=256, lr=1e-3)
    wrapper.model = ProSNPVAE().to(wrapper.device)
    
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=wrapper.device)
    if 'model_state_dict' in checkpoint:
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        wrapper.model.load_state_dict(checkpoint) 
        
    print("Extracting 256-Dimensional Latent Embeddings...")
    latent_matrix = wrapper.extract_latents(df)
    
    # Save the matrix to disk
    out_npy = args.data_csv.replace('.csv', '_latents.npy')
    np.save(out_npy, latent_matrix)
    print(f"\nSuccess! Extracted matrix of shape {latent_matrix.shape} saved to -> {out_npy}")
    
    # Save a lightweight metadata file to match rows
    out_meta = args.data_csv.replace('.csv', '_metadata.csv')
    meta_cols = ['chrom', 'start', 'stop', 'ATAC', 'CTCF', 'REST', 'EP300']
    available_cols = [c for c in meta_cols if c in df.columns]
    df[available_cols].to_csv(out_meta, index=False)
    print(f"Metadata saved to -> {out_meta}")

if __name__ == "__main__":
    main()