import os
import torch
import argparse
import pandas as pd
from prosnp_vae_model import ProSNPVAEWrapper, ProSNPVAE

def main():
    parser = argparse.ArgumentParser(description="Run ProSNP-VAE inference on test data.")
    parser.add_argument("--model_path", type=str, default="./prosnp_vae/checkpoints/best.pth")
    parser.add_argument("--test_csv", type=str, default="./data/test_dataset.csv")
    args = parser.parse_args()

    print(f"Loading test dataset from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    
    col_map = {}
    for col in test_df.columns:
        if col.lower() in ['chrom', 'chromosome']: col_map[col] = 'chr'
        elif col.lower() == 'stop': col_map[col] = 'end'
    test_df.rename(columns=col_map, inplace=True)
    
    print(f"Initializing VAE on CPU...")
    wrapper = ProSNPVAEWrapper(epochs=1, batch_size=256, lr=1e-3)
    wrapper.model = ProSNPVAE().to(wrapper.device)
    
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=wrapper.device)
    if 'model_state_dict' in checkpoint:
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        wrapper.model.load_state_dict(checkpoint) 
        
    print("Running inference...")
    preds_df = wrapper.infer(test_df)
    test_df['CTCF'], test_df['REST'], test_df['EP300'] = preds_df['CTCF'], preds_df['REST'], preds_df['EP300']
    
    out_cols = ['chr', 'start', 'end', 'ATAC', 'CTCF', 'REST', 'EP300']
    print("\nSaving chromosome files...")
    for chrom in ['chr3', 'chr10', 'chr17']:
        chrom_df = test_df[test_df['chr'] == chrom]
        if not chrom_df.empty:
            out_path = f"./data/{chrom}.tsv"
            chrom_df[out_cols].to_csv(out_path, sep='\t', index=False)
            print(f"Saved {len(chrom_df)} sequences to -> {out_path}")

if __name__ == "__main__":
    main()