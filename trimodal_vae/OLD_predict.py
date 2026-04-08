import os
import torch
import argparse
import pandas as pd
from trimodal_vae.OLD_trimodal_model import TriModalModel, TriModalVAE

def main():
    parser = argparse.ArgumentParser(description="Run Tri-Modal VAE inference on test data.")
    
    # Gives you command-line control over the checkpoint path, with the requested default
    parser.add_argument("--model_path", type=str, 
                        default="./trimodal_vae/results_latest/checkpoints/best_colab.pth",
                        # set the default to best.pth once you are done
                        help="Path to the trained .pth model weights")
    parser.add_argument("--test_csv", type=str, default="./data/test_dataset.csv")
    args = parser.parse_args()

    print(f"Loading test dataset from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    
    # ===================================================================
    # FIX: Standardize column names dynamically to match submission format
    # ===================================================================
    col_map = {}
    for col in test_df.columns:
        if col.lower() in ['chrom', 'chromosome']:
            col_map[col] = 'chr'
        elif col.lower() == 'stop':
            col_map[col] = 'end'
    
    test_df.rename(columns=col_map, inplace=True)
    
    # Safety Check
    if 'chr' not in test_df.columns or 'end' not in test_df.columns:
        print(f"CRITICAL ERROR: Could not find required coordinate columns. Available columns: {list(test_df.columns)}")
        return
    # ===================================================================
    
    # Initialize model wrapper (using CPU-friendly batch size)
    print("Initializing Tri-Modal VAE on CPU...")
    wrapper = TriModalModel(epochs=1, batch_size=256, lr=1e-4, target_beta=0.2, lambda_cls=500.0, use_context=True)
    
    # Manually instantiate the architecture
    wrapper.model = TriModalVAE().to(wrapper.device)
    
    # Safely load weights from Colab (GPU) to your local machine (CPU)
    print(f"Loading weights from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Make sure you downloaded it from your Google Drive!")
        
    checkpoint = torch.load(args.model_path, map_location=wrapper.device)
    
    # Handle the new "Full Checkpoint" dictionary format vs the old format
    if 'model_state_dict' in checkpoint:
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded full checkpoint (Epoch {checkpoint['epoch']}).")
    else:
        wrapper.model.load_state_dict(checkpoint) 
        print("Loaded legacy weights-only checkpoint.")
        
    # Run inference (Generates a dataframe with columns: CTCF, REST, EP300)
    print("Running inference...")
    preds_df = wrapper.infer(test_df, "test")
    
    # Merge the probabilities back with the original metadata
    test_df['CTCF'] = preds_df['CTCF']
    test_df['REST'] = preds_df['REST']
    test_df['EP300'] = preds_df['EP300']
    
    # Extract, format, and save each chromosome
    target_chroms = ['chr3', 'chr10', 'chr17']
    # FIX: Changed 'stop' to 'end' to match the submission prompt perfectly
    out_cols = ['chr', 'start', 'end', 'ATAC', 'CTCF', 'REST', 'EP300']
    
    print("\nSaving chromosome files...")
    for chrom in target_chroms:
        chrom_df = test_df[test_df['chr'] == chrom]
        
        if not chrom_df.empty:
            out_path = f"./data/{chrom}.tsv"
            # Using sep='\t' to create the tab-separated TSV format you requested
            chrom_df[out_cols].to_csv(out_path, sep='\t', index=False)
            print(f"Saved {len(chrom_df)} sequences to -> {out_path}")
        else:
            print(f"Warning: No sequences found for {chrom} in the test dataset.")
            
    print("\nInference complete!")

if __name__ == "__main__":
    main()