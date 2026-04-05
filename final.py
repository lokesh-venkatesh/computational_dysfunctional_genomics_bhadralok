import pandas as pd
import os

def main():
    ambig_csv_path = "./data/ambig_bins.csv"
    target_chroms = ['chr3', 'chr10', 'chr17']

    print(f"Loading ambiguous bins from {ambig_csv_path}...")
    if not os.path.exists(ambig_csv_path):
        print(f"Error: Could not find {ambig_csv_path}")
        return
    
    # Read the ambiguous dataset
    ambig_df = pd.read_csv(ambig_csv_path)

    # Standardize column names to match your output format
    col_map = {'chrom': 'chr', 'stop': 'end'}
    ambig_df.rename(columns=col_map, inplace=True)

    # Filter out chromosomes we don't care about
    ambig_df = ambig_df[ambig_df['chr'].isin(target_chroms)].copy()

    # Fill the missing/unpredicted labels with 'NA'
    ambig_df['CTCF'] = "NA"
    ambig_df['REST'] = "NA"
    ambig_df['EP300'] = "NA"

    # Drop the sequence column and strictly reorder to match the required submission headers
    out_cols = ['chr', 'start', 'end', 'ATAC', 'CTCF', 'REST', 'EP300']
    ambig_df = ambig_df[out_cols]

    print("\nMerging with predicted TSV files...")
    for chrom in target_chroms:
        tsv_path = f"./data/{chrom}.tsv"
        
        if not os.path.exists(tsv_path):
            print(f"Warning: {tsv_path} not found. Did you run predict.py yet? Skipping...")
            continue
            
        # 1. Load the model's predictions
        pred_df = pd.read_csv(tsv_path, sep='\t')
        
        # 2. Grab the ambiguous bins for this specific chromosome
        chrom_ambig_df = ambig_df[ambig_df['chr'] == chrom]
        
        # 3. Concatenate them together
        merged_df = pd.concat([pred_df, chrom_ambig_df], ignore_index=True)
        
        # 4. Sort sequentially by start position (critical for genomic formats)
        merged_df.sort_values(by='start', inplace=True)
        
        # 5. Save back to the same .tsv file
        merged_df.to_csv(tsv_path, sep='\t', index=False)
        
        print(f"  [{chrom}] -> Added {len(chrom_ambig_df)} 'NA' rows. Total bins: {len(merged_df)}")
        
    print("\nSuccess! Files are complete and ready for zipping/submission.")

if __name__ == "__main__":
    main()