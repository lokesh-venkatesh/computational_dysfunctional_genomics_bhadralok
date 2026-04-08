import pandas as pd
import os

def main():
    ambig_csv_path = "./data/ambig_bins.csv"
    target_chroms = ['chr3', 'chr10', 'chr17']

    print(f"Loading ambiguous bins from {ambig_csv_path}...")
    ambig_df = pd.read_csv(ambig_csv_path)

    ambig_df.rename(columns={'chrom': 'chr', 'stop': 'end'}, inplace=True)
    ambig_df = ambig_df[ambig_df['chr'].isin(target_chroms)].copy()

    ambig_df['CTCF'], ambig_df['REST'], ambig_df['EP300'] = "NA", "NA", "NA"
    ambig_df = ambig_df[['chr', 'start', 'end', 'ATAC', 'CTCF', 'REST', 'EP300']]

    print("\nMerging with predicted TSV files...")
    for chrom in target_chroms:
        tsv_path = f"./data/{chrom}.tsv"
        if not os.path.exists(tsv_path): continue
            
        pred_df = pd.read_csv(tsv_path, sep='\t')
        chrom_ambig_df = ambig_df[ambig_df['chr'] == chrom]
        
        merged_df = pd.concat([pred_df, chrom_ambig_df], ignore_index=True)
        merged_df.sort_values(by='start', inplace=True)
        merged_df.to_csv(tsv_path, sep='\t', index=False)
        
        print(f"  [{chrom}] -> Added {len(chrom_ambig_df)} 'NA' rows. Total bins: {len(merged_df)}")
        
    print("\nSuccess! Files are complete and ready for zipping/submission.")

if __name__ == "__main__":
    main()