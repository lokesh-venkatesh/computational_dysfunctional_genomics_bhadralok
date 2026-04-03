"""
Unified data processing pipeline for TF binding prediction project.
Combines genome downloading, FASTA conversion, and dataset creation.
Includes intelligent caching to skip redundant operations.

Stages:
1. Download hg38 reference genome (chromosome-wise)
2. Convert TSV bin data to FASTA format
3. Filter ambiguous sequences and create unified dataset
"""

import os
import requests
import gzip
import shutil
import pandas as pd


def stage1_download_genome():
    """Downloads the hg38 reference genome from UCSC.
    Skips chromosomes that already exist."""
    
    folder_name = "data/hg38"
    os.makedirs(folder_name, exist_ok=True)
    
    base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes"
    downloaded_count = 0
    
    for chrom in range(1, 23):
        filename = f"chr{chrom}.fa"
        filepath = os.path.join(folder_name, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists, skipping download")
            continue
        
        # Download and extract
        filename_gz = f"{filename}.gz"
        url = f"{base_url}/{filename_gz}"
        filepath_gz = os.path.join(folder_name, filename_gz)
        
        print(f"↓ Downloading {filename_gz}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath_gz, 'wb') as f:
                f.write(response.content)
            
            # Extract
            print(f"  Extracting {filename}...")
            with gzip.open(filepath_gz, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath_gz)
            downloaded_count += 1
            
        except Exception as e:
            print(f"✗ Error downloading {filename_gz}: {e}")
    
    if downloaded_count == 0:
        print("Stage 1: All chromosomes already downloaded, skipping.\n")
    else:
        print(f"Stage 1: Downloaded and extracted {downloaded_count} chromosome(s).\n")


def stage2_convert_to_fasta():
    """Converts TSV bin data to FASTA format.
    Skips chromosomes that already have corresponding FASTA files."""
    
    os.makedirs("data/fasta", exist_ok=True)
    converted_count = 0
    
    for chrom in range(1, 23):
        fasta_filepath = f"data/fasta/chr{chrom}_seqs.fa"
        
        # Skip if file already exists
        if os.path.exists(fasta_filepath):
            print(f"✓ chr{chrom}_seqs.fa already exists, skipping conversion")
            continue
        
        # Determine TSV file path
        if chrom in [3, 10, 17]:
            tsv_filepath = f"data/projectData/chr{chrom}_200bp_bins_unknown.tsv"
        else:
            tsv_filepath = f"data/projectData/chr{chrom}_200bp_bins.tsv"
        
        # Check if TSV file exists
        if not os.path.exists(tsv_filepath):
            print(f"⚠ TSV file not found: {tsv_filepath}, skipping")
            continue
        
        # Read TSV data
        try:
            sequence_data = pd.read_csv(tsv_filepath, sep='\t')
        except Exception as e:
            print(f"✗ Error reading {tsv_filepath}: {e}")
            continue
        
        # Read reference genome
        hg38_filepath = f"data/hg38/chr{chrom}.fa"
        if not os.path.exists(hg38_filepath):
            print(f"⚠ Genome file not found: {hg38_filepath}, skipping")
            continue
        
        with open(hg38_filepath, 'r') as f:
            fasta_lines = f.readlines()
        chromosome_sequence = "".join(fasta_lines[1:]).replace("\n", "")
        
        # Extract sequences
        chromosome_seqs_dict = {}
        for row in sequence_data.itertuples(index=False):
            start, end, ATAC = row[1], row[2], row[3]
            
            if chrom in [3, 10, 17]:
                seq_label = f">chr{chrom}_{start}_{end}_{ATAC}"
            else:
                CTCF, REST, EP300 = row[4], row[5], row[6]
                seq_label = f">chr{chrom}_{start}_{end}_{ATAC}_{CTCF}_{REST}_{EP300}"
            
            seq = chromosome_sequence[start:end]
            chromosome_seqs_dict[seq_label] = seq
        
        # Write FASTA file
        with open(fasta_filepath, "w") as fasta_file:
            for label, sequence in chromosome_seqs_dict.items():
                fasta_file.write(f"{label}\n{sequence}\n")
        
        print(f"✓ Sequences saved for Chromosome {chrom}")
        converted_count += 1
    
    if converted_count == 0:
        print("Stage 2: All FASTA files already exist, skipping.\n")
    else:
        print(f"Stage 2: Converted {converted_count} chromosome(s) to FASTA.\n")


def stage3_create_unified_dataset():
    """Filters ambiguous sequences and creates unified dataset.
    Skips if output files already exist."""
    
    dataset_output = "data/dataset.csv"
    ambig_output = "resources/ambig_bins.csv"
    
    # Check if both output files exist
    if os.path.exists(dataset_output) and os.path.exists(ambig_output):
        print(f"✓ {dataset_output} already exists")
        print(f"✓ {ambig_output} already exists")
        print("Stage 3: Dataset files already exist, skipping.\n")
        return
    
    # Create output directory
    os.makedirs("resources", exist_ok=True)
    
    all_seqs = {}
    ambig_seqs = {}
    
    for chrom_no in range(1, 23):
        fasta_filepath = f"data/fasta/chr{chrom_no}_seqs.fa"
        
        # Skip if FASTA file doesn't exist yet
        if not os.path.exists(fasta_filepath):
            print(f"⚠ FASTA file not found: {fasta_filepath}, skipping")
            continue
        
        with open(fasta_filepath, 'r') as f:
            fasta_lines = f.readlines()
        
        info = [odd_line.replace("\n", "") for odd_line in fasta_lines[0::2]]
        seqs = [even_line.replace("\n", "") for even_line in fasta_lines[1::2]]
        
        for info_label, seq in zip(info, seqs):
            deets = info_label[1:]  # Remove '>' character
            info = deets.split("_")
            
            chrom, start, stop, ATAC = info[0], int(info[1]), int(info[2]), info[3]
            
            # Identify and store ambiguous sequences
            if not all(nucl in 'ATGC' for nucl in seq):
                if len(info) == 4:
                    ambig_seqs[deets] = {
                        "chrom": chrom, "start": start, "stop": stop,
                        "ATAC": ATAC, "CTCF": None, "REST": None, "EP300": None,
                        "sequence": seq
                    }
                elif len(info) > 4:
                    CTCF, REST, EP300 = info[4], info[5], info[6]
                    ambig_seqs[deets] = {
                        "chrom": chrom, "start": start, "stop": stop,
                        "ATAC": ATAC, "CTCF": CTCF, "REST": REST, "EP300": EP300,
                        "sequence": seq
                    }
            
            # Clean and store valid sequences
            seq_clean = ''.join(nucl for nucl in seq if nucl in 'ATCG')
            if len(seq_clean) == 200:  # Ensure clean sequence is still 200bp
                if len(info) == 4:
                    all_seqs[deets] = {
                        "chrom": chrom, "start": start, "stop": stop,
                        "ATAC": ATAC, "CTCF": None, "REST": None, "EP300": None,
                        "sequence": seq_clean
                    }
                elif len(info) > 4:
                    CTCF, REST, EP300 = info[4], info[5], info[6]
                    all_seqs[deets] = {
                        "chrom": chrom, "start": start, "stop": stop,
                        "ATAC": ATAC, "CTCF": CTCF, "REST": REST, "EP300": EP300,
                        "sequence": seq_clean
                    }
        
        print(f"✓ Chromosome {chrom_no} parsed")
    
    # Save datasets
    print(f"\n→ Writing {dataset_output}...")
    all_seqs_df = pd.DataFrame.from_dict(all_seqs, orient='index')
    all_seqs_df.to_csv(dataset_output, index=False)
    
    print(f"→ Writing {ambig_output}...")
    ambig_seqs_df = pd.DataFrame.from_dict(ambig_seqs, orient='index')
    ambig_seqs_df.to_csv(ambig_output, index=False)
    
    print(f"Stage 3: Created unified dataset with {len(all_seqs)} valid sequences and {len(ambig_seqs)} ambiguous sequences.\n")


def main():
    """Run the complete data processing pipeline."""
    print("=" * 70)
    print("TF Binding Prediction - Data Processing Pipeline")
    print("=" * 70 + "\n")
    
    print("Stage 1: Downloading hg38 reference genome...")
    print("-" * 70)
    stage1_download_genome()
    
    print("Stage 2: Converting TSV bins to FASTA format...")
    print("-" * 70)
    stage2_convert_to_fasta()
    
    print("Stage 3: Creating unified dataset...")
    print("-" * 70)
    stage3_create_unified_dataset()
    
    print("=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()