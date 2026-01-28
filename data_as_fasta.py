import pandas as pd
import os

for chrom in range(1,23):
    if chrom in [3, 10, 17]: tsv_filepath = f"projectData/chr{chrom}_200bp_bins_unknown.tsv"
    else: tsv_filepath = f"projectData/chr{chrom}_200bp_bins.tsv"
    
    fasta_filepath = f"hg38/chr{chrom}.fa"
    sequence_data = pd.read_csv(tsv_filepath, sep='\t')
    
    with open(fasta_filepath, 'r') as f:
        fasta_lines = f.readlines()
    chromosome_sequence = "".join(fasta_lines[1:]).replace("\n", "")

    chromosome_seqs_dict = {}
    
    for row in sequence_data.itertuples(index=False):
        chr = f"chr{chrom}"
        start = row[1]
        end = row[2]

        if chrom in [3, 10, 17]:
            ATAC = row[3]
            seq_label = f">{chr}_{start}_{end}_{ATAC}"
        else: 
            ATAC = row[3]
            CTCF = row[4]
            REST = row[5]
            EP300 = row[6]
            seq_label = f">{chr}_{start}_{end}_{ATAC}_{CTCF}_{REST}_{EP300}"
        seq = chromosome_sequence[start-1:end-1]

        chromosome_seqs_dict[seq_label] = seq
    
    os.makedirs("data", exist_ok=True)

    with open(f"data/chr{chrom}_seqs.fa", "w") as fasta_file:
        for label, sequence in chromosome_seqs_dict.items():
            fasta_file.write(f"{label}\n{sequence}\n")

    print(f"Sequences and Sequence Data saved for Chromosome {chrom}")