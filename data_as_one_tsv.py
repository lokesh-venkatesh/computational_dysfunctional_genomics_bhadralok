# script for converting the .fa files of the 23 chromosomes into directly usable datasets

import pandas as pd
all_seqs = {}

for chrom in range(1,23):
    fasta_filepath = f"data/chr{chrom}_seqs.fa"
    with open(fasta_filepath, 'r') as f:
        fasta_lines = f.readlines()
    
    info = [odd_line.replace("\n", "") for odd_line in fasta_lines[0::2]]
    seqs = [even_line.replace("\n", "") for even_line in fasta_lines[1::2]]

    for info, seq in zip(info, seqs):
        deets = info[1:]
        info = deets.split("_")
        
        chrom = info[0]
        start = int(info[1])
        stop = int(info[2])
        ATAC = info[3]

        seq = seq.upper()
        seq = ''.join(nucl for nucl in seq if nucl in 'ATCG')
        
        if type(seq) is str and len(seq)==200:
            if len(info)==4:
                all_seqs[deets] = {"chrom":chrom, "start":start, "stop":stop,
                                "ATAC":ATAC, "CTCF":None, "REST":None, "EP300":None, "sequence":seq}
            elif len(info)>4:
                CTCF = info[4]
                REST = info[5]
                EP300 = info[6]

                all_seqs[deets] = {"chrom":chrom, "start":start, "stop":stop,
                                "ATAC":ATAC, "CTCF":CTCF, "REST":REST, "EP300":EP300, "sequence":seq}
    print(f"Chromosome {chrom} parsed")
        
all_seqs_df = pd.DataFrame.from_dict(all_seqs, orient='index')
#last_col_values = all_seqs_df.iloc[:, -1]
#non_string_mask = last_col_values.apply(lambda x: not isinstance(x, str))
#removed_indices = all_seqs_df[non_string_mask].index.tolist()
#print(f"Removed row indices: {removed_indices}")
#df = all_seqs_df[~non_string_mask]
all_seqs_df.to_csv("dataset.csv")

print("All seqs procssed into one .tsv file!")