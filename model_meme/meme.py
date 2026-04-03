import os
import subprocess
import pandas as pd
import numpy as np

class MemeClassifierModel:
    def __init__(self, data, max_train_seqs=100, output_dir="meme_temp"):
        """
        Initialize the MEME-based model.
        :param data: Training dataframe.
        :param max_train_seqs: Downsample bound sequences so MEME doesn't run forever.
        :param output_dir: Directory to store the intermediate FASTA and MEME/FIMO output files.
        """
        self.data = data
        self.max_train_seqs = max_train_seqs
        self.out_dir = output_dir
        self.tfs = ['CTCF', 'REST', 'EP300']
        
        # Ensure our working directory exists
        os.makedirs(self.out_dir, exist_ok=True)

    def _write_fasta(self, df, filepath):
        """Helper to write a dataframe's sequences to a FASTA file. Uses the index as the header."""
        with open(filepath, 'w') as f:
            for idx, row in df.iterrows():
                f.write(f">seq_{idx}\n{row['sequence']}\n")

    def fit(self):
        """Runs MEME on the bound sequences to discover a motif (the 'model')."""
        print(f"Training MEME Model (Max {self.max_train_seqs} sequences per TF)...")
        
        for tf in self.tfs:
            # 1. Get Bound sequences
            bound_df = self.data[self.data[tf] != 'U']
            
            # Subsample if necessary to save time
            if len(bound_df) > self.max_train_seqs:
                bound_df = bound_df.sample(n=self.max_train_seqs, random_state=42)
                
            # 2. Write to FASTA
            fasta_path = os.path.join(self.out_dir, f"train_{tf}.fa")
            self._write_fasta(bound_df, fasta_path)
            
            # 3. Run MEME
            meme_out_path = os.path.join(self.out_dir, f"meme_out_{tf}")
            cmd = [
                "meme", fasta_path,
                "-dna", "-oc", meme_out_path,
                "-nmotifs", "1",      # Just find the 1 best motif
                "-mod", "zoops",      # Zero or one occurrence per sequence
                "-maxw", "20"         # Max motif width of 20bp
            ]
            
            print(f"  Running MEME for {tf}...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        print("MEME training complete.")

    def infer(self, test_data):
        """Runs FIMO to score the test sequences using the discovered motifs."""
        print("Running Inference with FIMO...")
        
        # Write the whole test set to a FASTA file
        test_fasta_path = os.path.join(self.out_dir, "test_sequences.fa")
        self._write_fasta(test_data, test_fasta_path)
        
        predictions = {}
        
        for tf in self.tfs:
            print(f"  Scanning sequences for {tf} motif...")
            meme_txt_path = os.path.join(self.out_dir, f"meme_out_{tf}", "meme.txt")
            fimo_out_path = os.path.join(self.out_dir, f"fimo_out_{tf}")
            
            # Run FIMO
            cmd = [
                "fimo",
                "-oc", fimo_out_path,
                meme_txt_path, test_fasta_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Parse FIMO results
            fimo_tsv = os.path.join(fimo_out_path, "fimo.tsv")
            
            # Initialize all scores to a very low baseline (e.g., -50)
            # This handles sequences where FIMO found absolutely no match
            tf_scores = {f"seq_{idx}": -50.0 for idx in test_data.index}
            
            try:
                # FIMO output has columns like: motif_id, sequence_name, start, stop, strand, score, p-value, q-value, matched_sequence
                fimo_df = pd.read_csv(fimo_tsv, sep='\t', comment='#')
                if not fimo_df.empty:
                    # A sequence might have multiple matches; we take the max score for that sequence
                    max_scores = fimo_df.groupby('sequence_name')['score'].max().to_dict()
                    tf_scores.update(max_scores)
            except Exception as e:
                print(f"  Warning: Could not parse FIMO results for {tf} (maybe no motifs were found?). Error: {e}")
                
            # Map the scores back to the original test_data index order
            predictions[tf] = [tf_scores[f"seq_{idx}"] for idx in test_data.index]
            
        return pd.DataFrame(predictions, index=test_data.index)