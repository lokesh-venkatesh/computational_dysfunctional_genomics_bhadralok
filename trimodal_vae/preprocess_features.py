import os
import glob
import contextlib
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm

PWM_DIR = "./pwms/"
CTCF_PWM_FILE = "MA0139.1.pfm"
REST_PWM_FILE = "MA0138.2.pfm"

# ==========================================
# tqdm + joblib integration
# ==========================================
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# ==========================================

def load_pwm(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith(">"):
                numbers = [float(x) for x in line.split() if x.replace('.','',1).replace('-','',1).isdigit()]
                if numbers: matrix.append(numbers)
    if len(matrix) == 4:
        pfm = np.array(matrix).T 
        return np.log2((pfm + 0.1) / (pfm.sum(axis=1, keepdims=True) + 0.4) / 0.25)
    return None

def process_chunk(df_chunk, kmer_vocab, ctcf_pwm, rest_pwm, ep300_pwms):
    kmer_to_idx = {k: i for i, k in enumerate(kmer_vocab)}
    local_kmers, context_kmers = [], []
    base_cheats, ctcf_scores, rest_scores, ep300_scores = [], [], [], []
    
    for i, row in df_chunk.iterrows():
        seq = row['sequence'][:200].upper().ljust(200, 'N')
        left_seq = row['left_seq'].upper().ljust(200, 'N')
        right_seq = row['right_seq'].upper().ljust(200, 'N')
        ctx_seq = left_seq + seq + right_seq
        
        # 1. K-mers (K=4)
        def count_kmers(s):
            counts = np.zeros(256, dtype=np.float32)
            for j in range(len(s) - 3):
                kmer = s[j:j+4]
                if kmer in kmer_to_idx: counts[kmer_to_idx[kmer]] += 1
            return counts
            
        local_kmers.append(count_kmers(seq))
        context_kmers.append(count_kmers(ctx_seq))
        
        # 2. Base Cheat Codes
        length = len(seq)
        gc = (seq.count('G') + seq.count('C')) / max(length, 1)
        cpg = (seq.count('CG') * length) / max(seq.count('C') * seq.count('G'), 1)
        at_tracts = sum(1 for tract in seq.split('C') for t in tract.split('G') if len(t) >= 4)
        atac = 1.0 if ('ATAC' in row and row['ATAC'] != 'U') else 0.0
        base_cheats.append([atac, gc, cpg, at_tracts])
        
        # 3. PWM Scoring (Only on the 200bp local sequence)
        one_hot = np.zeros((length, 4))
        for j, nuc in enumerate(seq):
            if nuc in {'A': 0, 'C': 1, 'G': 2, 'T': 3}: one_hot[j, {'A': 0, 'C': 1, 'G': 2, 'T': 3}[nuc]] = 1
            
        def score_pwm(pwm):
            if pwm is None or length < pwm.shape[0]: return 0.0
            windows = np.lib.stride_tricks.sliding_window_view(one_hot, window_shape=(pwm.shape[0], 4)).reshape(-1, pwm.shape[0], 4)
            return np.max(np.einsum('nwc,wc->n', windows, pwm))
            
        ctcf_scores.append([score_pwm(ctcf_pwm)])
        rest_scores.append([score_pwm(rest_pwm)])
        ep_max = max([score_pwm(p) for p in ep300_pwms]) if ep300_pwms else 0.0
        ep300_scores.append([ep_max])

    return np.array(local_kmers), np.array(context_kmers), np.array(base_cheats), np.array(ctcf_scores), np.array(rest_scores), np.array(ep300_scores)

def compile_dataset(csv_path, out_prefix):
    print(f"\nLoading {csv_path} into memory...")
    df = pd.read_csv(csv_path)
    
    print("Mapping context groupings across genomic gaps...")
    df = df.sort_values(by=['chrom', 'start']).reset_index(drop=True)
    df['left_seq'] = np.where((df['chrom'] == df['chrom'].shift(1)) & (df['start'] == df['stop'].shift(1)), df['sequence'].shift(1), 'N'*200)
    df['right_seq'] = np.where((df['chrom'] == df['chrom'].shift(-1)) & (df['stop'] == df['start'].shift(-1)), df['sequence'].shift(-1), 'N'*200)
    
    kmer_vocab = [''.join(p) for p in product(['A', 'C', 'G', 'T'], repeat=4)]
    ctcf_pwm = load_pwm(os.path.join(PWM_DIR, CTCF_PWM_FILE))
    rest_pwm = load_pwm(os.path.join(PWM_DIR, REST_PWM_FILE))
    ep300_pwms = [load_pwm(f) for f in glob.glob(os.path.join(PWM_DIR, "*.pfm")) if os.path.basename(f) not in [CTCF_PWM_FILE, REST_PWM_FILE]]
    
    chunk_size = 5000
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    print(f"Beginning multiprocessing extraction for {out_prefix} data...")
    # Wrap the Parallel call with our tqdm context manager
    with tqdm_joblib(tqdm(desc=f"Compiling {out_prefix}", total=len(chunks), unit="chunk")) as pbar:
        results = Parallel(n_jobs=-1)(
            delayed(process_chunk)(chk, kmer_vocab, ctcf_pwm, rest_pwm, ep300_pwms) 
            for chk in chunks
        )
    
    print("Stacking arrays and saving to disk...")
    l_kmers = np.vstack([r[0] for r in results])
    c_kmers = np.vstack([r[1] for r in results])
    b_cheats = np.vstack([r[2] for r in results])
    c_scores = np.vstack([r[3] for r in results])
    r_scores = np.vstack([r[4] for r in results])
    e_scores = np.vstack([r[5] for r in results])
    
    os.makedirs("./data/processed", exist_ok=True)
    np.save(f"./data/processed/{out_prefix}_local_kmers.npy", l_kmers)
    np.save(f"./data/processed/{out_prefix}_context_kmers.npy", c_kmers)
    np.save(f"./data/processed/{out_prefix}_base_cheats.npy", b_cheats)
    np.save(f"./data/processed/{out_prefix}_ctcf_pwm.npy", c_scores)
    np.save(f"./data/processed/{out_prefix}_rest_pwm.npy", r_scores)
    np.save(f"./data/processed/{out_prefix}_ep300_pwm.npy", e_scores)
    print(f"Successfully saved all {out_prefix} arrays.")

if __name__ == "__main__":
    compile_dataset("./data/train_dataset.csv", "train")
    compile_dataset("./data/val_dataset.csv", "val")
    if os.path.exists("./data/test_dataset.csv"):
        compile_dataset("./data/test_dataset.csv", "test")