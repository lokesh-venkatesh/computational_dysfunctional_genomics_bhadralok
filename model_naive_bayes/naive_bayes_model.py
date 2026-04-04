import os
import re
import glob
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from joblib import Parallel, delayed
from scipy.special import logsumexp
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


class MixedNaiveBayes:
    """Fuses MultinomialNB (for discrete sparse features) and GaussianNB (for continuous dense features)."""
    def __init__(self):
        self.mnb = MultinomialNB() 
        self.gnb = GaussianNB() 

    def fit(self, X_discrete, X_continuous, y):
        if X_discrete.shape[1] > 0: self.mnb.fit(X_discrete, y) 
        if X_continuous.shape[1] > 0: self.gnb.fit(X_continuous, y) 

    def predict_log_probs(self, X_discrete, X_continuous): 
        jll_disc = self.mnb._joint_log_likelihood(X_discrete) if X_discrete.shape[1] > 0 else 0
        jll_cont = self.gnb._joint_log_likelihood(X_continuous) if X_continuous.shape[1] > 0 else 0
        
        if X_discrete.shape[1] > 0 and X_continuous.shape[1] > 0:
            joint_log_likelihood = jll_disc + jll_cont - self.mnb.class_log_prior_
        else:
            joint_log_likelihood = jll_disc + jll_cont 
            
        log_prob_x = logsumexp(joint_log_likelihood, axis=1) 
        return joint_log_likelihood - log_prob_x[:, np.newaxis]

    def predict_probs(self, X_discrete, X_continuous):
        return np.exp(self.predict_log_probs(X_discrete, X_continuous))


class NaiveBayesModel:
    def __init__(self, data, pwm_dir="./pwms/"):
        self.data = data
        self.pwm_dir = pwm_dir
        self.tfs = ['CTCF', 'REST', 'EP300']
        self.models = {}
        
        # ==========================================
        # TOGGLE FEATURES HERE FOR EXPERIMENTATION
        # ==========================================
        self.feature_flags = {
            'use_atac': True,        # ATAC-seq Boolean
            'use_gc_content': True,  # GC%
            'use_cpg_oe': True,      # CpG Observed/Expected
            'use_entropy': True,     # Shannon Entropy
            
            'use_pwm_max': True,     # Max PWM log-odds score
            'use_pwm_sum': True,     # Sum of positive PWM scores
            'use_pwm_hits': True     # Count of positive PWM hits
        }
        
        # Dynamically add toggles for 1-mers to 20-mers
        for k in range(1, 21):
            self.feature_flags[f'use_{k}mers'] = False
            
        # Set default active K-mers
        # self.feature_flags['use_3mers'] = True
        # self.feature_flags['use_4mers'] = True
        self.feature_flags['use_10mers'] = True
        
        # Initialize separate vectorizers for each requested K
        self.kmer_extractors = {}
        for k in range(1, 21):
            if self.feature_flags[f'use_{k}mers']:
                if k <= 8:
                    # CountVectorizer is perfectly fine for K <= 8
                    self.kmer_extractors[k] = CountVectorizer(analyzer='char', ngram_range=(k, k))
                else:
                    # HASHING TRICK: Prevents RAM overflow for massive K-mers (K > 8)
                    # alternate_sign=False ensures counts are strictly positive for Naive Bayes
                    self.kmer_extractors[k] = HashingVectorizer(
                        analyzer='char', ngram_range=(k, k), 
                        n_features=2**24, alternate_sign=False
                    )
                
        self.pwms = self._load_pwms() 

    def _load_pwms(self):
        """Parses .pfm files and converts Position Frequency Matrices to Log-Odds PWMs."""
        pwms = {}
        files = glob.glob(os.path.join(self.pwm_dir, "*.pfm"))
        
        for file in files:
            tf_name = os.path.basename(file).split('.')[0].upper()
            matrix = []
            
            with open(file, 'r') as f:
                for line in f:
                    if line.startswith(">"): continue
                    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if numbers:
                        matrix.append([float(x) for x in numbers])
            
            if len(matrix) == 4:
                pfm = np.array(matrix).T 
                pseudocount = 0.1
                col_sums = pfm.sum(axis=1, keepdims=True)
                pwms[tf_name] = np.log2((pfm + pseudocount) / (col_sums + 4 * pseudocount) / 0.25)
            else:
                print(f"Warning: {file} had {len(matrix)} rows instead of 4. Skipping.")
                
        print(f"Loaded {len(pwms)} PWMs from {self.pwm_dir}/")
        return pwms

    @staticmethod
    def _extract_worker(sequences, atac_labels, pwms, flags):
        """Worker function for parallel feature extraction over a chunk of data."""
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        chunk_discrete = []
        chunk_continuous = []
        
        for seq, atac in zip(sequences, atac_labels):
            seq = seq.upper()
            length = len(seq)
            
            counts = Counter(seq)
            if flags['use_gc_content']:
                gc_content = (counts.get('G', 0) + counts.get('C', 0)) / length
                
            if flags['use_cpg_oe']:
                cpg_count = seq.count('CG')
                cpg_oe = (cpg_count * length) / max(counts.get('C', 0) * counts.get('G', 0), 1)
                
            if flags['use_entropy']:
                kmers_3 = [seq[i:i+3] for i in range(length - 2)]
                probs = [c / len(kmers_3) for c in Counter(kmers_3).values()]
                entropy = -sum(p * math.log2(p) for p in probs) if probs else 0
            
            pwm_maxes, pwm_sums, pwm_hits = [], [], []
            if pwms and (flags['use_pwm_max'] or flags['use_pwm_sum'] or flags['use_pwm_hits']):
                one_hot = np.zeros((length, 4))
                for i, nuc in enumerate(seq):
                    if nuc in nuc_map: one_hot[i, nuc_map[nuc]] = 1
                    
                for name, pwm in pwms.items():
                    w_len = pwm.shape[0]
                    windows = np.lib.stride_tricks.sliding_window_view(one_hot, window_shape=(w_len, 4)).squeeze()
                    scores = np.einsum('nwc,wc->n', windows, pwm)
                    
                    if flags['use_pwm_max']: pwm_maxes.append(np.max(scores))
                    if flags['use_pwm_sum']: pwm_sums.append(np.sum(scores[scores > 0]) if len(scores[scores > 0]) > 0 else 0)
                    if flags['use_pwm_hits']: pwm_hits.append(len(scores[scores > 0]))

            discrete = []
            continuous = []
            
            if flags['use_atac']:       discrete.append(atac)
            if flags['use_pwm_hits']:   discrete.extend(pwm_hits)
                
            if flags['use_gc_content']: continuous.append(gc_content)
            if flags['use_cpg_oe']:     continuous.append(cpg_oe)
            if flags['use_entropy']:    continuous.append(entropy)
            if flags['use_pwm_max']:    continuous.extend(pwm_maxes)
            if flags['use_pwm_sum']:    continuous.extend(pwm_sums)
            
            chunk_discrete.append(discrete)
            chunk_continuous.append(continuous)
            
        return np.array(chunk_discrete), np.array(chunk_continuous)

    def extract_all_features(self, df):
        """Orchestrates parallel extraction and highly-efficient sparse K-mer counting."""
        print(f"Parallel computing enabled global & PWM features...")
        sequences = df['sequence'].tolist()
        atac_labels = (df['ATAC'] != 'U').astype(int).tolist()
        
        n_jobs = -1 
        chunk_size = max(1, math.ceil(len(sequences) / (os.cpu_count() or 1)))
        seq_chunks = [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]
        atac_chunks = [atac_labels[i:i + chunk_size] for i in range(0, len(atac_labels), chunk_size)]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_worker)(s_chunk, a_chunk, self.pwms, self.feature_flags) 
            for s_chunk, a_chunk in zip(seq_chunks, atac_chunks)
        )
        
        discrete_feats = np.vstack([res[0] for res in results]) if results[0][0].shape[1] > 0 else np.empty((len(sequences), 0))
        continuous_feats = np.vstack([res[1] for res in results]) if results[0][1].shape[1] > 0 else np.empty((len(sequences), 0))
        
        sparse_matrices = []
        for k, extractor in self.kmer_extractors.items():
            print(f"Extracting {k}-mer matrix...")
            # HashingVectorizer doesn't have a vocabulary to fit, it just transforms natively
            if isinstance(extractor, HashingVectorizer):
                sparse_matrices.append(extractor.transform(df['sequence']))
            else:
                if not hasattr(extractor, 'vocabulary_'):
                    sparse_matrices.append(extractor.fit_transform(df['sequence']))
                else:
                    sparse_matrices.append(extractor.transform(df['sequence']))

        if sparse_matrices:
            if discrete_feats.shape[1] > 0:
                discrete_feats_sparse = sp.csr_matrix(discrete_feats)
                final_discrete = sp.hstack(sparse_matrices + [discrete_feats_sparse]).tocsr()
            else:
                final_discrete = sp.hstack(sparse_matrices).tocsr()
        else:
            final_discrete = sp.csr_matrix(discrete_feats)
        
        return final_discrete, continuous_feats

    def fit(self):
        print("\n--- Training Naive Bayes Models ---")
        X_discrete, X_continuous = self.extract_all_features(self.data)
        
        for tf in self.tfs:
            print(f"Fitting Mixed NB for {tf}...")
            y = (self.data[tf] != 'U').astype(int)
            
            model = MixedNaiveBayes()
            model.fit(X_discrete, X_continuous, y)
            self.models[tf] = model

    def infer(self, test_data):
        print("\n--- Inference ---")
        if not self.models:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
            
        X_discrete, X_continuous = self.extract_all_features(test_data)
        predictions = {}
        
        for tf in self.tfs:
            probs = self.models[tf].predict_probs(X_discrete, X_continuous)[:, 1]
            predictions[tf] = probs
            
        return pd.DataFrame(predictions, index=test_data.index)