import os
import re
import glob
import math
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from itertools import product

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from tqdm import tqdm

class KmerSVMModel:
    def __init__(self, kernel='linear', kmer_size=5, max_iter=2000, n_components=500, pwm_dir="./pwms/"):
        """
        kernel: 'linear' or 'rbf_approx' (Uses Nystroem approximation for speed)
        n_components: Number of Monte Carlo samples for the RBF approximation
        """
        self.kernel = kernel
        self.kmer_size = kmer_size
        self.max_iter = max_iter
        self.n_components = n_components
        self.pwm_dir = pwm_dir
        self.tfs = ['CTCF', 'REST', 'EP300']
        self.models = {}
        
        # Explicit Vocabulary to prevent the "Missing K-mer" bug
        nucls = ['A', 'C', 'G', 'T']
        vocab = [''.join(p) for p in product(nucls, repeat=self.kmer_size)]
        
        self.vectorizer = CountVectorizer(
            analyzer='char', 
            ngram_range=(self.kmer_size, self.kmer_size), 
            vocabulary=vocab,
            lowercase=False
        )
        
        self.pwms = self._load_pwms()
        self.scaler = StandardScaler()

    def _load_pwms(self):
        pwms = {}
        files = glob.glob(os.path.join(self.pwm_dir, "*.pfm"))
        for file in files:
            tf_name = os.path.basename(file).split('.')[0].upper()
            matrix = []
            with open(file, 'r') as f:
                for line in f:
                    if line.startswith(">"): continue
                    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if numbers: matrix.append([float(x) for x in numbers])
            if len(matrix) == 4:
                pfm = np.array(matrix).T 
                pseudocount = 0.1
                col_sums = pfm.sum(axis=1, keepdims=True)
                pwms[tf_name] = np.log2((pfm + pseudocount) / (col_sums + 4 * pseudocount) / 0.25)
        return pwms

    @staticmethod
    def _extract_worker(sequences, pwms):
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        chunk_features = []
        for seq in sequences:
            seq = seq.upper()
            length = len(seq)
            counts = Counter(seq)
            
            gc_content = (counts.get('G', 0) + counts.get('C', 0)) / length
            cpg_oe = (seq.count('CG') * length) / max(counts.get('C', 0) * counts.get('G', 0), 1)
            
            kmers_3 = [seq[i:i+3] for i in range(length - 2)]
            probs = [c / len(kmers_3) for c in Counter(kmers_3).values()]
            entropy = -sum(p * math.log2(p) for p in probs) if probs else 0
            
            features = [gc_content, cpg_oe, entropy]
            
            if pwms:
                one_hot = np.zeros((length, 4))
                for i, nuc in enumerate(seq):
                    if nuc in nuc_map: one_hot[i, nuc_map[nuc]] = 1
                    
                for name, pwm in pwms.items():
                    w_len = pwm.shape[0]
                    windows = np.lib.stride_tricks.sliding_window_view(one_hot, window_shape=(w_len, 4)).squeeze()
                    scores = np.einsum('nwc,wc->n', windows, pwm)
                    features.extend([np.max(scores), np.sum(scores[scores > 0]) if len(scores[scores > 0]) > 0 else 0, len(scores[scores > 0])])
                    
            chunk_features.append(features)
        return np.array(chunk_features, dtype=np.float32)

    def extract_features(self, df, is_train=True):
        print(f"Extracting {self.kmer_size}-mers...")
        if is_train:
            counts = self.vectorizer.fit_transform(df['sequence']).toarray().astype(np.float32)
        else:
            counts = self.vectorizer.transform(df['sequence']).toarray().astype(np.float32)
        
        # Log1p transforms counts to prevent massive spikes
        counts_log = np.log1p(counts)
        
        print("Extracting biological cheat codes (Parallel)...")
        sequences = df['sequence'].tolist()
        n_jobs = -1 
        chunk_size = max(1, math.ceil(len(sequences) / (os.cpu_count() or 1)))
        seq_chunks = [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_worker)(chunk, self.pwms) for chunk in seq_chunks
        )
        cheat_features = np.vstack(results)
        atac_labels = (df['ATAC'] != 'U').astype(np.float32).values.reshape(-1, 1)
        
        # Combine all features
        X_raw = np.hstack([counts_log, cheat_features, atac_labels])
        
        # SVMs are HIGHLY sensitive to unscaled data. We MUST scale it.
        print("Scaling features...")
        if is_train:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            X_scaled = self.scaler.transform(X_raw)
            
        return X_scaled

    def fit(self, df_train, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        X_train = self.extract_features(df_train, is_train=True)
        
        for tf in self.tfs:
            print(f"\n--- Training Fast SVM for {tf} ---")
            y_train = (df_train[tf] != 'U').astype(int).values
            
            # Added C=0.1 (stronger regularization) and tol=1e-3 for blazing fast convergence
            base_svm = LinearSVC(class_weight='balanced', max_iter=self.max_iter, dual=False, C=0.1, tol=1e-3)
            
            if self.kernel == 'rbf_approx':
                print("Using Nystroem Non-Linear RBF Approximation...")
                clf = Pipeline([
                    ('nystroem', Nystroem(gamma=0.2, n_components=self.n_components, random_state=42)),
                    ('svm', base_svm)
                ])
            else:
                print("Using pure Linear Kernel...")
                clf = base_svm
                
            # No more slow CalibratedClassifier! Just fit the raw SVM directly.
            clf.fit(X_train, y_train)
            
            self.models[tf] = clf
            
            # Save the model
            model_path = os.path.join(save_dir, f"svm_{tf}.pkl")
            joblib.dump(clf, model_path)
            print(f"Saved {tf} model to {model_path}")
            
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, "svm_scaler.pkl"))

    def infer(self, df_test):
        if not self.models:
            raise ValueError("Models not loaded or trained!")
            
        X_test = self.extract_features(df_test, is_train=False)
        predictions = {}
        
        for tf in self.tfs:
            print(f"Predicting {tf}...")
            # Use raw distance from the hyperplane instead of slow probabilities
            scores = self.models[tf].decision_function(X_test)
            predictions[tf] = scores
            
        return pd.DataFrame(predictions, index=df_test.index)