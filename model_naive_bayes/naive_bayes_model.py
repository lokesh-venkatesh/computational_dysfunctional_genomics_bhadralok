import math
import pandas as pd
from itertools import product

class NaiveBayesModel:
    def __init__(self, data, kmer_size=3, pseudocount=1.0):
        """
        Initialize the Naive Bayes Model.
        :param data: Training dataframe containing TF labels and a 'sequence' column.
        :param kmer_size: The length of the k-mers to use as features (default is 3).
        :param pseudocount: Laplace smoothing parameter to prevent log(0) errors.
        """
        self.data = data
        self.k = kmer_size
        self.pseudocount = pseudocount
        self.tfs = ['CTCF', 'REST', 'EP300']
        self.models = {}
        
        # Pre-generate all possible k-mers for the vocabulary
        nucls = ['A', 'C', 'G', 'T']
        self.vocab = [''.join(p) for p in product(nucls, repeat=self.k)]

    def _extract_kmers(self, seq):
        """Helper to break a sequence into overlapping k-mers."""
        return [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]

    def _train_class(self, sequences):
        """
        Calculates k-mer probabilities for a specific class (Bound or Unbound).
        """
        # Initialize counts with pseudocounts (Laplace smoothing)
        kmer_counts = {kmer: self.pseudocount for kmer in self.vocab}
        
        # Count all k-mers across all sequences in this class
        total_kmers = len(self.vocab) * self.pseudocount
        for seq in sequences:
            for kmer in self._extract_kmers(seq):
                if kmer in kmer_counts:
                    kmer_counts[kmer] += 1
                    total_kmers += 1
                    
        # Convert counts to log probabilities
        log_probs = {kmer: math.log(count / total_kmers) for kmer, count in kmer_counts.items()}
        return log_probs

    def fit(self):
        """
        Given the dataset, calculate priors and likelihoods for each TF.
        """
        for tf in self.tfs:
            # Separate sequences into positive (Bound) and negative (Unbound)
            pos_seqs = self.data[self.data[tf] != 'U']['sequence'].tolist()
            neg_seqs = self.data[self.data[tf] == 'U']['sequence'].tolist()
            
            total_seqs = len(pos_seqs) + len(neg_seqs)
            
            # 1. Calculate Log Priors: P(Class)
            # We use log priors to prevent underflow later
            log_prior_pos = math.log(len(pos_seqs) / total_seqs) if len(pos_seqs) > 0 else float('-inf')
            log_prior_neg = math.log(len(neg_seqs) / total_seqs) if len(neg_seqs) > 0 else float('-inf')
            
            # 2. Calculate Log Likelihoods: P(k-mer | Class)
            pos_log_probs = self._train_class(pos_seqs)
            neg_log_probs = self._train_class(neg_seqs)
            
            self.models[tf] = {
                'log_prior_pos': log_prior_pos,
                'log_prior_neg': log_prior_neg,
                'pos_log_probs': pos_log_probs,
                'neg_log_probs': neg_log_probs
            }
            
        print(f"Naive Bayes Model (k-mer size = {self.k}) fitted successfully.")

    def infer(self, test_data):
        """
        Scores test sequences using the trained Naive Bayes equations.
        Outputs Log-Odds scores: Log( P(Bound|Seq) / P(Unbound|Seq) )
        """
        if not self.models:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
            
        predictions = []
        
        for _, row in test_data.iterrows():
            seq = row['sequence']
            kmers_in_seq = self._extract_kmers(seq)
            row_preds = {}
            
            for tf in self.tfs:
                model = self.models[tf]
                
                # Start with the log priors
                score_pos = model['log_prior_pos']
                score_neg = model['log_prior_neg']
                
                # Add log probabilities of each k-mer observed in the sequence
                for kmer in kmers_in_seq:
                    if kmer in self.vocab:
                        score_pos += model['pos_log_probs'][kmer]
                        score_neg += model['neg_log_probs'][kmer]
                
                # Log Odds: Positive Score - Negative Score
                # >0 means predicted Bound, <0 means predicted Unbound
                # This continuous score acts as a perfect prediction confidence for ROC/PRC
                log_odds = score_pos - score_neg
                row_preds[tf] = log_odds
                
            predictions.append(row_preds)
            
        return pd.DataFrame(predictions, index=test_data.index)