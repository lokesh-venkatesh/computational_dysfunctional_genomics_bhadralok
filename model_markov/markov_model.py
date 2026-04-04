import math
import pandas as pd
from itertools import product

class MarkovModel:
    def __init__(self, data, order=1, pseudocount=1):
        """Initialize the Markov Model.
        :param data: Training dataframe containing TF labels and a 'sequence' column.
        :param order: The order of the Markov model (m).
        :param pseudocount: Pseudocount to prevent log(0) errors."""
        self.data = data # this will be the training dataset
        self.order = order # taken to be 1 by default
        self.pseudocount = pseudocount # pseudocount default value is also 1
        self.tfs = ['CTCF', 'REST', 'EP300']
        self.models = {}

    def _build_matrix(self, seqs):
        """INTERNAL method to calculate the transition matrix for a list of sequences.
        Adapted from nth_order_markov_matrix in the old script."""

        # ------------- for calculating zeroth order probabilities only -------------
        nucls = ['A','C','G','T']

        total_counts = {n: 0 for n in nucls}
        for seq in seqs:
            for nucl in nucls:
                total_counts[nucl] += seq.count(nucl)
                
        total = sum(total_counts.values())
        if total == 0: zeroth_order_probs = {n: 0.25 for n in nucls}
        else: zeroth_order_probs = {n: total_counts[n]/total for n in nucls}

        if self.order == 0: # Return the model early if it's a 0th order model
            return {0: zeroth_order_probs} 

        # But otherwise, proceed to build the m-th order transition matrix
        nmer_list = [''.join(p) for p in product(nucls, repeat=self.order)]
        nth_order_mm = {kmer: {n: self.pseudocount for n in nucls} for kmer in nmer_list}

        for seq in seqs:
            for i in range(0, len(seq) - self.order):
                kmer = seq[i:i+self.order]
                next_nucl = seq[i+self.order]
                if kmer in nth_order_mm and next_nucl in nucls: # Filter out anomalous chars
                    nth_order_mm[kmer][next_nucl] += 1

        for kmer, transitions in nth_order_mm.items(): # Normalize to get transition probs
            tot = sum(transitions.values())
            for n in nucls:
                transitions[n] /= tot

        nth_order_mm[0] = zeroth_order_probs
        return nth_order_mm

    def fit(self):
        """Builds positive and negative sequence transition models for each TF."""
        for tf in self.tfs:
            pos_seqs = self.data[self.data[tf] != 'U']['sequence'].tolist()
            neg_seqs = self.data[self.data[tf] == 'U']['sequence'].tolist()

            pos_model = self._build_matrix(pos_seqs)
            neg_model = self._build_matrix(neg_seqs)
            self.models[tf] = {'pos': pos_model, 'neg': neg_model}
            
        print(f"Markov Model (Order {self.order}) fitted.")
        # print(f"\n the models built for each TF: {self.models}")

    def _score_sequence(self, seq, pos_model, neg_model):
        """Calculates the log-odds score for a single sequence.
        score = log2( P(seq | pos) / P(seq | neg) )"""
        pos_score = 0.0
        neg_score = 0.0

        if self.order == 0:
            for char in seq:
                if char in pos_model[0] and char in neg_model[0]:
                    pos_score += math.log2(pos_model[0][char])
                    neg_score += math.log2(neg_model[0][char])
        else:
            # Score first 'm' characters using 0th order probabilities
            for i in range(self.order):
                char = seq[i]
                if char in pos_model[0] and char in neg_model[0]:
                    pos_score += math.log2(pos_model[0][char])
                    neg_score += math.log2(neg_model[0][char])
                    
            # Score remaining characters using m-th order transition matrix
            for i in range(self.order, len(seq)):
                prev_nmer = seq[i-self.order:i]
                char = seq[i]
                if prev_nmer in pos_model and char in pos_model[prev_nmer]:
                    pos_score += math.log2(pos_model[prev_nmer][char])
                    neg_score += math.log2(neg_model[prev_nmer][char])

        return pos_score - neg_score

    def infer(self, test_data):
        """Using the trained pos/neg models, score the test sequences.
        Returns continuous log-odds scores representing the prediction confidence."""
        
        if not self.models:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
            
        predictions = []
        for _, row in test_data.iterrows(): # iterating over the test dataset sequences
            seq = row['sequence']
            row_wise_preds = {}
            
            for tf in self.tfs:
                pos_model = self.models[tf]['pos']
                neg_model = self.models[tf]['neg']
                
                score = self._score_sequence(seq, pos_model, neg_model)
                row_wise_preds[tf] = score
                
            predictions.append(row_wise_preds)
            
        return pd.DataFrame(predictions, index=test_data.index)