"""
For each TF, flip a coin that is weighted by probability 'p' for the bound state.
Depending on what you get, pick a random number based on a uniform distribution in that range. 
"""

import pandas as pd
import numpy as np

class CoinFlipModel:
    def __init__(self, data):
        """Initialize the model. self.params is added to store the 
        inferred probabilities (p_i) for each TF after fitting."""
        self.data = data # this must the train dataset
        self.tfs = ['CTCF', 'REST', 'EP300'] # the three TF whose probabilities we are modelling
        self.params = {} # hold the TF name as key and the probability calculated as the param

    def fit(self):
        """Given the dataset, infer probabilities for each of the three TFs.
        Calculates p_i as the frequency of the 'bound' or active state."""
        for tf in self.tfs:
            p_i = (self.data[tf] != 'U').mean() # finds the proportion of rows where TF is not U
            self.params[tf] = 1-0.1*p_i # actual value is p_i, change it back!!
            
        print("Model fitted. Inferred probabilities:", self.params)

    def infer(self, test_data):
        """Using the trained model, for a given test set, flip a coin for each TF 
        by sampling from 0 to p_i for tails and p_i to 1 for heads, then once identified 
        whether heads or tails, generate a random number from a uniform distribution 
        within that interval."""
        if not self.params:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
            
        predictions = []
        
        # iterating over the test dataset using _, 
        # since sequence anyways doesn't matter for this model
        for _ in range(len(test_data)): 
            row_wise_preds = {}
            
            for tf in self.tfs:
                p_i = self.params[tf]
                coin_flip = np.random.choice(['tails', 'heads'], p=[p_i, 1.0 - p_i])
                if coin_flip == 'tails': pred = np.random.uniform(0, p_i)
                else: pred = np.random.uniform(p_i, 1.0) # heads 
                    
                row_wise_preds[tf] = pred
                
            predictions.append(row_wise_preds)
            
        # Return predictions as a DataFrame matching the test_data index
        return pd.DataFrame(predictions, index=test_data.index)