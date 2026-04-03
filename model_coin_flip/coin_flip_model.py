import pandas as pd
import numpy as np

class CoinFlipModel:
    def __init__(self, data):
        """
        Initialize the model. self.params is added to store the 
        inferred probabilities (p_i) for each TF after fitting.
        """
        self.data = data
        self.tfs = ['CTCF', 'REST', 'EP300']
        self.params = {} 

    def fit(self):
        """
        Given the dataset, infer probabilities for each of the three TFs.
        Calculates p_i as the frequency of the 'bound' or active state.
        """
        for tf in self.tfs:
            # Assuming 'U' means Unbound. We calculate the proportion of instances 
            # that are NOT 'U' (adjust this condition if your active label is exactly 'B' or 1)
            p_i = (self.data[tf] != 'U').mean()
            self.params[tf] = p_i
            
        print("Model fitted. Inferred probabilities:", self.params)

    def infer(self, test_data):
        """
        Using the trained model, for a given test set, flip a coin for each TF 
        by sampling from 0 to p_i for tails and p_i to 1 for heads, then once identified 
        whether heads or tails, generate a random number from a uniform distribution 
        within that interval.
        """
        if not self.params:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
            
        predictions = []
        
        # Iterate over the test dataset
        for _ in range(len(test_data)):
            row_preds = {}
            
            for tf in self.tfs:
                p_i = self.params[tf]
                
                # The prompt implies P(Tails) = p_i (interval size p_i) 
                # and P(Heads) = 1 - p_i (interval size 1 - p_i)
                # Flip the coin based on these weights:
                coin_flip = np.random.choice(['tails', 'heads'], p=[p_i, 1.0 - p_i])
                
                # Sample from the uniform distribution within the designated interval
                if coin_flip == 'tails':
                    val = np.random.uniform(0, p_i)
                else: # heads
                    val = np.random.uniform(p_i, 1.0)
                    
                row_preds[tf] = val
                
            predictions.append(row_preds)
            
        # Return predictions as a DataFrame matching the test_data index
        return pd.DataFrame(predictions, index=test_data.index)