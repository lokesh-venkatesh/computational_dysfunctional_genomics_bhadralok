import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

class KmerSVMModel:
    def __init__(self, data, kmer_size=3, C=1.0):
        """
        Initialize the K-mer SVM Model.
        :param data: Training dataframe containing TF labels and a 'sequence' column.
        :param kmer_size: The length of the k-mers to use as features (default is 3).
        :param C: Regularization parameter for the SVM.
        """
        self.data = data
        self.k = kmer_size
        self.C = C
        self.tfs = ['CTCF', 'REST', 'EP300']
        self.models = {}
        
        # Scikit-learn's CountVectorizer returns sparse matrices perfectly suited for LinearSVC
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.k, self.k))

    def fit(self):
        """
        Converts sequences to k-mer features and trains a separate LinearSVC for each TF.
        """
        print(f"Extracting k-mer features (k={self.k}) for training...")
        sequences = self.data['sequence'].tolist()
        
        # This outputs a highly compressed Sparse Matrix. 
        # DO NOT convert this to a dense array (.toarray()), or you will run out of RAM!
        X_train = self.vectorizer.fit_transform(sequences)
        
        for tf in self.tfs:
            print(f"Training Linear SVM for {tf}...")
            y_train = (self.data[tf] != 'U').astype(int)
            
            # 1. Swapped SVC(kernel='linear') for LinearSVC
            # 2. dual=False makes optimization blazing fast when n_samples > n_features
            svm = LinearSVC(C=self.C, class_weight='balanced', dual=False, max_iter=2000)
            svm.fit(X_train, y_train)
            
            self.models[tf] = svm
            
        print(f"K-mer SVM Model (Linear) fitted successfully.")

    def infer(self, test_data):
        """
        Transforms test sequences into k-mer features and scores them.
        """
        if not self.models:
            raise ValueError("Model is not fitted yet. Call .fit() first.")
            
        print("Extracting k-mer features for inference...")
        test_sequences = test_data['sequence'].tolist()
        X_test = self.vectorizer.transform(test_sequences)
        
        predictions = {}
        
        for tf in self.tfs:
            scores = self.models[tf].decision_function(X_test)
            predictions[tf] = scores
            
        return pd.DataFrame(predictions, index=test_data.index)