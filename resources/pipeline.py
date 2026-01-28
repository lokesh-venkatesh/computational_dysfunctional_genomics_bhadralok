import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def import_dataset(filepath):
    """
    INPUT: none (unless filepath changes)
    OUTPUT: a list of strings (DNA sequences) all of equal lengths
            corresponding to the positive/negative sequences
    """
    pos_dataset = []
    with open(filepath, 'r') as file:
        for line in file:
            raw_seq = line.strip()
            raw_seq = ''.join(char for char in raw_seq if char in 'ATGC')
            pos_dataset.append(raw_seq)
    return pos_dataset

def nth_order_markov_matrix(n, seqs, pseudcnt=0.1):
    """
    INPUTS: 
    1. An integer 'n', that should lie between 0 and the length L of the sequences
    2. A list of sequences, all of equal length L

    OUTPUT: a matrix corresponding to the transition probabilities between different states
    (this will basically be a (4^n+1)*4 matrix, 
    the +1 coming from the zeroth order probabilities for the start nucleotide of each sequence)
    
    hint: a_XY = N_XY/N_X (where X and Y are nucleotides)

    NOTE: make sure to include pseudocounts... this will definitely matter for larger 'n'.
    """
    nucls = ['A','C','G','T']
    zeroth_order_probabilities = {nucl: list(''.join(seqs)).count(nucl) for nucl in nucls}
    norm = sum(list(zeroth_order_probabilities.values()))
    zeroth_order_probabilities = {i:zeroth_order_probabilities[i]/norm for i in list(zeroth_order_probabilities.keys())}

    if n==0:
        return {0: zeroth_order_probabilities}
    
    elif n>=1:
        def kmer_list(n):
            dummy_list = ['A','C','G','T']
            if n==0:
                return dummy_list
            elif n>=1:
                new_list = [i for i in dummy_list]
                for j in range(n-1):
                    placeholder_list = []
                    for nmer in new_list:
                        placeholder_list.extend([nmer+nucl for nucl in dummy_list])
                    new_list = placeholder_list
                return new_list
        
        nmer_list = kmer_list(n)
        nth_order_mm = {kmer: {nucl:pseudcnt for nucl in nucls} for kmer in nmer_list} 

        for seq in seqs:
            for i in range(0, len(seq)-n):
                kmer = seq[i:i+n]
                next_nucl = seq[i+n]
                nth_order_mm[kmer][next_nucl] += 1
        
        for kmer in nmer_list:
            normalisation_constant = 0
            for nucl in nucls:
                normalisation_constant += nth_order_mm[kmer][nucl]
            for nucl in nucls:
                nth_order_mm[kmer][nucl] = nth_order_mm[kmer][nucl]/normalisation_constant
        
        nth_order_mm[0] = zeroth_order_probabilities
        
        return nth_order_mm

def return_dict_of_scores(n, pos_seqs, neg_seqs, pos_model, neg_model):
    """
    INPUT: positive and negative sequences as well as the two corresponding models
    OUTPUT: a dictionary where the keys are the scores and the valeus are the labels, positive/negative
    """
    scores_dict = {}

    def calculate_score(n, seq, pos_model, neg_model):
        """
        INPUT: a sequence (the label is monitored for outside this function),
        and a positive and negative model/matrix
        OUTPUT: score = log_2(pos_model(seq)/neg_model(seq))
        """
        pos_score = 0
        neg_score = 0

        if n==0:
            for i in range(len(seq)):
                char = seq[i]
                pos_score += math.log2(pos_model[0][char])
                neg_score += math.log2(neg_model[0][char])
        elif n>=1:
            for i in range(n):
                char = seq[i]
                pos_score += math.log2(pos_model[0][char])
                neg_score += math.log2(neg_model[0][char])
            for i in range(n, len(seq)):
                prev_nmer = seq[i-n:i]
                char = seq[i]
                pos_score += math.log2(pos_model[prev_nmer][char])
                neg_score += math.log2(neg_model[prev_nmer][char])
        
        total_score = pos_score-neg_score
        return total_score

    for pos_seq in pos_seqs:
        scores_dict[calculate_score(n, pos_seq, pos_model, neg_model)] = 1
    for neg_seq in neg_seqs:
        scores_dict[calculate_score(n, neg_seq, pos_model, neg_model)] = 0
    return scores_dict

def plot_distributions(n, dict_scores):
    """
    INPUT: dictionary of scores and the order of the markov model used
    OUTPUT: a plot generated of the distributions of scores (and coloured accordingly), 
    and locally saved with some appropriate name
    """
    pos_scores = np.array([score for score in list(dict_scores.keys()) if dict_scores[score]==1])
    neg_scores = np.array([score for score in list(dict_scores.keys()) if dict_scores[score]==0])

    """
    # NOTE UNCOMMENT IF YOU WANT BIN WIDTH TO BE UNIFORM FOR BOTH HISTOGRAM DISTRIBUTIONS ON THE SAME CHART
    bin_width = 1
    min_val = min(pos_scores.min(), neg_scores.min())
    max_val = max(pos_scores.max(), neg_scores.max())
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    """

    bins = 10

    plt.figure(figsize=(8, 6))
    plt.hist(pos_scores, bins=bins, alpha = 0.5, color="blue", edgecolor="black", label = "Positive Sequences")
    plt.hist(neg_scores, bins=bins, alpha = 0.5, color="red", edgecolor="black", label = "Negative Sequences")

    plt.title("Distribution of scores, coloured by whether positive or negative sequence")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(f"score_distribution_plots/{n}-th_order_MM_scores_distribution.png", dpi=300)
    plt.close()

def plot_ROC(dict_scores):
    """
    We will either hard code this from scratch, or use some standard library like scikit-learn.
    Make sure we save this plot locally as well
    """
    scores = list(dict_scores.keys())
    true_labels = list(dict_scores.values())

    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {n}-th order markov model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"ROC_plots/{n}-th_order_MM_ROC_curve.png", dpi=300)
    plt.close()
        
def calculate_AUC_ROC(dict_scores):
    """
    Again, we can scam this by using scikit-learn functions.
    
    NOTE: ensure that the AUC_ROC is printed out as the final and ONLY output of this entire file.
    This will be useful when we run the mainpipeline across different values of 'n'.
    """
    scores = list(dict_scores.keys())
    true_labels = list(dict_scores.values())

    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    """ORDER CHOSEN FOR THE MARKOV MODEL"""

    pos_dataset = import_dataset(filepath="datasets/pos_sequences.fasta")
    neg_dataset = import_dataset(filepath="datasets/neg_sequences.fasta")

    pos_model = nth_order_markov_matrix(n, pos_dataset)
    neg_model = nth_order_markov_matrix(n, neg_dataset)

    dict_of_scores = return_dict_of_scores(n, pos_dataset, neg_dataset, pos_model, neg_model)
    
    plot_distributions(n, dict_scores=dict_of_scores)
    plot_ROC(dict_scores=dict_of_scores)

    AUC_ROC_val = calculate_AUC_ROC(dict_scores=dict_of_scores)
    print(AUC_ROC_val)