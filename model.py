import os
import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold

def import_dataset(filepath="dataset.csv", TF_name="CTCF", chromosome_number="All"):
    """
    INPUT: none (unless filepath changes)
    OUTPUT: a list of strings (DNA sequences) all of equal lengths
            corresponding to the positive/negative sequences
    """
    df = pd.read_csv(filepath, header=0, index_col=0, low_memory=False)
    df = df[~df['chrom'].isin(['chr3', 'chr10', 'chr17'])]
    if chromosome_number in range(1,23): # if a specific chromosome number is provided as input
        df = df[df['chrom'].isin([f'chr{chromosome_number}'])]
        # otherwise it just returns all sequences from the 19 chromosomes
    pos_dataset = df[df[TF_name] == 'B']['sequence'].tolist()
    neg_dataset = df[df[TF_name] == 'U']['sequence'].tolist()
    return pos_dataset, neg_dataset

def nth_order_markov_matrix(m, seqs, pseudcnt=0.1):
    """
    INPUTS:
    1. An integer 'n', that should lie between 0 and the length L of the sequences
    2. A list of sequences, all of equal length L

    OUTPUT: a matrix corresponding to the transition probabilities between different states
    (this will basically be a (4^n+1)*4 matrix,
    the +1 coming from the zeroth order probabilities for the start nucleotide of each sequence)

    hint: a_XY = N_XY/N_X (where X and Y are nucleotides)

    NOTE: make sure to include pseudocounts... this will definitely matter for larger model orders 'm'.
    """

    nucls = ['A','C','G','T']
    zeroth_order_probabilities = {nucl: list(''.join(seqs)).count(nucl) for nucl in nucls}
    norm = sum(list(zeroth_order_probabilities.values()))
    zeroth_order_probabilities = {i:zeroth_order_probabilities[i]/norm for i in list(zeroth_order_probabilities.keys())}

    if m==0:
        return {0: zeroth_order_probabilities}

    elif m>=1:
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

        nmer_list = kmer_list(m)
        nth_order_mm = {kmer: {nucl:pseudcnt for nucl in nucls} for kmer in nmer_list}

        count = 0
        for seq in seqs:
            count+=1
            if count%1000==0: print(f"{count} sequences done")

            for i in range(0, len(seq)-m):
                kmer = seq[i:i+m]
                next_nucl = seq[i+m]
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
    plt.savefig(f"results/{n}-th_order_MM_scores_distribution.png", dpi=300)
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {m}-th order markov model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"results/{m}-th_order_MM_ROC_curve.png", dpi=300)
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

def plot_PRC(dict_scores):
    """
    sklearn prc used to plot precision recall curve

    """
    scores=list(dict_scores.keys())
    true_labels=list(dict_scores.values())

    precision, recall, thresholds=precision_recall_curve(true_labels, scores)
    prc_auc=auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (AUC = {prc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PRC Curve for {m}-th order markov model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"results/{m}-th_order_MM_PRC_curve.png", dpi=300)
    plt.close()

def calculate_AUC_PRC(dict_scores):
    """
    Use sklearn auc to calculate area under the PRC
    """
    scores=list(dict_scores.keys())
    true_labels=list(dict_scores.values())

    precision, recall, thresholds=precision_recall_curve(true_labels, scores)
    prc_auc=auc(recall, precision)
    return prc_auc

if __name__ == "__main__":
    m = 8 # m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    k = 3
    c = 22 # modify such that these are all taken in as CLI arguments
    TF = "CTCF"

    """NOTE: CROSS-VALIDATION IS YET TO BE IMPLEMENTED IN THIS CODE"""
    '''PRC implementation inserted by me,'''

    os.makedirs("results", exist_ok=True)
    pos_dataset, neg_dataset = import_dataset(TF_name=TF, chromosome_number="All")

    # print(df.iloc[[394517, 394518, 394519, 394520, 394570, 394571, 394572, 394573, 3403336], -1])
    # NOTE: the above nine entries have been removed from the original .tsv files since they DO contain 'N's
    # or in other words, AMBIGOUS NUCLEOTIDE READS!

    pos_model = nth_order_markov_matrix(m, pos_dataset)
    neg_model = nth_order_markov_matrix(m, neg_dataset)

    dict_of_scores = return_dict_of_scores(m, pos_dataset, neg_dataset, pos_model, neg_model)

    plot_distributions(m, dict_scores=dict_of_scores)
    plot_ROC(dict_scores=dict_of_scores)
    AUC_ROC_val = calculate_AUC_ROC(dict_scores=dict_of_scores)

    plot_PRC(dict_scores=dict_of_scores)
    AUC_PRC_val = calculate_AUC_PRC(dict_scores=dict_of_scores) # what do with this, since you said output of this script should be AUC ROC?

    print(AUC_ROC_val)
