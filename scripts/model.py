"""This is the main script for this entire project repo, at least for pre-midterm.

Function-descriptions have been included with some helpful comments, 
The seed (right below the library imports) may be changed to introduce stochasticity into the code.

All outputs (generated data) for a run with a particular set of hyperparameters will be saved to ./outputs/chr{c}_order{m}_fold{k}_TF{TF}/
where c, m, k and TF are the corresponding hyperparameters passed as input for one run"""

import os
import math
import sys
import time
import subprocess
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold

pipeline_seed = 42 # CHANGE THIS VALUE FOR STOCHASTICITY IN THE PIPELINE

if not os.path.exists("data/dataset.csv"):
    subprocess.run([sys.executable, "data_as_one_tsv.py"])

def import_dataset(filepath="data/dataset.csv", TF_name="CTCF", chromosome_number="All"):
    """ INPUT: none (default .tsv file has been chosen already)
        OUTPUT: a list of strings (DNA sequences/Chip-Seq bins) all of equal lengths
            corresponding to the positive/negative sequences"""
    df = pd.read_csv(filepath, header=0, index_col=False, low_memory=False)
    df = df.loc[~df['chrom'].isin({"chr3", "chr10", "chr17"})]  # removes the three chromosomes that we are supposed to find predictions for
    if chromosome_number in range(1,23): # if a specific chromosome number is provided as input
        df = df[df['chrom'].isin([f'chr{chromosome_number}'])]
        # otherwise it just returns all sequences from the 19 chromosomes, 
        # if "All" or None is the input... NOTE need to incorporate this too 
    pos_dataset = df[df[TF_name] == 'B']['sequence'].tolist() # For a specific transcription factor
    neg_dataset = df[df[TF_name] == 'U']['sequence'].tolist() # Again, for a specific transcription factor
    return pos_dataset, neg_dataset

def nth_order_markov_matrix(m, seqs, filepath, pseudcnt=0.1):
    """ INPUTS: order of the markov model matrix and the dataset, as well as the filepath to savethe model to
        OUTPUT: a matrix corresponding to the transition probabilities between different states, as a dictionary"""

    nucls = ['A', 'C', 'G', 'T']  # initalising four possible nucleotides

    # we first calculate the vector of zeroth order transition probabilities
    total_counts = {'A':0 , 'C':0 , 'G':0 , 'T':0}
    for seq in seqs: # there is already one loop through all of the dataset right here
        for nucl in nucls:
            total_counts[nucl] += seq.count(nucl)
    total = sum(total_counts[nucl] for nucl in nucls)
    zeroth_order_probabilities = {n: total_counts[n] / total for n in nucls}

    # This is for the zero-th order case
    if m == 0:
        mm_model = {0: zeroth_order_probabilities}
        pd.DataFrame.from_dict(zeroth_order_probabilities, orient='index', columns=['probability']).to_csv(filepath, sep="\t")
        return mm_model

    # This is for the n-th order case, where n >= 1
    nmer_list = [''.join(p) for p in product(nucls, repeat=m)] # line of code obtained from the internet - works perfectly, dw
    nth_order_mm = {kmer: {n: pseudcnt for n in nucls} for kmer in nmer_list} # here we initialise the transition matrix

    for seq in seqs: # Now we have a second loop through all of the dataset right here
        for i in range(0, len(seq)-m): # 200 being len(seq)
            kmer = seq[i:i+m]
            next_nucl = seq[i+m]
            row = nth_order_mm[kmer]
            row[next_nucl] += 1

    # Normalise each of these transition probability vectors
    for kmer, transitions in nth_order_mm.items(): # of the order of 4^m, where m is the order of the markov model matrix
        total = sum(transitions.values())
        for n in nucls:
            transitions[n] /= total # Doing the actual normalisation

    # Add zeroth order probabilities to the rest of the n-th order transition rows
    nth_order_mm[0] = zeroth_order_probabilities
    df = pd.DataFrame.from_dict(nth_order_mm, orient='index')
    df.to_csv(filepath, sep="\t")

    return nth_order_mm

def return_array_of_scores(n, pos_seqs, neg_seqs, pos_model, neg_model, folder_path):
    """ INPUT: positive and negative sequences as well as the positive and negative trained-models
    OUTPUT: a dictionary where the keys are the scores and the valeus are the labels, either positive or negative"""

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

    score_array = {}

    for pos_seq in pos_seqs:
        score_array[calculate_score(n, pos_seq, pos_model, neg_model)] = 1
    for neg_seq in neg_seqs:
        score_array[calculate_score(n, neg_seq, pos_model, neg_model)] = 0
    
    score_series = pd.Series(score_array)
    os.makedirs(folder_path, exist_ok=True)
    score_series.to_csv(f"{folder_path}/array_of_all_scores.tsv", sep="\t")
    return score_array

def calculate_AUC_ROC(array_scores):
    """Calculating the AUC-ROC for a given dictionary of predicted scores and the true labels
    This is done for both positive and negative sequences, and we used scikit-learn for this"""
    scores = list(array_scores.keys())
    true_labels = list(array_scores.values())

    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def plot_ROC(array_scores, roc_auc, m, c, TF, i, k, folder_path):
    """Plotted the Receiver-Operating Characteristic Curve, again using scikit-learn"""
    scores = list(array_scores.keys())
    true_labels = list(array_scores.values())
    fpr, tpr, thresholds = roc_curve(true_labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {m}-th order markov model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{folder_path}/{m}-th_order_{i}-{k}-th_fold_MM_ROC_curve_for_chromosome_{c}_TF_{TF}.png", dpi=300)
    plt.close()

def calculate_AUC_PRC(array_scores):
    """Calculating the AUC-PRC for a given dictionary of predicted scores and the true labels
    This is done for both positive and negative sequences, and we used scikit-learn for this"""
    scores=list(array_scores.keys())
    true_labels=list(array_scores.values())

    precision, recall, thresholds=precision_recall_curve(true_labels, scores)
    prc_auc=auc(recall, precision)
    return prc_auc

def plot_PRC(array_scores, prc_auc, m, c, TF, i, k, folder_path):
    """Plotted the Precision-Recall Curve, again using scikit-learn"""
    scores=list(array_scores.keys())
    true_labels=list(array_scores.values())
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (AUC = {prc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PRC Curve for {m}-th order markov model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{folder_path}/{m}-th_order_{i}-{k}-th_fold_MM_PRC_curve_for_chromosome_{c}_TF_{TF}.png", dpi=300)
    plt.close()

os.makedirs("outputs", exist_ok=True)

if __name__ == "__main__":
    start_time = time.time() # recording the start of the code block we are interested in calculating time complexity for
    # tracemalloc.start() # the code block for which we are also interested in calculating memory complexity for
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    c = sys.argv[3] if len(sys.argv) > 3 else 4
    TF = sys.argv[4] if len(sys.argv) > 4 else "CTCF" 
    pseudocount_val = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    run_directory = f"outputs/chr{c}_order{m}_fold{k}_TF{TF}/"
    os.makedirs(run_directory, exist_ok=True)

    print("\n")
    print(f"Running model.py for hyperparams: Model Order = {m}, Number of folds = {k}, Chromosomes = {c}, Transcription Factor = {TF}")
    print("\n")
    print("NOTE: all outputs, namely the models generated for each fold, as well as the score-arrays produced")
    print(f"will be saved to ./outputs/chr{c}_order{m}_fold{k}_TF{TF}/")
    print("\n")
    
    pos_dataset, neg_dataset = import_dataset(TF_name=TF, chromosome_number=c)
    pos_kf = KFold(n_splits=k, shuffle=True, random_state=pipeline_seed)
    neg_kf = KFold(n_splits=k, shuffle=True, random_state=pipeline_seed)

    pfolds = [test_idx for _, test_idx in pos_kf.split(pos_dataset)]
    nfolds = [test_idx for _, test_idx in neg_kf.split(neg_dataset)]
    roc_aucs, prc_aucs = [], [] # storing aucs

    for fold_index in range(k): # looping over folds
        print(f"Processing fold {fold_index+1} of {k} for order {m}")
        
        postestindex = pfolds[fold_index] # test set
        negtestindex = nfolds[fold_index]
        postrainidx = np.concatenate([pfolds[i] for i in range(k) if i != fold_index]) # train set
        negtrainidx = np.concatenate([nfolds[i] for i in range(k) if i != fold_index])

        pos_train, pos_test = [pos_dataset[i] for i in postrainidx], [pos_dataset[i] for i in postestindex]
        neg_train, neg_test = [neg_dataset[i] for i in negtrainidx], [neg_dataset[i] for i in negtestindex]

        pos_model_filepath = run_directory + f"{fold_index}th_of_{k}_fold_pos_model.tsv" # train models
        neg_model_filepath = run_directory + f"{fold_index}th_of_{k}_fold_neg_model.tsv"
        pos_model = nth_order_markov_matrix(m=m, seqs=pos_train, filepath=pos_model_filepath, pseudcnt=pseudocount_val)
        neg_model = nth_order_markov_matrix(m=m, seqs=neg_train, filepath=neg_model_filepath, pseudcnt=pseudocount_val)

        array_of_all_scores=return_array_of_scores(m, pos_test, neg_test, pos_model, neg_model, run_directory)

        AUC_ROC_val = calculate_AUC_ROC(array_of_all_scores) # Add AUC-ROCs and AUC-PRCs to the lists
        AUC_PRC_val = calculate_AUC_PRC(array_of_all_scores)
        roc_aucs.append(AUC_ROC_val)
        prc_aucs.append(AUC_PRC_val)

        print(f"Fold {fold_index+1} of {k} processing done, for order {m}")

    AVG_AUC_ROC = sum(roc_aucs)/len(roc_aucs)
    AVG_AUC_PRC = sum(prc_aucs)/len(prc_aucs)

    end_time = time.time() # end of code block for which we want to calculate time complexity for
    time_complexity = end_time-start_time
    
    # memory_complexity = int(tracemalloc.get_traced_memory()[1]) # stopping calculation for memory complexityy
    # tracemalloc.stop()

    if os.path.exists("resources/run_stats.tsv"):
        stats_df = pd.read_csv("resources/run_stats.tsv", sep="\t")
        new_row = pd.DataFrame({'Chromosome': [c], 'K': [k], 'm': [m], 'TF': [TF], 
                                'AVG_AUC_PRC': [AVG_AUC_PRC], 'AVG_AUC_ROC': [AVG_AUC_ROC], 
                                'time complexity (in s)': [time_complexity]})
        mask = (stats_df['Chromosome'] == c) & (stats_df['K'] == k) & (stats_df['m'] == m) & (stats_df['TF'] == TF)
        if mask.any():
            stats_df = stats_df[~mask]
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
    else:
        stats_df = pd.DataFrame({'Chromosome': [c], 'K': [k], 'm': [m], 'TF': [TF], 
                                 'AVG_AUC_PRC': [AVG_AUC_PRC], 'AVG_AUC_ROC': [AVG_AUC_ROC], 
                                 'time complexity (in s)': [time_complexity]})
    os.makedirs("resources", exist_ok=True)
    stats_df = stats_df.sort_values(by=["TF", "m"])
    stats_df.to_csv("resources/run_stats.tsv", sep="\t", index=False)
    print(f"Run done and all outputs have been saved to ./outputs/chr{c}_order{m}_fold{k}_TF{TF}/")
    print("\n")
    print("\n")