""" # simplerVersion.py

Has three main functions:
1. import_fasta_file: which for a given fasta file, extracts the sequences into a list
2. nth_order_markov_matrix: infers the resulting 'n'-th order markov model from this dataset
3. return_array_or_scores: returns a list of log-likelihoods that the model infers for each sequence in the list

NOTE: The function used for calculating this 'n'th order markov model classifier's matrix
is the same as used in model.py, which is the main script of this whole repository."""

import math
import sys
from itertools import product

def import_fasta_file(fasta_filepath="data/example.fa"):
    """ INPUT: the fasta filepath (.fa) corresponding to an input dataset
        OUTPUT: a list of strings (sequences)"""
    dataset, current_sequence = [], []

    with open(fasta_filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line: continue  # Skip empty lines
            if line.startswith('>'):
                if current_sequence:
                    dataset.append(''.join(current_sequence))
                    current_sequence = []
            else: current_sequence.append(line)
        if current_sequence: # Add the last sequence if present
            dataset.append(''.join(current_sequence))
    return dataset

def nth_order_markov_matrix(m, seqs, pseudcnt=0.1):
    """ INPUTS: the order of the markov model 'm' and the list of sequences (the dataset)
        OUTPUT: a matrix corresponding to the m-th order transition probabilities between different states,
         this should also include zeroth order transition probablities, saved as '0' in the matrix"""
    
    nucls = ['A', 'G', 'T']  # initalising four possible nucleotides
    # 'C', 

    # we first calculate the vector of zeroth order transition probabilities
    total_counts = {'A': pseudcnt, 'G': pseudcnt, 'T': pseudcnt}
    # 'C': 0, 
    for seq in seqs: # there is already one loop through all of the dataset right here
        for nucl in nucls:
            total_counts[nucl] += seq.count(nucl)
    total = sum(total_counts[nucl] for nucl in nucls)
    zeroth_order_probabilities = {n: total_counts[n] / total for n in nucls}

    # This is for the zero-th order case
    if m == 0:
        mm_model = {0: zeroth_order_probabilities}
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
    return nth_order_mm

def return_array_of_scores(m, seqs, model):
    """ INPUT: a list of sequences as well as some custom model, not necessarily trained on this dataset
        OUTPUT: a list of the log-likelihood scores of each sequence as inferred from the input model"""

    def calculate_score(n, seq, model):
        """ INPUT: the markov model classifier's order 'n', the sequence itself, and the model
            OUTPUT: score = log_2(model(seq)) 
            
            NOTE: the assumption of a base-2 logarithm is by (arbitrary) choice"""
        score=0 
        if n==0:
            for i in range(0, len(seq)):
                score += math.log2(model[0][seq[i]])
                print(f"Score incremented for {seq[i]} is given by {math.log2(model[0][seq[i]])}")
        elif n>=1:
            for i in range(0, n):
                score += math.log2(model[0][seq[i]])
                print(f"Score incremented for {seq[i]} is given by {math.log2(model[0][seq[i]])}")
            for i in range(n, len(seq)):
                prev_kmer = seq[i-n:i]
                score += math.log2(model[prev_kmer][seq[i]])
                print(f"Score incremented for {seq[i]} is given by {math.log2(model[prev_kmer][seq[i]])}")
        return score

    return [calculate_score(m, seq, model) for seq in seqs]

if __name__ == "__main__":
    input_fasta_filepath = int(sys.argv[1]) if len(sys.argv) > 1 else "data/example.fa"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    dataset = import_fasta_file(fasta_filepath=input_fasta_filepath)
    mth_order_classifier = nth_order_markov_matrix(m=m, seqs=dataset)
    print("Markov Model Classifier is given by")
    print("\n")
    for key, val in mth_order_classifier.items():
        print(key, val)
    print("\n")
    array_of_all_scores = return_array_of_scores(m=m, seqs=dataset, model=mth_order_classifier)
    print("All scores:")
    for score in array_of_all_scores:
        print(score) # log2(model(seq)), as was required