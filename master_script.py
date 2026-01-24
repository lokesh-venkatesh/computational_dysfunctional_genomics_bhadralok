pos_seqs = ...
neg_seqs = ...

# assert that all sequences are of equal lengths

L = len(pos_seqs[0])

AUC_scores = {}

for i in range(0, L):
    ... #run pipeline.py
    AUC_scores[i] = ...

    # plot the distribution of AUC_ROC scores for different orders of markov models