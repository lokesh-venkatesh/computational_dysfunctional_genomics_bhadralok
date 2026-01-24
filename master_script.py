# assert that all sequences are of equal lengths
L = ...
AUC_scores = {}

for i in range(0, L):
    # run pipeline.py
    # NOTE need to figure out how to record the printed output here
    AUC_scores[i] = ...

    # plot the distribution of AUC_ROC scores for different orders of markov models
    # save this distribution and find the 'maxima', either manually or using code, 
    # to identify which is the best 'n'-th order model.