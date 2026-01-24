Scripts/functions we need to write for the pre-midsem stuff:

1. Generate dummy sequences to try out our pipeline on

2. Generate a n-th order markov model matrix given an input order "n" as well as a set of input sequences (of equal lengths, hopefully)... 

...Note: separately calculate P(X), since the first nucleotide of each sequence will require you to know what these values are, and these may not be sufficiently assumed as uniformly distributed.

ALSO TO NOTE: incorporate pseudocounts into the model for larger 'n' (or start with pseudocounts right at the beginning)

3. Calculate the 'score' given two matrix models as well as an input sequence, and do this for all sequences in both sets

4. Using this dict of scores (as well as the labels of the points, whether they belong to true or false), then plot the ROC and calculate the AUC-ROC. Make sure the plot is saved locally.

5. Write a single script that does all of these scripts in sequence for a given input 'n' as well as two inputs of positive and negative sequence sets.

FINALLY: write a master script that does this for all 'n' from i=0 to length L of the sequences given, and for each AUC-ROC value obtained, plot a distribution fo AUC-ROC versus 'n', and find the maximum (or maxima; plural).