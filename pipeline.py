# pipeline assuming we have an input 'n' passed as an argument in the CLI when running this file

def import_pos_dataset(filepath='pos_dataset.fasta'):
    """
    INPUT: none (unless filepath changes)
    OUTPUT: a list of strings (DNA sequences) all of equal lengths
            corresponding to the positive sequences
    """

def import_neg_dataset(filepath='neg_dataset.fasta'):
    """
    INPUT: none (unless filepath changes)
    OUTPUT: a list of strings (DNA sequences) all of equal lengths, 
            corresponding to the negative sequences
    """
def nth_order_markov_matrix(n, seqs):
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

def gen_pos_model(n, pos_seqs):
    return nth_order_markov_matrix(n, pos_seqs)

def gen_neg_model(n, neg_seqs):
    return nth_order_markov_matrix(n, neg_seqs)

def return_dict_of_scores(pos_seqs, neg_seqs, pos_model, neg_model):
    """
    INPUT: positive and negative sequences as well as the two corresponding models
    OUTPUT: a dictionary where the keys are the scores and the valeus are the labels, positive/negative
    """

    def calculate_score(seq, pos_model, neg_model):
        """
        INPUT: a sequence (the label is monitored for outside this function),
        and a positive and negative model/matrix
        OUTPUT: score = log_2(pos_model(seq)/neg_model(seq))
        """

def plot_distributions(n, dict_scores):
    """
    INPUT: dictionary of scores and the order of the markov model used
    OUTPUT: a plot generated of the distributions of scores (and coloured accordingly), 
    and locally saved with some appropriate name
    """

def plot_ROC():
    """
    We will either hard code this from scratch, or use some standard library like scikit-learn.
    Make sure we save this plot locally as well
    """    
        
def calculate_AUC_ROC():
    """
    Again, we can scam this by using scikit-learn functions.
    
    NOTE: ensure that the AUC_ROC is printed out as the final and ONLY output of this entire file.
    This will be useful when we run the mainpipeline across different values of 'n'.
    """

if __name__ == "__main__":
    n = ... #parameter passed as argument when running the script
    """ORDER CHOSEN FOR THE MARKOV MODEL"""
    pos_dataset = import_pos_dataset()
    neg_dataset = import_neg_dataset()

    pos_model = gen_pos_model(n, pos_dataset)
    neg_model = gen_neg_model(n, neg_dataset)

    dict_of_scores = return_dict_of_scores(pos_dataset, neg_dataset, pos_model, neg_model)

    plot_distributions(n, dict_scores=dict_of_scores)
    plot_ROC()

    AUC_ROC_val = calculate_AUC_ROC()
    print(AUC_ROC_val)