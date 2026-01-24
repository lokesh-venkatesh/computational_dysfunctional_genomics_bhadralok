# pipeline assuming we have an input 'n'

def import_pos_dataset():
    ...

def import_neg_dataset():
    ...

n = ... #parameter passed as argument when running the script

def nth_order_markov_matrix(n, seqs):
    ...

def gen_pos_model():
    ...

def gen_neg_model():
    ...

# NOTE: we need to incorporate pseudocounts, and also account for zeroth-order probabilities separately

def return_dict_of_scores(pos_seqs, neg_seqs, pos_model, neg_model):

    def calculate_score(seq, pos_model, neg_model):
        ...

def plot_ROC():
        
    def conf_matrix():
        ...

def calculate_AUC_ROC():
    ...