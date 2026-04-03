import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Import our new Naive Bayes class instead of Coin Flip
from model_naive_bayes.naive_bayes_model import NaiveBayesModel

# ==========================================
# 0. Setup Directory and Parameters
# ==========================================
# Grab the current time so we don't accidentally overwrite our old results.
run_id = int(time.time())
results_dir = f"model_naive_bayes/results_{run_id}"

os.makedirs(results_dir, exist_ok=True)

# The three transcription factors we care about for this run
tfs = ['CTCF', 'REST', 'EP300']
kmer_size = 6 # Hyperparameter: length of the sequence "words"

# ==========================================
# 1. Load the Data & Run the Model
# ==========================================
print("Loading train and test datasets...")
    
train_df = pd.read_csv("data/train_dataset.csv")
test_df = pd.read_csv("data/val_dataset.csv")

# Initialize the Naive Bayes model with our chosen k-mer size
model = NaiveBayesModel(train_df, kmer_size=kmer_size, pseudocount=1.0)
model.fit()

# Now make predictions on the unseen test data
# The output will be continuous log-odds scores
predictions_df = model.infer(test_df)

# ==========================================
# 2. Evaluate and Plot 
# ==========================================
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_prc, ax_prc = plt.subplots(figsize=(8, 6))

for tf in tfs:
    # Convert 'U's to 0s (Unbound) and anything else to 1s (Bound) 
    y_true = (test_df[tf] != 'U').astype(int)
    y_pred = predictions_df[tf]
    
    if len(np.unique(y_true)) > 1:
        # Calculate AUC-ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC-PRC
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = average_precision_score(y_true, y_pred)
        
        # Plot lines
        ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUC = {roc_auc:.3f})')
        ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUC = {prc_auc:.3f})')
        
    else:
        print(f"Warning: Data for {tf} only contains one class. Skipping metrics.")
    
# --- Prettify and Save the ROC Plot ---
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title(f'ROC Curve (Naive Bayes, k={kmer_size})')
ax_roc.legend(loc="lower right")

fig_roc.savefig(os.path.join(results_dir, "ROC.png"), dpi=300)
plt.close(fig_roc)

# --- Prettify and Save the PRC Plot ---
ax_prc.set_xlim([0.0, 1.0])
ax_prc.set_ylim([0.0, 1.05])
ax_prc.set_xlabel('Recall')
ax_prc.set_ylabel('Precision')
ax_prc.set_title(f'Precision-Recall Curve (Naive Bayes, k={kmer_size})')
ax_prc.legend(loc="lower left")

fig_prc.savefig(os.path.join(results_dir, "PRC.png"), dpi=300)
plt.close(fig_prc)

print(f"\nPipeline complete! All plots saved to: {results_dir}/")

# Clean up pycache
if os.path.exists("./model_naive_bayes/__pycache__"):
    shutil.rmtree("./model_naive_bayes/__pycache__")