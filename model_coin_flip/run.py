import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from model_coin_flip.coin_flip_model import CoinFlipModel

# ==========================================
# 0. Setup Directory and Parameters
# ==========================================
# Grab the current time so we don't accidentally overwrite our old results.
# This gives us a unique folder name for every run.
run_id = int(time.time())
results_dir = f"model_coin_flip/results_{run_id}"

# exist_ok=True is a lifesaver here; it prevents the script from crashing 
# if the directory somehow already exists.
os.makedirs(results_dir, exist_ok=True)

# The three transcription factors we care about for this run
tfs = ['CTCF', 'REST', 'EP300']

# ==========================================
# 1. Load the Data & Run the Model
# ==========================================
print("Loading train and test datasets...")
    
# Since we ditched cross-validation, we just load our explicit train and test sets directly
train_df = pd.read_csv("data/train_dataset.csv")
test_df = pd.read_csv("data/val_dataset.csv")

# Fire up the coin flip model and train it (calculate those p_i probabilities)
model = CoinFlipModel(train_df)
model.fit()

# Now make predictions on the unseen test data
predictions_df = model.infer(test_df)

# ==========================================
# 2. Evaluate and Plot 
# ==========================================
# Set up the blank canvases for our two plots. 8x6 is usually a solid standard size.
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_prc, ax_prc = plt.subplots(figsize=(8, 6))

# Loop through each transcription factor to calculate metrics and add its line to our plots
for tf in tfs:
    # Convert our 'U's to 0s (Unbound) and anything else to 1s (Bound) 
    # scikit-learn metrics strictly require numerical data, not strings!
    y_true = (test_df[tf] != 'U').astype(int)
    y_pred = predictions_df[tf]
    
    # Quick sanity check: make sure we actually have both 0s and 1s in the test set. 
    # If a TF happens to be 100% Unbound in the test split, the ROC/PRC math will throw an error.
    if len(np.unique(y_true)) > 1:
        # Calculate the math for the ROC curve and its Area Under the Curve (AUC)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Calculate the math for the Precision-Recall Curve and its AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = average_precision_score(y_true, y_pred)
        
        # Draw the actual lines on our canvases
        ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUC = {roc_auc:.3f})')
        ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUC = {prc_auc:.3f})')
        
    else:
        # Just in case the test set was too small or completely skewed
        print(f"Warning: Data for {tf} only contains one class. Skipping metrics.")
    
# --- Prettify and Save the ROC Plot ---
# Always add this diagonal dashed line for ROC—it visually represents random guessing
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05]) # Give the top of the graph a tiny bit of breathing room
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc="lower right")

# Save and immediately close the figure to free up system memory
fig_roc.savefig(os.path.join(results_dir, "ROC.png"), dpi=300)
plt.close(fig_roc)

# --- Prettify and Save the PRC Plot ---
ax_prc.set_xlim([0.0, 1.0])
ax_prc.set_ylim([0.0, 1.05])
ax_prc.set_xlabel('Recall')
ax_prc.set_ylabel('Precision')
ax_prc.set_title('Precision-Recall Curve')
ax_prc.legend(loc="lower left")

fig_prc.savefig(os.path.join(results_dir, "PRC.png"), dpi=300)
plt.close(fig_prc)

print(f"\nPipeline complete! All plots saved to: {results_dir}/")

# Tidy up Python's compiled cache folder so our directory stays clean
if os.path.exists("./model_coin_flip/__pycache__"):
    shutil.rmtree("./model_coin_flip/__pycache__")