import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from coin_flip_model import CoinFlipModel

# ========================================== FITTING & INFERERENCE ==========================================

run_id = int(time.time())
results_dir = f"model_coin_flip/results_{run_id}"
os.makedirs(results_dir, exist_ok=True) # create a run folder to save all run results
tfs = ['CTCF', 'REST', 'EP300']

print("Loading train and test datasets...")
train_df = pd.read_csv("data/project_dataset.csv") # train_dataset
val_df = pd.read_csv("data/val_dataset.csv")

model = CoinFlipModel(train_df) # load the coin model with the train_df (but no param fit yet)

print("\n Training the model using the Test Dataset")
model.fit() # trains the coin model using the train_df (now identifies the params)

print("\n Making predictions on the Validation Dataset")
predictions_df = model.infer(val_df) # making model predictions


# ========================================== PLOTTING METRICS ==========================================
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_prc, ax_prc = plt.subplots(figsize=(8, 6))

for tf in tfs:
    # Convert our 'U's to 0s (Unbound) and anything else to 1s (Bound) 
    y_true = (val_df[tf] != 'U').astype(int)
    y_pred = predictions_df[tf]
    
    # Quick sanity check: make sure we actually have both 0s and 1s in the test set. 
    # If a TF happens to be 100% Unbound in the test split, the ROC/PRC math will throw an error.
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prc_auc = average_precision_score(y_true, y_pred)
    
    ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUC = {roc_auc:.5f})') # PLOTTING ROC AND PRC
    ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUC = {prc_auc:.5f})')
        
ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # diagonal line which represents random guessing
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05]) # Give the top of the graph a tiny bit of breathing room
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc="lower right")
fig_roc.savefig(os.path.join(results_dir, "ROC.png"), dpi=300)
plt.close(fig_roc)

ax_prc.set_xlim([0.0, 1.0])
ax_prc.set_ylim([0.0, 1.05])
ax_prc.set_xlabel('Recall')
ax_prc.set_ylabel('Precision')
ax_prc.set_title('Precision-Recall Curve')
ax_prc.legend(loc="lower left")
fig_prc.savefig(os.path.join(results_dir, "PRC.png"), dpi=300)
plt.close(fig_prc)

print(f"\n Pipeline complete, all results saved to: {results_dir}/")

# Delete python's compiled cache folder
if os.path.exists("./model_coin_flip/__pycache__"):
    shutil.rmtree("./model_coin_flip/__pycache__")