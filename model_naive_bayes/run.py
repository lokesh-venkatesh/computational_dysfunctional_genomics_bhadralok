import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from naive_bayes_model import NaiveBayesModel

run_id = int(time.time())
results_dir = f"model_naive_bayes/results_{run_id}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs("pwms", exist_ok=True) 

tfs = ['CTCF', 'REST', 'EP300']

# ==========================================
# GUARD BLOCK FOR MULTIPROCESSING
# ==========================================
if __name__ == '__main__':
    train_df = pd.read_csv("data/train_dataset.csv")
    test_df = pd.read_csv("data/val_dataset.csv")

    model = NaiveBayesModel(train_df, pwm_dir="./pwms/")
    model.fit()
    predictions_df = model.infer(test_df)

    # =================== PLOT ===================
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_prc, ax_prc = plt.subplots(figsize=(8, 6))

    for tf in tfs:
        y_true = (test_df[tf] != 'U').astype(int)
        y_pred = predictions_df[tf]
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            prc_auc = average_precision_score(y_true, y_pred)
            
            ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUC = {roc_auc:.3f})')
            ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUC = {prc_auc:.3f})')
        else:
            print(f"Warning: Data for {tf} only contains one class. Skipping metrics.")
        
    
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve (Mixed Naive Bayes)')
    ax_roc.legend(loc="lower right")
    fig_roc.savefig(os.path.join(results_dir, "ROC.png"), dpi=300)
    plt.close(fig_roc)

    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve (Mixed Naive Bayes)')
    ax_prc.legend(loc="lower left")
    fig_prc.savefig(os.path.join(results_dir, "PRC.png"), dpi=300)
    plt.close(fig_prc)

    print(f"\n Pipeline complete, all results saved to: {results_dir}/")

    if os.path.exists("./model_naive_bayes/__pycache__"):
        shutil.rmtree("./model_naive_bayes/__pycache__")