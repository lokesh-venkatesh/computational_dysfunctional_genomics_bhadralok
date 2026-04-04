import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from svm_model import KmerSVMModel

# ==========================================
# 0. Setup Directory and Parameters
# ==========================================
run_id = int(time.time())
results_dir = f"model_svm/results_latest" 
checkpoints_dir = os.path.join(results_dir, "checkpoints")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# ==========================================
# GLOBAL HYPERPARAMETER CONTROL
# ==========================================
kernel_choice = 'linear'  # Options: 'linear' or 'rbf_approx' # NOTE TRY THIS LATER...
kmer_size = 5         
n_components = 800        # Only used if kernel='rbf_approx' (Higher = closer to true RBF, but slower)
max_iter = 3000           # For LinearSVC convergence

tfs = ['CTCF', 'REST', 'EP300']

if __name__ == '__main__':
    print("Loading train and validation datasets...")
    train_df = pd.read_csv("data/train_dataset.csv")
    test_df = pd.read_csv("data/val_dataset.csv")

    # Initialize SVM Model
    model = KmerSVMModel(
        kernel=kernel_choice,
        kmer_size=kmer_size,
        max_iter=max_iter,
        n_components=n_components,
        pwm_dir="./pwms/"
    )

    # 1. Train the Models
    model.fit(train_df, save_dir=checkpoints_dir)

    # 2. Run Inference on Validation Set
    print("\nRunning Inference on Validation Data...")
    predictions_df = model.infer(test_df)

    # 3. Evaluate and Plot 
    print("\nGenerating Metrics and Plots...")
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
            
            print(f"{tf} -> ROC AUC: {roc_auc:.3f} | PRC AUC: {prc_auc:.3f}")
            
            ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUC = {roc_auc:.3f})')
            ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUC = {prc_auc:.3f})')
        else:
            print(f"Warning: Data for {tf} only contains one class. Skipping metrics.")
        
    # --- Prettify and Save Plots ---
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve (K-mer SVM - {kernel_choice.upper()})')
    ax_roc.legend(loc="lower right")
    fig_roc.savefig(os.path.join(results_dir, "ROC_SVM.png"), dpi=300)
    plt.close(fig_roc)

    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title(f'Precision-Recall Curve (K-mer SVM - {kernel_choice.upper()})')
    ax_prc.legend(loc="lower left")
    fig_prc.savefig(os.path.join(results_dir, "PRC_SVM.png"), dpi=300)
    plt.close(fig_prc)

    print(f"\nPipeline complete! All plots and models saved to: {results_dir}/")