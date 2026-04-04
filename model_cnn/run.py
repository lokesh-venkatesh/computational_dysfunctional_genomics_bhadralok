import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from cnn_model import DeepCNNModel

# ==========================================
# 0. Setup Directory and Parameters
# ==========================================
run_id = int(time.time())
results_dir = f"model_cnn/results_latest" 
checkpoints_dir = os.path.join(results_dir, "checkpoints")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# ==========================================
# GLOBAL HYPERPARAMETER CONTROL
# ==========================================
epochs = 20
batch_size = 256          # CNNs are fast, we can use larger batches
learning_rate = 1e-3
pwm_dir = "./pwms/"       

tfs = ['CTCF', 'REST', 'EP300']

if __name__ == '__main__':
    print("Loading datasets...")
    train_df = pd.read_csv("data/train_dataset.csv")
    val_df = pd.read_csv("data/val_dataset.csv")

    # Initialize CNN Model
    model = DeepCNNModel(
        pwm_dir=pwm_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )

    # 1. Train the Dual-Stream CNN
    model.fit(train_df, val_df, save_dir=checkpoints_dir)

    # 2. Run Inference on Validation Set
    print("\nRunning Inference on Validation Data...")
    predictions_df = model.infer(val_df)

    # 3. Evaluate and Plot 
    print("\nGenerating Metrics and Plots...")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_prc, ax_prc = plt.subplots(figsize=(8, 6))

    for tf in tfs:
        y_true = (val_df[tf] != 'U').astype(int)
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
    ax_roc.set_title('ROC Curve (Dual-Stream DeepCNN)')
    ax_roc.legend(loc="lower right")
    fig_roc.savefig(os.path.join(results_dir, "ROC_CNN.png"), dpi=300)
    plt.close(fig_roc)

    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve (Dual-Stream DeepCNN)')
    ax_prc.legend(loc="lower left")
    fig_prc.savefig(os.path.join(results_dir, "PRC_CNN.png"), dpi=300)
    plt.close(fig_prc)

    print(f"\nPipeline complete! All plots and models saved to: {results_dir}/")