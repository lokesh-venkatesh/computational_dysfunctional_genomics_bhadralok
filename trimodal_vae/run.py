import os, time, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from trimodal_model import TriModalModel

results_dir = "trimodal_vae/results_latest"
checkpoints_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

# ==========================================
# GLOBAL HYPERPARAMETERS
# ==========================================
USE_CONTEXTUAL_KMERS = True   

epochs = 15                   # INCREASED: VAE needs 10-15 epochs to organize latent space
batch_size = 1024             # INCREASED: 4x larger for faster epoch processing
learning_rate = 4e-4          # SCALED: 4x larger to match the batch size rule
beta = 0.2                    # TARGET VAE Regularization (will anneal from 0 to 0.2)
lambda_cls = 500.0            

if __name__ == '__main__':
    train_df = pd.read_csv("data/train_dataset.csv")
    val_df = pd.read_csv("data/val_dataset.csv")

    if not os.path.exists("./data/processed/train_base_cheats.npy"):
        print("ERROR: Run preprocess_features.py first to generate the .npy arrays!")
        exit()

    model = TriModalModel(epochs, batch_size, learning_rate, beta, lambda_cls, USE_CONTEXTUAL_KMERS)
    model.fit(train_df, val_df, checkpoints_dir)

    print("\nRunning Inference on Validation Data...")
    preds = model.infer(val_df, "val")

    print("\nGenerating Metrics and Plots...")
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Tri-Modal VAE Performance', fontsize=16, fontweight='bold')

    for tf in ['CTCF', 'REST', 'EP300']:
        y_true, y_pred = (val_df[tf] != 'U').astype(int), preds[tf]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = average_precision_score(y_true, y_pred)
        
        ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUROC = {roc_auc:.3f})')
        ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUPRC = {prc_auc:.3f})')
        
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlim([0.0, 1.0]), ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate'), ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve'), ax_roc.legend(loc="lower right")

    ax_prc.set_xlim([0.0, 1.0]), ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall'), ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve'), ax_prc.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "Metrics_TriModal.png"), dpi=300)
    print(f"Done! Saved to {results_dir}")