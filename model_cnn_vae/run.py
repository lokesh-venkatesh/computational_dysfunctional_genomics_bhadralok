import os, time, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from cnn_vae_model import CNNVAEModel

results_dir = "model_cnn_vae/results_latest"
checkpoints_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

# --- HYPERPARAMETERS ---
epochs = 1
batch_size = 256          
learning_rate = 5e-4      
weight_decay = 1e-4       
dropout_rate = 0.4        
latent_dim = 100          
motif_embed_dim = 32      
beta = 0.1             
lambda_cls = 500.0     
pwm_dir = "./pwms/"       

if __name__ == '__main__':
    train_df = pd.read_csv("data/train_dataset.csv")
    test_df = pd.read_csv("data/val_dataset.csv")

    model = CNNVAEModel(train_df, test_df, epochs, batch_size, learning_rate, weight_decay, dropout_rate, latent_dim, motif_embed_dim, beta, lambda_cls, pwm_dir)
    
    resume_path = os.path.join(checkpoints_dir, "last.pth")
    model.fit(checkpoints_dir, resume_from=resume_path if os.path.exists(resume_path) else None)

    preds = model.infer(test_df)
    np.save(os.path.join(results_dir, "latent_vectors_val.npy"), model.extract_latent_vectors(test_df))

    # --- PLOTTING ROC AND PRC SIDE BY SIDE ---
    print("\nGenerating ROC and PRC plots...")
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Sequence-CNN VAE Performance', fontsize=16, fontweight='bold')

    for tf in ['CTCF', 'REST', 'EP300']:
        y_true, y_pred = (test_df[tf] != 'U').astype(int), preds[tf]
        
        # ROC calculations
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # PRC calculations
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = average_precision_score(y_true, y_pred)
        
        # Plotting
        ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUROC = {roc_auc:.3f})')
        ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUPRC = {prc_auc:.3f})')
        
    # ROC Formatting
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
    ax_roc.legend(loc="lower right")

    # PRC Formatting
    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall', fontsize=12)
    ax_prc.set_ylabel('Precision', fontsize=12)
    ax_prc.set_title('Precision-Recall Curve (PRC)', fontsize=14)
    ax_prc.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "Metrics_CNN_VAE.png"), dpi=300)
    plt.close(fig)
    print(f"Done! Saved plots and vectors to {results_dir}")