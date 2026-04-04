import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from vae_model import CVAEModel

# ==========================================
# 0. Setup Directory and Parameters
# ==========================================
run_id = int(time.time())
results_dir = f"model_vae/results_latest" 
checkpoints_dir = os.path.join(results_dir, "checkpoints")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# ==========================================
# GLOBAL HYPERPARAMETER CONTROL
# ==========================================
epochs = 5
batch_size = 128      
learning_rate = 1e-4  
weight_decay = 1e-3       # Controls L2 Regularization to prevent overfitting
dropout_rate = 0.5        # Controls neuron dropout to prevent memorization

latent_dim = 100          # Size of the K-mer VAE bottleneck
motif_embed_dim = 32      # Size of the Cheat Code embedding stream
beta = 0.1             
lambda_cls = 1000.0     
kmer_size = 5         
pwm_dir = "./pwms/"       # Point to your JASPAR matrices

tfs = ['CTCF', 'REST', 'EP300']

if __name__ == '__main__':
    print("Loading train and validation datasets...")
    train_df = pd.read_csv("data/train_dataset.csv")
    test_df = pd.read_csv("data/val_dataset.csv")

    # Pass all global variables to the model
    model = CVAEModel(
        train_data=train_df, 
        val_data=test_df, 
        epochs=epochs, 
        batch_size=batch_size, 
        lr=learning_rate,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        latent_dim=latent_dim,
        motif_embed_dim=motif_embed_dim,
        beta=beta,
        lambda_cls=lambda_cls,
        kmer_size=kmer_size,
        pwm_dir=pwm_dir
    )

    last_checkpoint = os.path.join(checkpoints_dir, "last_bindvae_checkpoint.pth")
    resume_path = last_checkpoint if os.path.exists(last_checkpoint) else None

    model.fit(checkpoint_dir=checkpoints_dir, resume_from=resume_path)

    # ==========================================
    # 3. Inference & Latent Space Extraction
    # ==========================================
    predictions_df = model.infer(test_df)

    print("Extracting latent space vectors for downstream analysis...")
    latent_matrix = model.extract_latent_vectors(test_df)
    np.save(os.path.join(results_dir, "latent_vectors_val.npy"), latent_matrix)

    # ==========================================
    # 4. Evaluate and Plot 
    # ==========================================
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
        ax_roc.set_title(f'ROC Curve (BindVAE-Lite)')
        ax_roc.legend(loc="lower right")
        fig_roc.savefig(os.path.join(results_dir, "ROC.png"), dpi=300)
        plt.close(fig_roc)

        ax_prc.set_xlim([0.0, 1.0])
        ax_prc.set_ylim([0.0, 1.05])
        ax_prc.set_xlabel('Recall')
        ax_prc.set_ylabel('Precision')
        ax_prc.set_title(f'Precision-Recall Curve (BindVAE-Lite)')
        ax_prc.legend(loc="lower left")
        fig_prc.savefig(os.path.join(results_dir, "PRC.png"), dpi=300)
        plt.close(fig_prc)

    print(f"\nPipeline complete! All plots, models, and latent vectors saved to: {results_dir}/")

    if os.path.exists("./model_vae/__pycache__"):
        shutil.rmtree("./model_vae/__pycache__")