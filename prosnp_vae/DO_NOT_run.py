import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from prosnp_vae_model import ProSNPVAEWrapper

results_dir = "/content/drive/MyDrive/CFG_Final/results_prosnp_vae"
checkpoints_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

epochs = 40                   
batch_size = 1024             
learning_rate = 1e-3          
beta = 0.2
lambda_cls = 500.0

if __name__ == '__main__':
    train_df = pd.read_csv("data/train_dataset.csv")
    val_df = pd.read_csv("data/val_dataset.csv")

    model = ProSNPVAEWrapper(epochs, batch_size, learning_rate, beta, lambda_cls)
    model.fit(train_df, val_df, checkpoints_dir)

    print("\nRunning Inference on Validation Data...")
    preds = model.infer(val_df)

    print("\nGenerating Metrics and Plots...")
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('ProSNP-VAE Performance', fontsize=16, fontweight='bold')

    for tf in ['CTCF', 'REST', 'EP300']:
        y_true, y_pred = (val_df[tf] != 'U').astype(int), preds[tf]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUROC = {auc(fpr, tpr):.3f})')
        ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUPRC = {average_precision_score(y_true, y_pred):.3f})')
        
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlim([0.0, 1.0]), ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate'), ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve'), ax_roc.legend(loc="lower right")

    ax_prc.set_xlim([0.0, 1.0]), ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall'), ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve'), ax_prc.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "Metrics_ProSNP_VAE.png"), dpi=300)
    print(f"Done! Saved plots and checkpoints to your Google Drive ({results_dir}).")