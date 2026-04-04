import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from markov_model import MarkovModel

# ========================================== FITTING & INFERERENCE ==========================================

order = 1 # order of the markov model

for order in range(0,11):
    tfs = ['CTCF', 'REST', 'EP300']

    run_id = int(time.time())
    results_dir = f"model_markov/order{order}/results_{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_csv("data/train_dataset.csv")
    test_df = pd.read_csv("data/val_dataset.csv")

    model = MarkovModel(train_df, order=order, pseudocount=1)
    model.fit()

    # Now make predictions on the unseen test data
    # The output will be continuous log-likelihood ratios, which works perfectly for ROC/PRC
    predictions_df = model.infer(test_df)

    # ========================================== PLOTTING METRICS ==========================================
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_prc, ax_prc = plt.subplots(figsize=(8, 6))

    for tf in tfs:
        y_true = (test_df[tf] != 'U').astype(int)
        y_pred = predictions_df[tf]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = average_precision_score(y_true, y_pred)
        
        ax_roc.plot(fpr, tpr, lw=2, label=f'{tf} (AUC = {roc_auc:.3f})')
        ax_prc.plot(recall, precision, lw=2, label=f'{tf} (AUC = {prc_auc:.3f})')
        
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve (Markov Model, Order {order})')
    ax_roc.legend(loc="lower right")
    fig_roc.savefig(os.path.join(results_dir, "ROC.png"), dpi=300)
    plt.close(fig_roc)

    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title(f'Precision-Recall Curve (Markov Model, Order {order})')
    ax_prc.legend(loc="lower left")
    fig_prc.savefig(os.path.join(results_dir, "PRC.png"), dpi=300)
    plt.close(fig_prc)

    print(f"\n Run complete for order {order}, all results saved to: {results_dir}/")

    if os.path.exists("./model_markov/__pycache__"):
        shutil.rmtree("./model_markov/__pycache__")