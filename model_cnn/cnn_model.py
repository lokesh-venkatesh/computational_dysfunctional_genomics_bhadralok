import os
import re
import glob
import math
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ==========================================
# 1. One-Hot DNA PyTorch Dataset
# ==========================================
class DNASequenceDataset(Dataset):
    def __init__(self, sequences, cheat_features, tf_labels):
        self.sequences = sequences
        self.X_cheat = torch.tensor(cheat_features, dtype=torch.float32)
        self.tfs = torch.tensor(tf_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].upper()
        # Create a (4, Length) One-Hot Matrix
        one_hot = np.zeros((4, len(seq)), dtype=np.float32)
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i, nuc in enumerate(seq):
            if nuc in nuc_map:
                one_hot[nuc_map[nuc], i] = 1.0
                
        X_seq = torch.tensor(one_hot, dtype=torch.float32)
        return X_seq, self.X_cheat[idx], self.tfs[idx]

# ==========================================
# 2. Deeper Dual-Stream 1D-CNN (DeepSEA-inspired)
# ==========================================
class DualStreamCNN(nn.Module):
    def __init__(self, num_cheat_features, num_classes=3, dropout_rate=0.4):
        super(DualStreamCNN, self).__init__()
        
        # STREAM A: Deep Motif Scanner
        self.conv_stream = nn.Sequential(
            # Layer 1: Detects primary, low-level motifs (e.g., 5-mers to 15-mers)
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=15, padding=7),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout1d(dropout_rate / 2.0), # Spatial Dropout
            
            # Layer 2: Combines low-level motifs into larger structural features
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=7, padding=3),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout1d(dropout_rate / 2.0),
            
            # Layer 3: Captures long-range dependencies and flanking regions
            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=5, padding=2),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool1d(1), # Smashes length dimension to 1
            nn.Flatten()             # Output shape: (Batch, 960)
        )
        
        # STREAM B: Biological Context
        self.cheat_stream = nn.Sequential(
            nn.Linear(num_cheat_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2.0)
        )
        
        # FUSION HEAD
        self.classifier = nn.Sequential(
            nn.Linear(960 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_seq, x_cheat):
        seq_features = self.conv_stream(x_seq)
        cheat_features = self.cheat_stream(x_cheat)
        combined = torch.cat((seq_features, cheat_features), dim=1)
        logits = self.classifier(combined)
        return logits

# ==========================================
# 3. Training & Inference Wrapper
# ==========================================
class DeepCNNModel:
    def __init__(self, train_data, val_data=None, epochs=30, batch_size=128, lr=1e-3, 
                 weight_decay=1e-4, dropout_rate=0.4, pwm_dir="./pwms/"):
                 
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.pwm_dir = pwm_dir
        self.tfs = ['CTCF', 'REST', 'EP300']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initializing Dual-Stream CNN on: {self.device}")
        
        self.pwms = self._load_pwms()
        self.scaler = StandardScaler()

    def _load_pwms(self):
        pwms = {}
        files = glob.glob(os.path.join(self.pwm_dir, "*.pfm"))
        for file in files:
            tf_name = os.path.basename(file).split('.')[0].upper()
            matrix = []
            with open(file, 'r') as f:
                for line in f:
                    if line.startswith(">"): continue
                    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if numbers: matrix.append([float(x) for x in numbers])
            if len(matrix) == 4:
                pfm = np.array(matrix).T 
                pseudocount = 0.1
                col_sums = pfm.sum(axis=1, keepdims=True)
                pwms[tf_name] = np.log2((pfm + pseudocount) / (col_sums + 4 * pseudocount) / 0.25)
        return pwms

    @staticmethod
    def _extract_worker(sequences, pwms, atac_labels):
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        chunk_features = []
        for seq, atac in zip(sequences, atac_labels):
            seq = seq.upper()
            length = len(seq)
            counts = Counter(seq)
            
            gc_content = (counts.get('G', 0) + counts.get('C', 0)) / length
            cpg_oe = (seq.count('CG') * length) / max(counts.get('C', 0) * counts.get('G', 0), 1)
            features = [gc_content, cpg_oe, atac]
            
            if pwms:
                one_hot = np.zeros((length, 4))
                for i, nuc in enumerate(seq):
                    if nuc in nuc_map: one_hot[i, nuc_map[nuc]] = 1
                    
                for name, pwm in pwms.items():
                    w_len = pwm.shape[0]
                    windows = np.lib.stride_tricks.sliding_window_view(one_hot, window_shape=(w_len, 4)).squeeze()
                    scores = np.einsum('nwc,wc->n', windows, pwm)
                    features.extend([np.max(scores), np.sum(scores[scores > 0]) if len(scores[scores > 0]) > 0 else 0, len(scores[scores > 0])])
                    
            chunk_features.append(features)
        return np.array(chunk_features, dtype=np.float32)

    def extract_cheat_codes(self, df, is_train=True):
        print("Extracting biological cheat codes (Parallel)...")
        sequences = df['sequence'].tolist()
        atac_labels = (df['ATAC'] != 'U').astype(int).tolist()
        
        n_jobs = -1 
        chunk_size = max(1, math.ceil(len(sequences) / (os.cpu_count() or 1)))
        
        seq_chunks = [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]
        atac_chunks = [atac_labels[i:i + chunk_size] for i in range(0, len(atac_labels), chunk_size)]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_worker)(s_chunk, self.pwms, a_chunk) 
            for s_chunk, a_chunk in zip(seq_chunks, atac_chunks)
        )
        
        X_cheat_raw = np.vstack(results)
        
        if is_train:
            X_cheat_scaled = self.scaler.fit_transform(X_cheat_raw)
        else:
            X_cheat_scaled = self.scaler.transform(X_cheat_raw)
            
        return X_cheat_scaled

    def _prepare_data(self, df, is_train=True):
        X_cheat = self.extract_cheat_codes(df, is_train=is_train)
        y = (df[self.tfs] != 'U').astype(int).values
        return DNASequenceDataset(df['sequence'].tolist(), X_cheat, y)

    def fit(self, checkpoint_dir="checkpoints", resume_from=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, "best_cnn_model.pth")
        last_checkpoint_path = os.path.join(checkpoint_dir, "last_cnn_checkpoint.pth")
        
        train_dataset = self._prepare_data(self.train_data, is_train=True)
        val_dataset = self._prepare_data(self.val_data, is_train=False) if self.val_data is not None else None
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None

        num_cheat_features = train_dataset.X_cheat.shape[1]
        self.model = DualStreamCNN(
            num_cheat_features=num_cheat_features, 
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Mimic the VAE's handling of class imbalance
        pos_weight = torch.tensor([5.0, 5.0, 5.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        start_epoch = 1
        best_val_loss = float('inf')

        # Resume logic
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"\nStarting CNN Training (Epoch {start_epoch} to {self.epochs})...")
        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs} [Train]")
            for x_seq, x_cheat, y in loop:
                x_seq, x_cheat, y = x_seq.to(self.device), x_cheat.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_seq, x_cheat)
                loss = criterion(logits, y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(Loss=loss.item())
                
            avg_train_loss = train_loss / len(train_loader)

            val_loss_str = ""
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for x_seq, x_cheat, y in val_loader:
                        x_seq, x_cheat, y = x_seq.to(self.device), x_cheat.to(self.device), y.to(self.device)
                        logits = self.model(x_seq, x_cheat)
                        loss = criterion(logits, y)
                        val_loss += loss.item()
                        
                        all_preds.append(torch.sigmoid(logits).cpu().numpy())
                        all_targets.append(y.cpu().numpy())
                        
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                metrics_log = []
                
                for i, tf in enumerate(self.tfs):
                    y_true = all_targets[:, i]
                    y_pred = all_preds[:, i]
                    if len(np.unique(y_true)) > 1:
                        fpr, tpr, _ = roc_curve(y_true, y_pred)
                        prc, rec, _ = precision_recall_curve(y_true, y_pred)
                        metrics_log.append(f"{tf} [ROC: {auc(fpr, tpr):.3f} | PRC: {auc(rec, prc):.3f}]")

                val_loss_str = f" | Val Loss: {avg_val_loss:.4f}"
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), best_model_path)
                    val_loss_str += " (Saved Best!)"
                    
                    with open(os.path.join(checkpoint_dir, "best_metrics_report_cnn.txt"), "w") as f:
                        f.write(f"Best Metrics (Epoch {epoch}):\n" + "\n".join(metrics_log))

            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f}{val_loss_str}")
            if val_loader: print("    -> Stats:", " || ".join(metrics_log))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, last_checkpoint_path)

        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

    def infer(self, df_test):
        self.model.eval()
        test_dataset = self._prepare_data(df_test, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for x_seq, x_cheat, _ in tqdm(test_loader, desc="Inference"):
                x_seq, x_cheat = x_seq.to(self.device), x_cheat.to(self.device)
                logits = self.model(x_seq, x_cheat)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        predictions = {self.tfs[i]: all_preds[:, i] for i in range(3)}
        return pd.DataFrame(predictions, index=df_test.index)