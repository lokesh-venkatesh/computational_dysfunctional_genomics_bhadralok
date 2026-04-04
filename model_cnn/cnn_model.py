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
        # Create a (4, Length) One-Hot Matrix for Conv1d
        # PyTorch expects the Channel dimension first!
        one_hot = np.zeros((4, len(seq)), dtype=np.float32)
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i, nuc in enumerate(seq):
            if nuc in nuc_map:
                one_hot[nuc_map[nuc], i] = 1.0
                
        X_seq = torch.tensor(one_hot, dtype=torch.float32)
        return X_seq, self.X_cheat[idx], self.tfs[idx]

# ==========================================
# 2. Dual-Stream 1D-CNN Architecture
# ==========================================
class DualStreamCNN(nn.Module):
    def __init__(self, num_cheat_features, num_classes=3):
        super(DualStreamCNN, self).__init__()
        
        # STREAM A: Motif Scanner (1D Convolution)
        # Using a kernel size of 15 because CTCF motifs are roughly 15-20bp long.
        self.conv_stream = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), # Smashes any remaining length dimension down to 1
            nn.Flatten()             # Output shape: (Batch, 256)
        )
        
        # STREAM B: Biological Context (Dense Network)
        self.cheat_stream = nn.Sequential(
            nn.Linear(num_cheat_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # FUSION HEAD
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_cheat):
        seq_features = self.conv_stream(x_seq)
        cheat_features = self.cheat_stream(x_cheat)
        combined = torch.cat((seq_features, cheat_features), dim=1)
        logits = self.classifier(combined)
        return logits

# ==========================================
# 3. Model Wrapper & Training Logic
# ==========================================
class DeepCNNModel:
    def __init__(self, pwm_dir="./pwms/", batch_size=128, learning_rate=1e-3, epochs=30, device=None):
        self.pwm_dir = pwm_dir
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tfs = ['CTCF', 'REST', 'EP300']
        
        self.pwms = self._load_pwms()
        self.scaler = StandardScaler()
        self.model = None

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

    def fit(self, train_df, val_df, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        
        print("Preparing Data...")
        train_dataset = self._prepare_data(train_df, is_train=True)
        val_dataset = self._prepare_data(val_df, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        num_cheat_features = train_dataset.X_cheat.shape[1]
        self.model = DualStreamCNN(num_cheat_features=num_cheat_features).to(self.device)
        
        # BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy safely
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        best_val_loss = float('inf')
        best_model_path = os.path.join(save_dir, "best_cnn.pth")

        print(f"\nStarting CNN Training on {self.device}...")
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for x_seq, x_cheat, y in train_loader:
                x_seq, x_cheat, y = x_seq.to(self.device), x_cheat.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_seq, x_cheat)
                loss = criterion(logits, y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_seq, x_cheat, y in val_loader:
                    x_seq, x_cheat, y = x_seq.to(self.device), x_cheat.to(self.device), y.to(self.device)
                    logits = self.model(x_seq, x_cheat)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            status = f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                status += " (Saved Best!)"
            print(status)

        # Load best model for inference later
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