import os
import re
import glob
import math
import time
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ==========================================
# 1. Engineered PyTorch Dataset (Two Streams)
# ==========================================
class EngineeredDataset(Dataset):
    def __init__(self, kmer_matrix, cheat_matrix, atac_labels, tf_labels):
        self.X_kmer = torch.tensor(kmer_matrix, dtype=torch.float32)
        self.X_cheat = torch.tensor(cheat_matrix, dtype=torch.float32)
        self.atac = torch.tensor(atac_labels, dtype=torch.float32).unsqueeze(1)
        self.tfs = torch.tensor(tf_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X_kmer)

    def __getitem__(self, idx):
        return self.X_kmer[idx], self.X_cheat[idx], self.atac[idx], self.tfs[idx]

# ==========================================
# 2. Dual-Stream BindVAE Architecture
# ==========================================
class BindVAELite(nn.Module):
    def __init__(self, num_kmer_features, num_cheat_features, latent_dim=100, motif_embed_dim=32, dropout_rate=0.5):
        super(BindVAELite, self).__init__()
        self.latent_dim = latent_dim
        
        # ------------------------------------------
        # STREAM 1: The VAE (K-mers Only)
        # ------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(num_kmer_features + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Here is the fully restored Decoder!
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_kmer_features) 
        )
        
        # ------------------------------------------
        # STREAM 2: Motif Extractor (Cheat Codes Only)
        # ------------------------------------------
        self.cheat_extractor = nn.Sequential(
            nn.Linear(num_cheat_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(64, motif_embed_dim), 
            nn.ReLU()
        )
        
        # ------------------------------------------
        # FUSION LAYER: Classifier Head
        # ------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + motif_embed_dim + 1, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )

    def encode(self, x_kmer, atac):
        cond = torch.cat([x_kmer, atac], dim=1)
        h = self.encoder(cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_kmer, x_cheat, atac):
        # Stream 1 Pass
        mu, logvar = self.encode(x_kmer, atac)
        z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([z, atac], dim=1)
        recon_x_kmer = self.decoder(z_cond)
        
        # Stream 2 Pass
        cheat_emb = self.cheat_extractor(x_cheat)
        
        # Fusion Pass
        cls_input = torch.cat([z, cheat_emb, atac], dim=1)
        tf_logits = self.classifier(cls_input)
        
        return recon_x_kmer, tf_logits, mu, logvar

# ==========================================
# 3. Training & Inference Wrapper
# ==========================================
class CVAEModel:
    def __init__(self, train_data, val_data=None, epochs=100, batch_size=128, lr=1e-4, 
                 weight_decay=1e-3, dropout_rate=0.5, latent_dim=100, motif_embed_dim=32, 
                 beta=0.1, lambda_cls=50.0, kmer_size=5, pwm_dir="./pwms/"):
                 
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta           
        self.lambda_cls = lambda_cls 
        self.kmer_size = kmer_size
        self.tfs = ['CTCF', 'REST', 'EP300']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initializing Dual-Stream BindVAE on: {self.device}")
        
        nucls = ['A', 'C', 'G', 'T']
        vocab = [''.join(p) for p in product(nucls, repeat=self.kmer_size)]
        
        self.vectorizer = CountVectorizer(
            analyzer='char', 
            ngram_range=(self.kmer_size, self.kmer_size), 
            vocabulary=vocab,
            lowercase=False
        )
        
        self.pwm_dir = pwm_dir
        self.pwms = self._load_pwms()
        
        self.num_kmer_features = 4 ** self.kmer_size
        self.num_cheat_features = 3 + (len(self.pwms) * 3) 
        
        self.model = BindVAELite(
            num_kmer_features=self.num_kmer_features, 
            num_cheat_features=self.num_cheat_features, 
            latent_dim=latent_dim,
            motif_embed_dim=motif_embed_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

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
    def _extract_worker(sequences, pwms):
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        chunk_features = []
        for seq in sequences:
            seq = seq.upper()
            length = len(seq)
            counts = Counter(seq)
            
            gc_content = (counts.get('G', 0) + counts.get('C', 0)) / length
            cpg_oe = (seq.count('CG') * length) / max(counts.get('C', 0) * counts.get('G', 0), 1)
            kmers_3 = [seq[i:i+3] for i in range(length - 2)]
            probs = [c / len(kmers_3) for c in Counter(kmers_3).values()]
            entropy = -sum(p * math.log2(p) for p in probs) if probs else 0
            
            features = [gc_content, cpg_oe, entropy]
            
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

    def _prepare_data(self, df, is_train=True):
        print(f"Extracting {self.kmer_size}-mers and biological cheat codes...")
        
        if is_train:
            counts = self.vectorizer.fit_transform(df['sequence']).toarray().astype(np.float32)
        else:
            counts = self.vectorizer.transform(df['sequence']).toarray().astype(np.float32)
        counts_log = np.log1p(counts)
        
        sequences = df['sequence'].tolist()
        n_jobs = -1 
        chunk_size = max(1, math.ceil(len(sequences) / (os.cpu_count() or 1)))
        seq_chunks = [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_worker)(chunk, self.pwms) for chunk in seq_chunks
        )
        cheat_features = np.vstack(results)
        
        if is_train:
            self.cheat_mean = np.mean(cheat_features, axis=0)
            self.cheat_std = np.std(cheat_features, axis=0) + 1e-8
        cheat_features = (cheat_features - self.cheat_mean) / self.cheat_std
        
        atac_labels = (df['ATAC'] != 'U').astype(np.float32).values
        if is_train or all(tf in df.columns for tf in self.tfs):
            tf_labels = df[self.tfs].apply(lambda x: x != 'U').astype(np.float32).values
        else:
            tf_labels = np.zeros((len(df), 3), dtype=np.float32)
            
        return EngineeredDataset(counts_log, cheat_features, atac_labels, tf_labels)

    def _loss_function(self, recon_x_kmer, x_kmer, tf_logits, tf_targets, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x_kmer, x_kmer, reduction='sum') / x_kmer.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_kmer.size(0)
        
        pos_weight = torch.tensor([5.0, 5.0, 5.0]).to(self.device)
        cls_loss = nn.functional.binary_cross_entropy_with_logits(
            tf_logits, tf_targets, reduction='mean', pos_weight=pos_weight
        )
        
        total_loss = recon_loss + (self.beta * kld) + (self.lambda_cls * cls_loss)
        return total_loss, recon_loss, kld, cls_loss

    def fit(self, checkpoint_dir="checkpoints", resume_from=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, "best_bindvae_model.pth")
        last_checkpoint_path = os.path.join(checkpoint_dir, "last_bindvae_checkpoint.pth")
        
        train_dataset = self._prepare_data(self.train_data, is_train=True)
        val_dataset = self._prepare_data(self.val_data, is_train=False) if self.val_data is not None else None
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None

        start_epoch = 1
        best_val_loss = float('inf')

        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"\nStarting Dual-Stream Training (Epoch {start_epoch} to {self.epochs})...")
        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            train_loss, train_cls = 0.0, 0.0
            
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs} [Train]")
            for x_kmer, x_cheat, atac, tfs in loop:
                x_kmer, x_cheat = x_kmer.to(self.device), x_cheat.to(self.device)
                atac, tfs = atac.to(self.device), tfs.to(self.device)
                
                self.optimizer.zero_grad()
                recon_x_kmer, tf_logits, mu, logvar = self.model(x_kmer, x_cheat, atac)
                loss, r_loss, kld, c_loss = self._loss_function(recon_x_kmer, x_kmer, tf_logits, tfs, mu, logvar)
                
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_cls += c_loss.item()
                loop.set_postfix(Loss=loss.item(), ClsLoss=c_loss.item())

            avg_train_loss = train_loss / len(train_loader)
            
            val_loss_str = ""
            if val_loader:
                self.model.eval()
                val_loss, val_cls = 0.0, 0.0
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for x_kmer, x_cheat, atac, tfs in val_loader:
                        x_kmer, x_cheat = x_kmer.to(self.device), x_cheat.to(self.device)
                        atac, tfs = atac.to(self.device), tfs.to(self.device)
                        
                        recon_x_kmer, tf_logits, mu, logvar = self.model(x_kmer, x_cheat, atac)
                        loss, _, _, c_loss = self._loss_function(recon_x_kmer, x_kmer, tf_logits, tfs, mu, logvar)
                        
                        val_loss += loss.item()
                        val_cls += c_loss.item()
                        
                        all_preds.append(torch.sigmoid(tf_logits).cpu().numpy())
                        all_targets.append(tfs.cpu().numpy())
                        
                avg_val_loss = val_loss / len(val_loader)
                self.scheduler.step(avg_val_loss)
                
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

                val_loss_str = f" | Val Loss: {avg_val_loss:.4f} | Val Cls: {val_cls/len(val_loader):.4f}"
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), best_model_path)
                    val_loss_str += " (Saved Best!)"
                    
                    with open(os.path.join(checkpoint_dir, "best_metrics_report.txt"), "w") as f:
                        f.write(f"Best Metrics (Epoch {epoch}):\n" + "\n".join(metrics_log))

            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f}{val_loss_str}")
            if val_loader: print("    -> Stats:", " || ".join(metrics_log))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, last_checkpoint_path)

        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

    def infer(self, test_data):
        self.model.eval()
        dataset = self._prepare_data(test_data, is_train=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for x_kmer, x_cheat, atac, _ in tqdm(loader, desc="Inference"):
                x_kmer, x_cheat = x_kmer.to(self.device), x_cheat.to(self.device)
                atac = atac.to(self.device)
                _, tf_logits, _, _ = self.model(x_kmer, x_cheat, atac)
                probs = torch.sigmoid(tf_logits)
                all_preds.append(probs.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        predictions = {self.tfs[i]: all_preds[:, i] for i in range(3)}
        return pd.DataFrame(predictions, index=test_data.index)

    def extract_latent_vectors(self, data):
        self.model.eval()
        dataset = self._prepare_data(data, is_train=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        latent_vecs = []
        with torch.no_grad():
            for x_kmer, x_cheat, atac, _ in tqdm(loader, desc="Extracting Latent Space"):
                x_kmer, atac = x_kmer.to(self.device), atac.to(self.device)
                mu, _ = self.model.encode(x_kmer, atac)
                latent_vecs.append(mu.cpu().numpy())
                
        return np.vstack(latent_vecs)