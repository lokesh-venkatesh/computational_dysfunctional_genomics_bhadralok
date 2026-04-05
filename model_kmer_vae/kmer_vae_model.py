import os, re, glob, math, torch, numpy as np, pandas as pd, torch.nn as nn, torch.optim as optim
from collections import Counter
from joblib import Parallel, delayed
from itertools import product
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class KmerDataset(Dataset):
    def __init__(self, kmer_sparse, cheat_matrix, atac_labels, tf_labels):
        # We DO NOT convert kmer_sparse to a dense tensor here! We keep it as a scipy sparse matrix.
        self.X_kmer_sparse = kmer_sparse 
        self.X_cheat = torch.tensor(cheat_matrix, dtype=torch.float32)
        self.atac = torch.tensor(atac_labels, dtype=torch.float32).unsqueeze(1)
        self.tfs = torch.tensor(tf_labels, dtype=torch.float32)

    def __len__(self): 
        return self.X_kmer_sparse.shape[0]

    def __getitem__(self, idx):
        # Convert ONLY this specific row to a dense numpy array, then to a tensor
        row_dense = self.X_kmer_sparse[idx].toarray().squeeze(0)
        x_kmer_tensor = torch.tensor(row_dense, dtype=torch.float32)
        
        return x_kmer_tensor, self.X_cheat[idx], self.atac[idx], self.tfs[idx]

class BindVAEKmer(nn.Module):
    def __init__(self, num_kmer_features, num_cheat_features, latent_dim=100, motif_embed_dim=32, dropout_rate=0.5):
        super(BindVAEKmer, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_kmer_features + 1, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, num_kmer_features) 
        )
        
        self.cheat_extractor = nn.Sequential(
            nn.Linear(num_cheat_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate / 2.0),
            nn.Linear(64, motif_embed_dim), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + motif_embed_dim + 1, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3) 
        )

    def forward(self, x_kmer, x_cheat, atac):
        cond = torch.cat([x_kmer, atac], dim=1)
        h = self.encoder(cond)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)
        
        recon_x_kmer = self.decoder(torch.cat([z, atac], dim=1))
        cheat_emb = self.cheat_extractor(x_cheat)
        tf_logits = self.classifier(torch.cat([z, cheat_emb, atac], dim=1))
        
        return recon_x_kmer, tf_logits, mu, logvar

class KmerVAEModel:
    def __init__(self, train_data, val_data=None, epochs=100, batch_size=128, lr=1e-4, weight_decay=1e-3, dropout_rate=0.5, latent_dim=100, motif_embed_dim=32, beta=0.1, lambda_cls=500.0, kmer_size=5, pwm_dir="./pwms/"):
        self.train_data, self.val_data = train_data, val_data
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.weight_decay, self.dropout_rate = weight_decay, dropout_rate
        self.latent_dim, self.motif_embed_dim = latent_dim, motif_embed_dim
        self.beta, self.lambda_cls, self.kmer_size, self.pwm_dir = beta, lambda_cls, kmer_size, pwm_dir
        self.tfs = ['CTCF', 'REST', 'EP300']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing K-mer VAE on: {self.device}")
        
        nucls = ['A', 'C', 'G', 'T']
        vocab = [''.join(p) for p in product(nucls, repeat=self.kmer_size)]
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.kmer_size, self.kmer_size), vocabulary=vocab, lowercase=False)
        self.pwms = self._load_pwms()
        self.scaler = StandardScaler()

    def _load_pwms(self):
        pwms = {}
        for file in glob.glob(os.path.join(self.pwm_dir, "*.pfm")):
            tf_name = os.path.basename(file).split('.')[0].upper()
            matrix = []
            with open(file, 'r') as f:
                for line in f:
                    if not line.startswith(">"):
                        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                        if numbers: matrix.append([float(x) for x in numbers])
            if len(matrix) == 4:
                pfm = np.array(matrix).T 
                pwms[tf_name] = np.log2((pfm + 0.1) / (pfm.sum(axis=1, keepdims=True) + 0.4) / 0.25)
        return pwms

    @staticmethod
    def _extract_worker(sequences, pwms):
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        chunk_features = []
        for seq in sequences:
            seq = seq.upper()
            length = len(seq)
            counts = Counter(seq)
            gc = (counts.get('G', 0) + counts.get('C', 0)) / max(length, 1)
            cpg = (seq.count('CG') * length) / max(counts.get('C', 0) * counts.get('G', 0), 1)
            features = [gc, cpg]
            if pwms:
                one_hot = np.zeros((length, 4))
                for i, nuc in enumerate(seq):
                    if nuc in nuc_map: one_hot[i, nuc_map[nuc]] = 1
                for pwm in pwms.values():
                    if length >= pwm.shape[0]:
                        scores = np.einsum('nwc,wc->n', np.lib.stride_tricks.sliding_window_view(one_hot, window_shape=(pwm.shape[0], 4)).squeeze(), pwm)
                        features.extend([np.max(scores), len(scores[scores > 0])])
                    else:
                        features.extend([0, 0])
            chunk_features.append(features)
        return np.array(chunk_features, dtype=np.float32)

    def _prepare_data(self, df, is_train=True):
        print("Extracting features...")
        
        # 1. Fit/Transform but LEAVE AS SPARSE MATRIX
        sparse_counts = self.vectorizer.fit_transform(df['sequence']) if is_train else self.vectorizer.transform(df['sequence'])
        
        # 2. Apply log1p directly to the compressed non-zero data (super fast, uses zero extra memory)
        sparse_counts.data = np.log1p(sparse_counts.data)
        sparse_counts = sparse_counts.astype(np.float32)
        
        seqs = df['sequence'].tolist()
        chunk_size = max(1, math.ceil(len(seqs) / (os.cpu_count() or 1)))
        results = Parallel(n_jobs=-1)(delayed(self._extract_worker)(chunk, self.pwms) for chunk in [seqs[i:i + chunk_size] for i in range(0, len(seqs), chunk_size)])
        X_cheat = np.vstack(results)
        
        X_cheat = self.scaler.fit_transform(X_cheat) if is_train else self.scaler.transform(X_cheat)
        atac = (df['ATAC'] != 'U').astype(np.float32).values
        tfs = df[self.tfs].apply(lambda x: x != 'U').astype(np.float32).values
        
        # Pass the sparse matrix into the Dataset
        return KmerDataset(sparse_counts, X_cheat, atac, tfs)

    def fit(self, checkpoint_dir="checkpoints", resume_from=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path, last_path = os.path.join(checkpoint_dir, "best.pth"), os.path.join(checkpoint_dir, "last.pth")
        
        train_loader = DataLoader(self._prepare_data(self.train_data, True), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self._prepare_data(self.val_data, False), batch_size=self.batch_size, shuffle=False) if self.val_data is not None else None

        self.model = BindVAEKmer(train_loader.dataset.X_kmer_sparse.shape[1], train_loader.dataset.X_cheat.shape[1], self.latent_dim, self.motif_embed_dim, self.dropout_rate).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        pos_weight = torch.tensor([5.0, 5.0, 5.0]).to(self.device)

        start_epoch, best_val = 1, float('inf')
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch, best_val = checkpoint['epoch'] + 1, checkpoint.get('best_val_loss', float('inf'))

        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs}")
            for x_kmer, x_cheat, atac, tfs in loop:
                x_kmer, x_cheat, atac, tfs = x_kmer.to(self.device), x_cheat.to(self.device), atac.to(self.device), tfs.to(self.device)
                optimizer.zero_grad()
                recon, logits, mu, logvar = self.model(x_kmer, x_cheat, atac)
                
                recon_loss = nn.functional.mse_loss(recon, x_kmer, reduction='sum') / x_kmer.size(0)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_kmer.size(0)
                cls_loss = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                
                loss = recon_loss + (self.beta * kld) + (self.lambda_cls * cls_loss)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_str = ""
            if val_loader:
                self.model.eval()
                val_loss, all_preds, all_targets = 0.0, [], []
                with torch.no_grad():
                    for x_kmer, x_cheat, atac, tfs in val_loader:
                        x_kmer, x_cheat, atac, tfs = x_kmer.to(self.device), x_cheat.to(self.device), atac.to(self.device), tfs.to(self.device)
                        recon, logits, mu, logvar = self.model(x_kmer, x_cheat, atac)
                        
                        recon_loss = nn.functional.mse_loss(recon, x_kmer, reduction='sum') / x_kmer.size(0)
                        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_kmer.size(0)
                        val_loss += recon_loss + (self.beta * kld) + (self.lambda_cls * nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight))
                        
                        all_preds.append(torch.sigmoid(logits).cpu().numpy())
                        all_targets.append(tfs.cpu().numpy())
                
                avg_val = val_loss.item() / len(val_loader)
                scheduler.step(avg_val)
                metrics = [f"{tf} [ROC: {auc(*roc_curve(np.vstack(all_targets)[:, i], np.vstack(all_preds)[:, i])[:2]):.3f}]" for i, tf in enumerate(self.tfs)]
                val_str = f" | Val Loss: {avg_val:.4f}"
                if avg_val < best_val:
                    best_val = avg_val
                    torch.save(self.model.state_dict(), best_path)
                    val_str += " (Saved Best!)"

            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss/len(train_loader):.4f}{val_str}")
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val}, last_path)
        if os.path.exists(best_path): self.model.load_state_dict(torch.load(best_path))

    def infer(self, test_data):
        self.model.eval()
        loader = DataLoader(self._prepare_data(test_data, False), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for x_kmer, x_cheat, atac, _ in tqdm(loader, desc="Inference"):
                _, logits, _, _ = self.model(x_kmer.to(self.device), x_cheat.to(self.device), atac.to(self.device))
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return pd.DataFrame({tf: np.vstack(preds)[:, i] for i, tf in enumerate(self.tfs)}, index=test_data.index)

    def extract_latent_vectors(self, data):
        self.model.eval()
        loader = DataLoader(self._prepare_data(data, False), batch_size=self.batch_size, shuffle=False)
        vecs = []
        with torch.no_grad():
            for x_kmer, _, atac, _ in tqdm(loader, desc="Extracting Latent"):
                cond = torch.cat([x_kmer.to(self.device), atac.to(self.device)], dim=1)
                vecs.append(self.model.encoder(cond).cpu().numpy()) # Return raw features or MU depending on plotting style
        return np.vstack(vecs)