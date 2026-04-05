import os, torch, numpy as np, pandas as pd, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class TriModalDataset(Dataset):
    def __init__(self, df, prefix, use_context=True):
        self.seqs = df['sequence'].tolist()
        kmer_file = f"./data/processed/{prefix}_context_kmers.npy" if use_context else f"./data/processed/{prefix}_local_kmers.npy"
        
        # Load preprocessed arrays (Memory Mapped for extreme efficiency)
        self.kmers = torch.tensor(np.log1p(np.load(kmer_file, mmap_mode='r')), dtype=torch.float32)
        self.base_cheats = torch.tensor(np.load(f"./data/processed/{prefix}_base_cheats.npy", mmap_mode='r'), dtype=torch.float32)
        self.pwm_ctcf = torch.tensor(np.load(f"./data/processed/{prefix}_ctcf_pwm.npy", mmap_mode='r'), dtype=torch.float32)
        self.pwm_rest = torch.tensor(np.load(f"./data/processed/{prefix}_rest_pwm.npy", mmap_mode='r'), dtype=torch.float32)
        self.pwm_ep300 = torch.tensor(np.load(f"./data/processed/{prefix}_ep300_pwm.npy", mmap_mode='r'), dtype=torch.float32)
        
        # Handle test set gracefully (where labels might be 'U' entirely or missing)
        self.has_labels = 'CTCF' in df.columns
        if self.has_labels:
            self.tfs = torch.tensor(df[['CTCF', 'REST', 'EP300']].apply(lambda x: x != 'U').astype(np.float32).values)
        else:
            self.tfs = torch.zeros((len(df), 3), dtype=torch.float32)

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:200].upper().ljust(200, 'N')
        one_hot = np.zeros((4, 200), dtype=np.float32)
        for i, nuc in enumerate(seq):
            if nuc in {'A': 0, 'C': 1, 'G': 2, 'T': 3}: one_hot[{'A': 0, 'C': 1, 'G': 2, 'T': 3}[nuc], i] = 1.0
            
        return (torch.tensor(one_hot), self.kmers[idx], self.base_cheats[idx], 
                self.pwm_ctcf[idx], self.pwm_rest[idx], self.pwm_ep300[idx], self.tfs[idx])

class TriModalVAE(nn.Module):
    def __init__(self, latent_dim=100, dropout_rate=0.4):
        super(TriModalVAE, self).__init__()
        
        # STREAM 1: Sequence CNN
        self.seq_enc = nn.Sequential(
            nn.Conv1d(4, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(),
            nn.Linear(128 * 50, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        # STREAM 2: K-mers (256-dim for K=4)
        self.kmer_enc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        # STREAM 3: Base Cheats (ATAC, GC, CpG, AT-tracts = 4 features)
        self.cheat_enc = nn.Sequential(
            nn.Linear(4, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        
        # LATENT BOTTLENECK (256 + 128 + 32 = 416)
        self.fc_mu = nn.Linear(416, latent_dim)
        self.fc_logvar = nn.Linear(416, latent_dim)
        
        # RECONSTRUCTION DECODERS
        self.seq_dec = nn.Sequential(
            nn.Linear(latent_dim, 128 * 50), nn.ReLU(),
            nn.Unflatten(1, (128, 50)),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 4, 5, padding=2)
        )
        self.kmer_dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # TASK-SPECIFIC BRANCHES (Z + specific PWM -> Prediction)
        self.ctcf_head = nn.Sequential(nn.Linear(latent_dim + 1, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1))
        self.rest_head = nn.Sequential(nn.Linear(latent_dim + 1, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1))
        self.ep300_head = nn.Sequential(nn.Linear(latent_dim + 1, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1))

    def forward(self, x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300):
        # Encode
        e_seq = self.seq_enc(x_seq)
        e_kmer = self.kmer_enc(x_kmer)
        e_cheat = self.cheat_enc(x_cheats)
        
        # Fuse & Reparameterize
        cond = torch.cat([e_seq, e_kmer, e_cheat], dim=1)
        mu, logvar = self.fc_mu(cond), self.fc_logvar(cond)
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)
        
        # Decode
        recon_seq = self.seq_dec(z)
        recon_kmer = self.kmer_dec(z)
        
        # Branch Classifiers
        out_ctcf = self.ctcf_head(torch.cat([z, p_ctcf], dim=1))
        out_rest = self.rest_head(torch.cat([z, p_rest], dim=1))
        out_ep300 = self.ep300_head(torch.cat([z, p_ep300], dim=1))
        
        logits = torch.cat([out_ctcf, out_rest, out_ep300], dim=1)
        return recon_seq, recon_kmer, logits, mu, logvar

class TriModalModel:
    def __init__(self, epochs, batch_size, lr, beta, lambda_cls, use_context):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.beta, self.lambda_cls, self.use_context = beta, lambda_cls, use_context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Tri-Modal VAE initialized on {self.device} | Context Kmers: {use_context}")

    def fit(self, train_df, val_df, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path, last_path = os.path.join(checkpoint_dir, "best.pth"), os.path.join(checkpoint_dir, "last.pth")
        
        train_loader = DataLoader(TriModalDataset(train_df, "train", self.use_context), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TriModalDataset(val_df, "val", self.use_context), batch_size=self.batch_size, shuffle=False)

        self.model = TriModalVAE().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        pos_weight = torch.tensor([5.0, 5.0, 5.0]).to(self.device)

        best_val = float('inf')
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs}")
            for x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs in loop:
                x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs = [t.to(self.device) for t in (x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs)]
                optimizer.zero_grad()
                r_seq, r_kmer, logits, mu, logvar = self.model(x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)
                
                # Composite Loss
                loss_seq = nn.functional.cross_entropy(r_seq, x_seq.argmax(dim=1), reduction='sum') / x_seq.size(0)
                loss_kmer = nn.functional.mse_loss(r_kmer, x_kmer, reduction='sum') / x_kmer.size(0)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_seq.size(0)
                loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                
                loss = loss_seq + loss_kmer + (self.beta * kld) + (self.lambda_cls * loss_cls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss, all_preds, all_targets = 0.0, [], []
            with torch.no_grad():
                for x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs in val_loader:
                    x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs = [t.to(self.device) for t in (x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs)]
                    r_seq, r_kmer, logits, mu, logvar = self.model(x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)
                    
                    l_seq = nn.functional.cross_entropy(r_seq, x_seq.argmax(dim=1), reduction='sum') / x_seq.size(0)
                    l_kmer = nn.functional.mse_loss(r_kmer, x_kmer, reduction='sum') / x_kmer.size(0)
                    l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_seq.size(0)
                    l_cls = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                    
                    val_loss += l_seq + l_kmer + (self.beta * l_kld) + (self.lambda_cls * l_cls)
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_targets.append(tfs.cpu().numpy())
            
            avg_val = val_loss.item() / len(val_loader)
            scheduler.step(avg_val)
            
            val_str = f" | Val Loss: {avg_val:.4f}"
            if avg_val < best_val:
                best_val = avg_val
                torch.save(self.model.state_dict(), best_path)
                val_str += " (Saved Best!)"

            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss/len(train_loader):.4f}{val_str}")
        if os.path.exists(best_path): self.model.load_state_dict(torch.load(best_path))

    def infer(self, test_df, prefix):
        self.model.eval()
        loader = DataLoader(TriModalDataset(test_df, prefix, self.use_context), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, _ in tqdm(loader, desc="Inference"):
                x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300 = [t.to(self.device) for t in (x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)]
                _, _, logits, _, _ = self.model(x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return pd.DataFrame({tf: np.vstack(preds)[:, i] for i, tf in enumerate(['CTCF', 'REST', 'EP300'])}, index=test_df.index)