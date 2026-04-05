import os, torch, numpy as np, pandas as pd, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class TriModalDataset(Dataset):
    def __init__(self, df, prefix, use_context=True):
        self.seqs = df['sequence'].tolist()
        kmer_file = f"./data/processed/{prefix}_context_kmers.npy" if use_context else f"./data/processed/{prefix}_local_kmers.npy"
        
        self.kmers = np.load(kmer_file)
        self.base_cheats = np.load(f"./data/processed/{prefix}_base_cheats.npy")
        self.pwm_ctcf = np.load(f"./data/processed/{prefix}_ctcf_pwm.npy")
        self.pwm_rest = np.load(f"./data/processed/{prefix}_rest_pwm.npy")
        self.pwm_ep300 = np.load(f"./data/processed/{prefix}_ep300_pwm.npy")
        
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
            
        x_kmer = torch.tensor(np.log1p(self.kmers[idx]), dtype=torch.float32)
        x_cheats = torch.tensor(self.base_cheats[idx], dtype=torch.float32)
        p_ctcf = torch.tensor(self.pwm_ctcf[idx], dtype=torch.float32)
        p_rest = torch.tensor(self.pwm_rest[idx], dtype=torch.float32)
        p_ep300 = torch.tensor(self.pwm_ep300[idx], dtype=torch.float32)
            
        return (torch.tensor(one_hot), x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, self.tfs[idx])

class TriModalVAE(nn.Module):
    def __init__(self, latent_dim=128, dropout_rate=0.4): # Increased latent_dim to 128
        super(TriModalVAE, self).__init__()
        
        # UPGRADED STREAM 1: Deeper & Wider Sequence CNN
        self.seq_enc = nn.Sequential(
            nn.Conv1d(4, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout1d(dropout_rate/2),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout1d(dropout_rate/2),
            nn.Flatten(),
            nn.Linear(256 * 50, 512), nn.BatchNorm1d(512), nn.ReLU()
        )
        
        # STREAM 2: K-mers 
        self.kmer_enc = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        
        # STREAM 3: Base Cheats 
        self.cheat_enc = nn.Sequential(
            nn.Linear(4, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        
        # LATENT BOTTLENECK (512 + 128 + 32 = 672)
        self.fc_mu = nn.Linear(672, latent_dim)
        self.fc_logvar = nn.Linear(672, latent_dim)
        
        # RECONSTRUCTION DECODERS
        self.seq_dec = nn.Sequential(
            nn.Linear(latent_dim, 256 * 50), nn.ReLU(),
            nn.Unflatten(1, (256, 50)),
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 4, 5, padding=2)
        )
        self.kmer_dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # TASK-SPECIFIC BRANCHES 
        self.ctcf_head = nn.Sequential(nn.Linear(latent_dim + 1, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1))
        self.rest_head = nn.Sequential(nn.Linear(latent_dim + 1, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1))
        self.ep300_head = nn.Sequential(nn.Linear(latent_dim + 1, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 1))

    def forward(self, x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300):
        e_seq = self.seq_enc(x_seq)
        e_kmer = self.kmer_enc(x_kmer)
        e_cheat = self.cheat_enc(x_cheats)
        
        cond = torch.cat([e_seq, e_kmer, e_cheat], dim=1)
        mu, logvar = self.fc_mu(cond), self.fc_logvar(cond)
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)
        
        recon_seq = self.seq_dec(z)
        recon_kmer = self.kmer_dec(z)
        
        out_ctcf = self.ctcf_head(torch.cat([z, p_ctcf], dim=1))
        out_rest = self.rest_head(torch.cat([z, p_rest], dim=1))
        out_ep300 = self.ep300_head(torch.cat([z, p_ep300], dim=1))
        
        logits = torch.cat([out_ctcf, out_rest, out_ep300], dim=1)
        return recon_seq, recon_kmer, logits, mu, logvar

class TriModalModel:
    def __init__(self, epochs, batch_size, lr, target_beta, lambda_cls, use_context):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.target_beta, self.lambda_cls, self.use_context = target_beta, lambda_cls, use_context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Tri-Modal VAE initialized on {self.device} | Context Kmers: {use_context}")

    def fit(self, train_df, val_df, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path, last_path = os.path.join(checkpoint_dir, "best.pth"), os.path.join(checkpoint_dir, "last.pth")
        
        # UPGRADE: Added Multiprocessing (num_workers=6) and Memory Pinning
        # FIX: Set num_workers=0 for Windows compatibility and remove persistent_workers
        train_loader = DataLoader(
            TriModalDataset(train_df, "train", self.use_context), 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,      # Changed to 0
            pin_memory=True
        )
        
        val_loader = DataLoader(
            TriModalDataset(val_df, "val", self.use_context), 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,      # Changed to 0
            pin_memory=True
        )

        self.model = TriModalVAE().to(self.device)
        
        # UPGRADE: PyTorch 2.0 torch.compile() (With safe fallback for Windows)
        try:
            self.model = torch.compile(self.model)
            print("Successfully compiled model with torch.compile().")
        except Exception as e:
            print(f"torch.compile() bypassed (Expected on Windows environments): {e}")

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr * 5, steps_per_epoch=len(train_loader), epochs=self.epochs)
        pos_weight = torch.tensor([5.0, 5.0, 5.0]).to(self.device)

        # Dynamic AMP Device Setup
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        amp_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
        scaler = torch.amp.GradScaler(device_type) if device_type == 'cuda' else None

        best_val = float('inf')
        for epoch in range(1, self.epochs + 1):
            
            current_beta = self.target_beta * min(1.0, epoch / 10.0)
            self.model.train()
            train_loss = 0.0
            
            train_loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs} [Train]")
            for x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs in train_loop:
                # UPGRADE: Added non_blocking=True
                x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs = [t.to(self.device, non_blocking=True) for t in (x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs)]
                
                # UPGRADE: set_to_none=True is slightly faster than standard zero_grad()
                optimizer.zero_grad(set_to_none=True)
                
                # UPGRADE: Automatic Mixed Precision (AMP)
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                    r_seq, r_kmer, logits, mu, logvar = self.model(x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)
                    
                    loss_seq = nn.functional.cross_entropy(r_seq, x_seq.argmax(dim=1), reduction='sum') / x_seq.size(0)
                    loss_kmer = nn.functional.mse_loss(r_kmer, x_kmer, reduction='sum') / x_kmer.size(0)
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_seq.size(0)
                    loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                    
                    loss = loss_seq + loss_kmer + (current_beta * kld) + (self.lambda_cls * loss_cls)
                
                # Scale gradients if CUDA, else standard backprop for CPU
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    
                scheduler.step()
                
                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item(), beta=current_beta)
                
            avg_train_loss = train_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            
            val_loop = tqdm(val_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs} [Val]")
            with torch.no_grad():
                for x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs in val_loop:
                    x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs = [t.to(self.device, non_blocking=True) for t in (x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, tfs)]
                    
                    # Apply AMP to Validation as well
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                        r_seq, r_kmer, logits, mu, logvar = self.model(x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)
                        
                        l_seq = nn.functional.cross_entropy(r_seq, x_seq.argmax(dim=1), reduction='sum') / x_seq.size(0)
                        l_kmer = nn.functional.mse_loss(r_kmer, x_kmer, reduction='sum') / x_kmer.size(0)
                        l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_seq.size(0)
                        l_cls = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                        
                        batch_val_loss = l_seq + l_kmer + (current_beta * l_kld) + (self.lambda_cls * l_cls)
                        
                    val_loss += batch_val_loss
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_targets.append(tfs.cpu().numpy())
            
            avg_val_loss = val_loss.item() / len(val_loader)
            
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            metrics_log = []
            
            for i, tf in enumerate(['CTCF', 'REST', 'EP300']):
                y_true = all_targets[:, i]
                y_pred = all_preds[:, i]
                if len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    metrics_log.append(f"{tf} [ROC: {auc(fpr, tpr):.3f} | PRC: {average_precision_score(y_true, y_pred):.3f}]")
                else:
                    metrics_log.append(f"{tf} [ROC: N/A | PRC: N/A]")
            
            val_str = f" | Val Loss: {avg_val_loss:.4f}"
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(self.model.state_dict(), best_path)
                val_str += " (Saved Best!)"

            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f}{val_str}")
            print("    -> Stats: " + " || ".join(metrics_log))
            
        if os.path.exists(best_path): self.model.load_state_dict(torch.load(best_path))

    def infer(self, test_df, prefix):
        self.model.eval()
        # FIX: Changed num_workers to 0 here as well
        loader = DataLoader(TriModalDataset(test_df, prefix, self.use_context), batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        preds = []
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        amp_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
        
        with torch.no_grad():
            for x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300, _ in tqdm(loader, desc="Inference"):
                x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300 = [t.to(self.device, non_blocking=True) for t in (x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)]
                
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                    _, _, logits, _, _ = self.model(x_seq, x_kmer, x_cheats, p_ctcf, p_rest, p_ep300)
                
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return pd.DataFrame({tf: np.vstack(preds)[:, i] for i, tf in enumerate(['CTCF', 'REST', 'EP300'])}, index=test_df.index)