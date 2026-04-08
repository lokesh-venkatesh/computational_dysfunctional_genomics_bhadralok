import os, torch, numpy as np, pandas as pd, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class ProSNPDataset(Dataset):
    def __init__(self, df):
        self.seqs = df['sequence'].tolist()
        self.atac = (df['ATAC'] == 'B').astype(np.float32).values if 'ATAC' in df.columns else np.zeros(len(df), dtype=np.float32)
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
        return torch.tensor(one_hot), torch.tensor([self.atac[idx]], dtype=torch.float32), self.tfs[idx]

class ProSNPVAE(nn.Module):
    def __init__(self, latent_dim=256, dropout_rate=0.4):
        super(ProSNPVAE, self).__init__()
        
        # Sridhar Hannenhalli CNN Encoder adapted to perfectly pool 200bp -> 25
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 320, 11, padding=5), nn.BatchNorm1d(320), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2), # Output: 320 x 100
            nn.Conv1d(320, 480, 7, padding=3), nn.BatchNorm1d(480), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2), # Output: 480 x 50
            nn.Conv1d(480, 960, 5, padding=2), nn.BatchNorm1d(960), nn.ReLU(), nn.MaxPool1d(2), # Output: 960 x 25
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(960 * 25, latent_dim)
        self.fc_logvar = nn.Linear(960 * 25, latent_dim)
        
        # Decoder (Reconstructs Sequence from Z)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 960 * 25), nn.ReLU())
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(960, 480, 4, stride=2, padding=1), nn.BatchNorm1d(480), nn.ReLU(),
            nn.ConvTranspose1d(480, 320, 4, stride=2, padding=1), nn.BatchNorm1d(320), nn.ReLU(),
            nn.ConvTranspose1d(320, 4, 4, stride=2, padding=1)
        )
        
        # Classifier (Z + ATAC -> 3 TFs)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + 1, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 3)
        )

    def forward(self, x_seq, x_atac):
        enc = self.encoder(x_seq)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)
        
        recon_seq = self.decoder_conv(self.decoder_fc(z).view(-1, 960, 25))
        logits = self.classifier(torch.cat([z, x_atac], dim=1))
        return recon_seq, logits, mu, logvar

class ProSNPVAEWrapper:
    def __init__(self, epochs, batch_size, lr, target_beta=0.2, lambda_cls=500.0):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.target_beta, self.lambda_cls = target_beta, lambda_cls
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ProSNP-VAE initialized on {self.device} | Biological features excluded.")

    def fit(self, train_df, val_df, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, "best.pth")
        
        train_loader = DataLoader(ProSNPDataset(train_df), batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(ProSNPDataset(val_df), batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.model = ProSNPVAE().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr * 5, steps_per_epoch=len(train_loader), epochs=self.epochs)
        pos_weight = torch.tensor([5.0, 5.0, 5.0]).to(self.device)
        scaler = torch.amp.GradScaler('cuda')

        start_epoch, best_val = 1, float('inf')

        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            if 'epoch' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                best_val, start_epoch = checkpoint['best_val'], checkpoint['epoch'] + 1
                print(f"\n[INFO] Auto-resuming! Picked up at Epoch {start_epoch}.")

        for epoch in range(start_epoch, self.epochs + 1):
            current_beta = self.target_beta * min(1.0, epoch / 15.0) # Anneal over 15 epochs
            self.model.train()
            train_loss = 0.0
            
            for x_seq, x_atac, tfs in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                x_seq, x_atac, tfs = [t.to(self.device, non_blocking=True) for t in (x_seq, x_atac, tfs)]
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    r_seq, logits, mu, logvar = self.model(x_seq, x_atac)
                    loss_seq = nn.functional.cross_entropy(r_seq, x_seq.argmax(dim=1), reduction='sum') / x_seq.size(0)
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_seq.size(0)
                    loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                    loss = loss_seq + (current_beta * kld) + (self.lambda_cls * loss_cls)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)

            self.model.eval()
            val_loss, all_preds, all_targets = 0.0, [], []
            
            with torch.no_grad():
                for x_seq, x_atac, tfs in tqdm(val_loader, leave=False, desc=f"Epoch {epoch} [Val]"):
                    x_seq, x_atac, tfs = [t.to(self.device, non_blocking=True) for t in (x_seq, x_atac, tfs)]
                    with torch.amp.autocast('cuda'):
                        r_seq, logits, mu, logvar = self.model(x_seq, x_atac)
                        l_seq = nn.functional.cross_entropy(r_seq, x_seq.argmax(dim=1), reduction='sum') / x_seq.size(0)
                        l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_seq.size(0)
                        l_cls = nn.functional.binary_cross_entropy_with_logits(logits, tfs, reduction='mean', pos_weight=pos_weight)
                        val_loss += l_seq + (current_beta * l_kld) + (self.lambda_cls * l_cls)
                        
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_targets.append(tfs.cpu().numpy())
            
            avg_val_loss = (val_loss / len(val_loader)).item()
            all_preds, all_targets = np.vstack(all_preds), np.vstack(all_targets)
            metrics_log = []
            
            for i, tf in enumerate(['CTCF', 'REST', 'EP300']):
                y_true, y_pred = all_targets[:, i], all_preds[:, i]
                if len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    metrics_log.append(f"{tf} [ROC: {auc(fpr, tpr):.3f} | PRC: {average_precision_score(y_true, y_pred):.3f}]")
            
            val_str = f" | Val Loss: {avg_val_loss:.4f}"
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val': best_val}, best_path)
                val_str += " (Saved Best!)"

            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f}{val_str}")
            print("    -> Stats: " + " || ".join(metrics_log))

    def infer(self, test_df):
        self.model.eval()
        loader = DataLoader(ProSNPDataset(test_df), batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        preds = []
        with torch.no_grad():
            for x_seq, x_atac, _ in tqdm(loader, desc="Inference"):
                x_seq, x_atac = [t.to(self.device, non_blocking=True) for t in (x_seq, x_atac)]
                with torch.amp.autocast('cuda'):
                    _, logits, _, _ = self.model(x_seq, x_atac)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return pd.DataFrame({tf: np.vstack(preds)[:, i] for i, tf in enumerate(['CTCF', 'REST', 'EP300'])}, index=test_df.index)

    # -----------------------------------------------------
    # NEW METHOD: EXTRACT LATENT VECTORS
    # -----------------------------------------------------
    def extract_latents(self, df):
        self.model.eval()
        loader = DataLoader(ProSNPDataset(df), batch_size=self.batch_size, shuffle=False, num_workers=2)
        all_mu = []
        with torch.no_grad():
            for x_seq, x_atac, _ in tqdm(loader, desc="Extracting Latents"):
                x_seq = x_seq.to(self.device, non_blocking=True)
                # Only need the encoder for extraction
                enc = self.model.encoder(x_seq)
                mu = self.model.fc_mu(enc)
                all_mu.append(mu.cpu().numpy())
        return np.vstack(all_mu)