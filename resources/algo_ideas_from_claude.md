# Motif Discovery for TF Binding Peaks: Brainstorm & Strategy Guide

## The Problem (Recap)

You have:
- **Grouped sequences**: Each is a contiguous binding peak (1+ bins, 200bp+ long)
- **Known motifs**: JASPAR database has PWMs for CTCF, REST, EP300
- **Goal**: Find motifs (if any) within these peak sequences AND validate/refine your binding predictions

The challenge is that:
1. Not every peak has an obvious motif (noise, weak sites, complex regulation)
2. Your TF may bind via degenerate variants not perfectly matching JASPAR
3. Your TF may recruit co-factors → multi-TF peaks with combined motif patterns
4. Single-bin peaks (200bp) are small—harder to define a "motif"

---

## Strategy Landscape: 6 Approaches

### Tier 1: Rapid Baseline - PWM Scoring (JASPAR Direct)

**The Idea**: Download JASPAR PWM → scan peak sequences → score matches

**Method**:
```
For each TF (CTCF, REST, EP300):
  1. Download PWM from JASPAR
  2. For each peak sequence:
     - Scan using PWM (sliding window)
     - Calculate log-odds score at each position
     - Find max score
     - Record: position, score, strand
  3. Threshold: Keep sites with score > threshold (e.g., top 5%)
  4. Count: how many peaks have ≥1 high-scoring site?
```

**From CFG class**: **Bowtie/MACS** taught you how peak calling works. PWM scoring is the natural next step.

**Advantages**:
- ✓ Extremely fast
- ✓ Baseline for validation ("do real peaks contain real motifs?")
- ✓ Easy to implement (use `pyscores` or write yourself)
- ✓ Results are interpretable

**Limitations**:
- ✗ Only finds sites matching JASPAR exactly
- ✗ JASPAR PWMs are ~6-10 bits information → degenerate variants missed
- ✗ No context awareness (adjacent nucleotides, chromatin state)
- ✗ May report false positives in low-complexity sequences

**When to use**: Start here. First validation: run on your peak sequences → compare to random genomic regions. Expect peaks to have higher average scores.

**Implementation**: Use existing tools like `motifbreakR` (R), `biopython` (Python), or write a simple PWM scanner yourself.

---

### Tier 2: Hybrid - Learn Context-Specific Variants (kmer-SVM + JASPAR)

**The Idea**: Use JASPAR to seed high-confidence sites → extract k-mers around them → train ML model to capture variants

**Method**:

```
Step 1: PWM Scoring (as above)
  - Score all peaks with JASPAR PWM
  - Keep top 10% (high-confidence binding sites)
  
Step 2: Feature Extraction
  - Extract k-mers (k=6 to k=10) around high-scoring sites
  - Also extract k-mers from non-peak (random) regions
  - Build positive/negative examples
  
Step 3: Train kmer-SVM
  - Input: k-mers as feature vectors (or k-mer frequency)
  - Output: Binary classifier (TF-binding site vs. non-binding)
  - Use cross-validation on your training chromosome peaks
  
Step 4: Rescore all peaks
  - Use trained SVM to rescore all peaks
  - Identify sites the PWM missed but SVM catches
  - Visualization: compare PWM scores vs. SVM scores
```

**From CFG class**: **kmer-SVM** is exactly this approach. **MEME** can also be used here instead of k-mers.

**Advantages**:
- ✓ Captures variants and degenerate sites
- ✓ Incorporates local sequence context
- ✓ Semi-supervised: seed with JASPAR, learn from your data
- ✓ Better performance than PWM alone

**Limitations**:
- ✗ Still sequence-only (no chromatin features)
- ✗ Requires choosing k, feature representation
- ✗ Can overfit if few peaks

**When to use**: Medium effort, high value. Use for intermediate milestone + final submission.

**Implementation from class**: You already know this from kmer-SVM papers.

---

### Tier 3: De Novo Discovery - Motif Learning from Peaks (MEME/WEEDER)

**The Idea**: Don't assume JASPAR—learn what motifs are actually enriched in your peaks

**Method A: MEME (Simple)**
```
Input:  peaks_CTCF.fa (your grouped sequences from Script 2)
Output: Discovered motifs (PWM format)

Command:
meme peaks_CTCF.fa -dna -revcomp -nmotifs 5 -maxw 20 -minw 6
  -revcomp: search both strands
  -nmotifs 5: find up to 5 motifs
  -maxw 20: max motif width
  
Result: MEME reports motifs ranked by E-value (enrichment)
```

**Method B: MEME + Validation**
```
Step 1: Run MEME on peak sequences
Step 2: Convert MEME motifs to PWM
Step 3: Compare to JASPAR:
  - Does MEME find the known TF motif? If yes → validates your peaks
  - Does MEME find extra motifs? → may indicate co-factors or noise
```

**Method C: Differential MEME**
```
Step 1: Run MEME on bound peaks (TF='B')
Step 2: Run MEME on unbound peaks (TF='U')
Step 3: Compare: which motifs are enriched in bound vs. unbound?
```

**From CFG class**: **MEME** is exactly the motif discovery algorithm. **WEEDER** is a variant optimized for weak/degenerate sites.

**Advantages**:
- ✓ Discovers actual motifs in your data
- ✓ Can find variants JASPAR doesn't capture
- ✓ Can identify co-factors (extra motifs)
- ✓ E-values tell you significance
- ✓ Validates your peak-finding

**Limitations**:
- ✗ Requires ≥100 sequences for good results (fewer peaks → overfitting)
- ✗ Takes longer than PWM scoring
- ✗ Output is motifs, not per-peak predictions
- ✗ May find noise if peaks are not enriched

**When to use**: For validation/exploration. If you want to publish: "We ran MEME on peaks, confirmed known motifs, found no unexpected co-factors."

**Implementation**: Use MEME-Suite (command-line) or RPMM package (R).

---

### Tier 4: Integrated Features - Context Matters (Random Forest / XGBoost)

**The Idea**: Sequence alone is weak → add chromatin context

**Features to include**:
1. **Sequence**: PWM score, k-mer frequencies, GC content, repeat status
2. **Chromatin**: ATAC status (B/U), conservation score (PhaseCons), TAD boundary proximity
3. **Spatial**: Distance to other TF peaks, peak length, within/outside nucleosome

**Method**:
```
Step 1: Build feature matrix
  For each bin/peak:
    - PWM score (from JASPAR)
    - Top-5 k-mers (frequency or presence)
    - ATAC (B/U)
    - Conservation score
    - Peak length
    - Neighboring TF binding (CTCF near REST? etc.)
    
Step 2: Train ensemble
  - Positive examples: bins with TF='B'
  - Negative examples: bins with TF='U' (or random)
  - Classifier: RandomForest or XGBoost
  - Cross-validate on training chromosomes
  
Step 3: Extract feature importance
  - Which features drive predictions?
  - E.g., "CTCF binding 8× more likely if (PWM score > 0.7 AND ATAC=B)"
  
Step 4: Predict on test chromosomes (3, 10, 17)
  - Use trained model to predict binding probability
```

**From CFG class**: **ChromHMM** taught chromatin states. **TopDom** → TAD context. **PhaseCons** → conservation. This tier integrates all.

**Advantages**:
- ✓ Much better performance than sequence-only
- ✓ Identifies which features matter most
- ✓ Explains *why* TF binds (not just *whether*)
- ✓ Easy to implement (sklearn, XGBoost)
- ✓ Fast predictions

**Limitations**:
- ✗ Requires feature engineering
- ✗ Feature availability (do you have conservation scores?)
- ✗ May overfit if collinear features
- ✗ Less interpretable than explicit PWM

**When to use**: Intermediate + final submission. Medium effort, high improvement in accuracy.

**Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Build feature matrix X, labels y
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
feature_importance = rf.feature_importances_
y_pred = rf.predict_proba(X_test)[:, 1]
```

---

### Tier 5: End-to-End Deep Learning (CNN/RNN)

**The Idea**: Let neural network learn motifs and their interactions end-to-end

**Method A: Convolutional Neural Network**
```
Input:  One-hot encoded sequence (L × 4, where L = sequence length)
        + optional ATAC indicator (L × 1)
        
Layers:
  Conv1D(32 filters, k=10) → learns motif patterns
  MaxPool → reduce dimensionality
  Conv1D(64 filters, k=5) → learns motif combinations
  GlobalMaxPool → aggregate
  Dense(128) → fully connected
  Dense(3) → output (3 TFs)
  Sigmoid → per-TF binding probability
  
Loss: Binary cross-entropy per TF
Optimizer: Adam

Output: For each peak → P(CTCF), P(REST), P(EP300)
```

**Method B: Interpretability**
```
Extract learned filters (convolutional weights)
  - Each filter is (4 × kernel_width)
  - Visualize as PWM / sequence logo
  - "What did the network learn as important?"
  
Attention visualization:
  - Use gradient-based methods (Grad-CAM)
  - Highlight which nucleotides matter for each TF
```

**From CFG class**: **Akita** and **Deep Learning Methods** are CNN-based predictions.

**Advantages**:
- ✓ State-of-the-art performance
- ✓ Learns all dependencies automatically
- ✓ Handles non-linear patterns
- ✓ Can visualize learned motifs
- ✓ Scalable to large datasets

**Limitations**:
- ✗ Requires lots of training data (1000+ peaks)
- ✗ Slow to train
- ✗ Hard to interpret ("black box")
- ✗ High risk of overfitting
- ✗ Requires GPU for reasonable speed

**When to use**: Final submission only, if you have time + computational resources.

**Implementation**:
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv1D(32, 10, activation='relu', input_shape=(seq_len, 5)),
    keras.layers.MaxPooling1D(4),
    keras.layers.Conv1D(64, 5, activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
```

---

### Tier 6: Multi-TF Networks (COMPETE-style)

**The Idea**: TFs don't act independently → model cooperativity and competition

**Method**:
```
Step 1: For each pair of TFs (CTCF-REST, CTCF-EP300, REST-EP300):
  - Extract bins where both are bound (B, B)
  - Extract bins where one is bound (B, U) or (U, B)
  - Calculate: P(TF2=B | TF1=B) vs. P(TF2=B)
  - If P(TF2=B | TF1=B) >> P(TF2=B) → cooperative
  - If P(TF2=B | TF1=B) << P(TF2=B) → competitive
  
Step 2: Build interaction matrix
  - TF × TF → cooperativity score
  - Visualize as heatmap
  
Step 3: Model joint binding
  - Instead of P(CTCF) × P(REST) × P(EP300)
  - Use P(CTCF, REST, EP300) = learned joint model
```

**From CFG class**: **COMPETE** models TF cooperativity explicitly.

**Advantages**:
- ✓ Explains multi-TF binding patterns
- ✓ Better predictions for multi-TF regions
- ✓ Reveals regulatory logic

**Limitations**:
- ✗ Requires sufficient data for each TF pair
- ✗ May not generalize across cell types
- ✗ Very high complexity

**When to use**: Advanced final submission, if time permits.

---

## Practical Recommendation: Multi-Stage Approach

### For Intermediate Milestone (Feb 17)

```
1. Markov Models (required by project)
   
2. Optional validation:
   - Run MEME on your peak sequences
   - Do discovered motifs match known TF motifs?
   - Report: E-value, motif width, enrichment p-value
```

### For Final Submission (April 7)

**Recommended pipeline**:
```
Stage 1: Baseline PWM Scoring
  - Download JASPAR PWMs for all 3 TFs
  - Score each 200bp bin using all 3 PWMs
  - Use scores as features in downstream models

Stage 2: Feature Engineering
  - Compute PWM scores (from stage 1)
  - Extract k-mer frequencies (k=5 to k=8)
  - Get ATAC status (already have)
  - Get conservation if possible (optional)
  
Stage 3: Train Ensemble Model
  - RandomForest on all features
  - 5-fold cross-validation on training chromosomes
  - Extract feature importance
  
Stage 4: Generate Predictions
  - Apply model to chr 3, 10, 17
  - Output: P(CTCF), P(REST), P(EP300) for each bin
  
Stage 5 (Optional): Deep Learning
  - If Tier 3 not sufficient, train CNN
  - Use same features as input (or just sequence)
  - Compare performance
  
Stage 6 (Optional): Multi-TF Integration
  - If time: Model CTCF-REST cooperativity
  - Adjust predictions based on neighboring TFs
```

---

## Specific Algorithms & Tools from CFG Class

| Algorithm | How to Use | Code Source |
|-----------|-----------|------------|
| **MEME** | `meme peaks_CTCF.fa -dna -revcomp -nmotifs 3` | MEME-Suite (command-line) |
| **WEEDER** | Similar to MEME, better for weak sites | WEEDER2 tool |
| **kmer-SVM** | Extract k-mers from peaks → train SVM | sklearn.svm |
| **Bowtie** | Understanding (already done—ENCODE used it) | Not needed |
| **MACS** | Understanding (already done—ENCODE used it) | Not needed |
| **ChromHMM** | Extract chromatin state features | ChromHMM tool or ROADMAP data |
| **PhaseCons** | Get conservation scores from UCSC | phastCons bigWig files |
| **TopDom** | Get TAD boundaries | TopDom tool or 3D Genome Browser |
| **COMPETE** | Model TF cooperativity | Implement yourself |
| **Akita** | Sequence → binding prediction | Pre-trained model or train from scratch |
| **Deep Learning** | CNN for motif + context learning | TensorFlow / PyTorch |

---

## Implementation Priority

### Must Do (Milestone + Final):
1. ✓ Markov models (required)
2. ✓ JASPAR PWM scoring
3. ✓ kmer-SVM or simple feature ML
4. ✓ Feature importance analysis

### Should Do (if time):
5. ✓ MEME validation
6. ✓ ChromHMM / ATAC integration
7. ✓ Conservation scores

### Nice to Have (if you have time + motivation):
8. ⭐ Deep learning CNN
9. ⭐ Multi-TF cooperativity modeling
10. ⭐ Cross-validation across all chromosomes

---

## Quick Start Code Skeleton

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load peaks from your FASTA files
peaks_ctcf = load_fasta('peaks_CTCF.fa')

# Step 1: Get JASPAR PWM scores for each peak
pwm_scores = score_peaks_with_jaspar(peaks_ctcf, 'CTCF')

# Step 2: Extract k-mer features
kmers = extract_kmers(peaks_ctcf, k=6)

# Step 3: Get chromatin features from dataset.csv
atac = get_atac_status(peaks_ctcf)

# Step 4: Combine features
X = np.hstack([pwm_scores, kmers, atac])
y = get_labels(peaks_ctcf)  # B=1, U=0

# Step 5: Train and evaluate
model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation AUC: {scores.mean():.3f}")

# Step 6: Predict on test set
X_test = prepare_test_features('chr3', 'chr10', 'chr17')
y_pred = model.predict_proba(X_test)[:, 1]

# Save predictions
save_predictions(y_pred, 'chr3_10_17_predictions.tsv')
```

---

## Key Insights

1. **PWM alone is weak** — that's exactly what your project is showing. But it's a good baseline.

2. **Variants matter** — JASPAR is consensus, but your TF may prefer slightly different sequences in this cell line. kmer-SVM captures this.

3. **Context is critical** — open chromatin + conservation + neighboring TFs dramatically improve predictions. This is why Tier 4 (ensemble features) often beats Tier 5 (CNN on sequence alone).

4. **De novo motif discovery (MEME) is validation, not prediction** — use it to confirm your peaks contain real motifs, not to predict new peaks.

5. **Start simple, scale up** — Tier 1 (PWM) → Tier 2 (kmer-SVM) → Tier 3 (MEME validation) → Tier 4 (ensemble) → Tier 5 (deep learning).

6. **Multi-TF context is your edge** — most published papers ignore it. If you model CTCF-REST cooperativity, you'll outperform baselines.

---

## Recommended Submission Strategy

**Intermediate (Feb 17):**
- Markov model (required)
- + Brief mention of JASPAR PWM validation (optional bonus)

**Final (April 7):**
- Markov model baseline (comparison point)
- + kmer-SVM or ensemble model (main submission)
- + MEME results (supplementary validation)
- + Feature importance analysis
- + Optional: CNN if time permits

This gives you a **multi-pronged approach** that's defensible (multiple corroborating methods) and shows deep understanding of the problem.