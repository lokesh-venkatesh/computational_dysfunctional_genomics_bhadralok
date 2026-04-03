# CFG Algorithms: Application to TF Binding Prediction Project

## Quick Relevance Matrix

| Algorithm | Relevance | Project Stage | Key Use |
|-----------|-----------|---------------|---------|
| **MEME** | ⭐⭐⭐ High | Both | Motif discovery, sequence pattern extraction |
| **WEEDER** | ⭐⭐⭐ High | Both | De novo motif finding, validation |
| **kmer-SVM** | ⭐⭐⭐ High | Final | ML classifier on sequence k-mers |
| **Bowtie** | ⭐⭐⭐ High | Background | Understanding ChIP-seq data pipeline |
| **MACS** | ⭐⭐⭐ High | Background | Understanding peak calling methodology |
| **ChromHMM** | ⭐⭐ Moderate | Both | Chromatin state integration, ATAC utilization |
| **PhaseCons** | ⭐⭐ Moderate | Final | Evolutionary conservation as feature |
| **TopDom** | ⭐⭐ Moderate | Final | TAD boundaries as contextual features |
| **COMPETE** | ⭐⭐ Moderate | Final | Inter-TF competition, multi-TF interactions |
| **cisTopic** | ⭐⭐ Moderate | Final | Latent factor discovery, dimensionality reduction |
| **TopHat** | ⭐⭐ Moderate | Final | Expression correlation validation |
| **C-Origami** | ⭐ Limited | Final | Advanced: 3D contact prediction |
| **Akita** | ⭐⭐⭐ High | Final | Deep learning-based sequence→binding |
| **Deep Learning Methods** | ⭐⭐⭐ High | Final | CNNs, RNNs, attention mechanisms |

---

## Detailed Analysis by Category

### Tier 1: Essential for Milestone & Final (⭐⭐⭐ High Relevance)

#### **MEME / MEME Suite**
**What it does:** Discovers DNA sequence motifs from a set of related sequences.

**Why it matters:**
- Learns position-weight matrices (PWMs) from ChIP-seq peak sequences
- Can validate whether your Markov models discover known TF motifs
- Helps you understand what sequence patterns are predictive

**How to use:**
- Run MEME on bound regions for each TF to extract canonical motifs
- Compare discovered motifs against JASPAR database
- Use high-information motifs as seeds for your model training

**Intermediate milestone:** Not strictly required, but useful for validation/comparison
**Final submission:** High value—demonstrates sequence understanding

---

#### **WEEDER**
**What it does:** De novo motif discovery optimized for finding weak/degenerate binding sites.

**Why it matters:**
- Designed for ChIP-seq data (your use case)
- Finds motifs that are more degenerate/variable than MEME may report
- Better at discovering low-information-content sites

**How to use:**
- Run on bound vs. unbound regions separately
- Identifies TF-specific motifs even with weak signal
- Compare against MEME results—if WEEDER finds different sites, your model may need to account for degeneracy

**Intermediate milestone:** Validation tool
**Final submission:** Feature engineering—motif hits as predictive features

---

#### **kmer-SVM**
**What it does:** Support vector machine classifier trained on k-mer composition.

**Why it matters:**
- Simple but powerful baseline—k-mers capture short-range sequence patterns
- Can model TF binding with minimal assumptions
- Easy to interpret: which k-mers are discriminative?

**How to use:**
- Train on your training chromosome sequences (bound vs. unbound)
- Compare performance to your Markov models
- Use learned k-mer weights to identify important motifs
- Implement as an alternative/complementary approach to Markov models

**Intermediate milestone:** Consider as alternative to pure Markov models
**Final submission:** Strong baseline; combine with other methods

---

#### **Bowtie & MACS**
**What it does:** 
- **Bowtie:** Aligns short reads to reference genome (ultra-fast)
- **MACS:** Calls peaks from ChIP-seq read alignments

**Why it matters:**
- Foundational—your training data comes from ChIP-seq peaks identified by tools like these
- Understanding the peak-calling process improves your interpretation
- Helps you know the biases/limitations of your training labels

**How to use:**
- Don't need to re-run MACS (ENCODE2 data already processed)
- Study MACS parameters: q-value thresholds affect which peaks are "true" binding sites
- Understand that your "B" labels are probabilistic, not certain (explains why sequence alone fails)
- For final submission, consider only high-confidence peaks as positive training examples

**Intermediate milestone:** Background knowledge
**Final submission:** Informative for interpreting results

---

#### **Akita & Deep Learning Methods**
**What it does:** End-to-end deep neural networks that predict TF binding directly from DNA sequence.

**Why it matters:**
- State-of-the-art performance on similar tasks
- Can learn complex, non-linear patterns that Markov models miss
- Naturally integrates multiple features (sequence + ATAC + 3D structure)

**How to use:**
- **Intermediate milestone:** Not required, but if you're interested—implement a simple CNN
- **Final submission:** Primary strategy for best performance
  - Train a CNN on sequence + ATAC + other features
  - Use pre-trained models (e.g., from Enformer, Akita publications) as starting point
  - Fine-tune on K562 data

**Implementation path:**
```
Simple CNN: Conv1D(filters=32) → ReLU → MaxPool → Conv1D(64) → ReLU → Dense(3)
Input: one-hot encoded 200bp sequence (4-channel)
Output: 3 probabilities (CTCF, REST, EP300)
```

---

### Tier 2: Feature Engineering & Advanced Analysis (⭐⭐ Moderate Relevance)

#### **ChromHMM**
**What it does:** Learns a hidden Markov model of chromatin states from histone marks and accessibility.

**Why it matters:**
- Your data includes ATAC (chromatin accessibility), which is a strong TF binding predictor
- ChromHMM tells you *why* certain regions are accessible
- Can define chromatin state features for your model

**How to use:**
- Learn chromatin states on your training data
- Assign each bin a chromatin state
- Use as a feature: e.g., model = P(TF binding | sequence, chromatin state)
- Bins marked "open" are much more likely to have TF binding

**Intermediate milestone:** Optional—focus on sequence first
**Final submission:** Important feature if you're not using deep learning

---

#### **PhaseCons (Conservation)**
**What it does:** Scores genomic regions based on evolutionary conservation.

**Why it matters:**
- TF binding sites that are conserved across species are likely *true* functional sites
- Low-conservation bound regions may be noise in ChIP-seq
- Conservation is complementary to sequence: both matter

**How to use:**
- Fetch phastCons or phyloP scores for each 200bp bin (from UCSC genome browser)
- Add as a feature: model = P(binding | sequence, conservation)
- Down-weight or filter out low-conservation "bound" regions in training

**Intermediate milestone:** Not required
**Final submission:** Feature engineering step—improves specificity

---

#### **TopDom (TAD Detection)**
**What it does:** Calls topologically associating domains (TADs) from Hi-C contact data.

**Why it matters:**
- TF binding at one location can be affected by binding upstream in the same TAD
- TAD boundaries correlate with CTCF binding (one of your TFs!)
- Spatial organization matters, not just sequence

**How to use:**
- Call TADs from publicly available K562 Hi-C data
- Features: is this bin at a TAD boundary? Inside a strong TAD?
- Model TF binding with spatial context

**Intermediate milestone:** Too advanced
**Final submission:** Advanced feature; moderate improvement

---

#### **COMPETE**
**What it does:** Models cooperativity and competition between TF binding sites.

**Why it matters:**
- Your three TFs (CTCF, REST, EP300) likely compete for binding sites
- One TF binding can block another (or facilitate it)
- Single-TF model misses these interactions

**How to use:**
- After building individual TF models, model joint probabilities
- Learn interaction parameters: does CTCF binding suppress REST?
- Use multi-TF training data (all 19 chromosomes) together, not independently

**Intermediate milestone:** Single TF per run—not applicable
**Final submission:** Joint modeling of 3 TFs improves all three predictions

---

#### **cisTopic**
**What it does:** Non-negative matrix factorization to discover latent regulatory topics in chromatin.

**Why it matters:**
- Discovers "regulatory topics": sets of genomic regions that co-vary in accessibility/binding
- Reveals which TFs work together
- Reduces dimensionality: instead of modeling each region independently, learn latent factors

**How to use:**
- Apply to all 19 training chromosomes
- Learn latent topics (e.g., "metabolic regulation", "developmental control")
- Use topic assignments as features for your classifier

**Intermediate milestone:** Not required
**Final submission:** Advanced dimensionality reduction; may improve generalization

---

#### **TopHat**
**What it does:** Aligns RNA-seq reads, maps transcripts.

**Why it matters:**
- TFs regulate genes; if your predicted binding is correct, nearby genes should be expressed
- Can validate predictions: predicted binding → expression correlation

**How to use:**
- Use K562 RNA-seq data (publicly available)
- For each TF, check: do high-binding-score regions have higher expression of downstream genes?
- Use expression data as indirect validation (not a training feature—that would be circular)

**Intermediate milestone:** Not required
**Final submission:** Validation tool; shows biological relevance

---

### Tier 3: Advanced/Specialized (⭐ Limited Relevance)

#### **C-Origami**
**What it does:** Predicts 3D chromatin structure from DNA sequence.

**Why it matters:**
- Models long-range interactions: two distant regions interact if they fold together
- TF at one location can affect binding 100kb away via 3D contact
- Very advanced: most projects don't need this

**How to use:**
- Only if you're going for maximum final submission sophistication
- Predict 3D contacts between chromatin bins
- Use as a feature: bins in close 3D contact have correlated TF binding

**Intermediate milestone:** Not applicable
**Final submission:** Only if other features plateau; high effort for modest gain

---

## Recommended Algorithms for Each Stage

### **Intermediate Milestone (Feb 17)**

**Core algorithm:**
- **Markov models** (from course) — m=0 to 10 order

**Optional validation:**
- **MEME** — extract true motifs, check if Markov rediscovers them
- **WEEDER** — find degenerate sites

**Not needed:**
- Deep learning, 3D structure, advanced features

---

### **Final Submission (April 7)**

**Recommended approach (in priority order):**

1. **Start:** Markov model from intermediate milestone
2. **Add:** Deep learning (Akita-style or custom CNN)
   - Input: sequence + ATAC + optional conservation
   - Output: 3 TF probabilities
3. **Enhance with features:**
   - ChromHMM states (if not deep learning)
   - Conservation scores (PhaseCons)
   - TAD boundaries (TopDom)
   - Multi-TF interactions (COMPETE-inspired)
4. **Validate:**
   - RNA-seq correlation (TopHat)
   - Known motif recovery (MEME)

**Alternative (less ambitious):**
- Ensemble: Markov + kmer-SVM + boosted decision trees
- Feature engineering: conservation + chromatin state + sequence
- No deep learning needed, but performance will be lower

---

## Implementation Strategy

### What NOT to do:
❌ Re-implement MEME, MACS, or other complex tools—use existing software
❌ Use publicly available ENCODE-DREAM solutions (rule violation)
❌ Train on all three TFs simultaneously in Markov model (milestone requires single TF)

### What to do:
✅ **Use existing tools as validation/feature sources:**
  - Run MEME on your predicted binding sites
  - Compare to known motifs
  - Extract conservation scores from public databases

✅ **Implement your own models:**
  - Markov model (required for milestone)
  - CNN or other DL model (final)
  - Feature combination logic

✅ **Acknowledge external tools:**
  - If you use MEME, ChromHMM, etc.—cite them
  - Comment in code: "Downloaded conservation scores from UCSC"

---

## Key Insights from CFG Algorithms

**Why PWMs alone fail (what these algorithms teach you):**
1. **Bowtie + MACS:** Peak calling is probabilistic—your "B" labels have false positives/negatives
2. **ChromHMM:** Chromatin state is as important as sequence—closed chromatin blocks TF regardless of motif
3. **TopDom/C-Origami:** 3D structure matters—distant TF binding affects local sites
4. **COMPETE:** TF-TF competition—one binding blocks another
5. **TopHat:** Gene regulation is multi-step—sequence + context + targets matter
6. **PhaseCons:** Weak/non-conserved sites may be noise
7. **cisTopic:** TFs act in coordinated groups—look for patterns, not isolated sites

**What Akita/Deep Learning teach you:**
- These interactions are too complex for hand-coded rules
- Neural networks can learn the right weighting automatically
- Given enough data, learn_sequence_to_binding is feasible

---

## Deliverables Checklist

### Intermediate (Feb 17):
- [ ] Markov model code (your implementation)
- [ ] ROC/PR curves for m=0 to 10, k=3 to 5
- [ ] AUC values
- [ ] Validation: does model rediscover known TF motifs? (MEME comparison)

### Final (April 7):
- [ ] Predictions for chr3, chr10, chr17 (CTCF, REST, EP300)
- [ ] Justification: which algorithms did you use and why?
- [ ] Presentation: walk through your approach
- [ ] Code comments: acknowledge external tools/papers

---

## Quick Reference: Algorithm → Your Code

```
Dataset (TSV)
    ↓
1_convert_to_fasta.py → Sequences (FASTA)
2_data_as_one_tsv.py → Clean dataset (CSV)
    ↓
┌─ MEME/WEEDER ─→ Motif validation
│
├─ Your Markov Model (Intermediate)
│
├─ Your CNN (Final, optional but recommended)
│   ├─ Input: sequence + ATAC + features
│   └─ Output: 3 TF probabilities
│
├─ Feature engineers
│   ├─ ChromHMM states
│   ├─ Conservation (PhaseCons)
│   ├─ TAD boundaries (TopDom)
│   └─ Multi-TF model (COMPETE ideas)
│
├─ Validation
│   ├─ RNA-seq correlation (TopHat)
│   └─ Motif recovery (MEME)
│
└─ Output: chr3/10/17 predictions (TSV.gz)
```