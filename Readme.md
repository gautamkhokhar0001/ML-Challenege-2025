
# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Defenders

**Team Members:** 

Gautam Khokhar:-gautamkhokhar2005@gmail.com

Sandeep Kumar:-sandeepdwivedii032@gmail.com 

Maskeen Singh:-maskeensingh30@gmail.com

**Submission Date:** [13/10/2025]

---

## 1. Executive Summary
We build a robust **price prediction** pipeline driven primarily by product **text**. Cleaned titles, bullets, and descriptions are encoded with a Sentence‑Transformer and passed through an attention‑pooled regression head with multi‑sample dropout. Training uses a SMAPE‑oriented objective on **log(price)** with mixed precision; inference inverts the transform and emits a portal‑ready **`test_out.csv`**.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
We framed the task as single‑target regression with outlier‑resistant behavior. Exploratory checks showed 

(i) heavy‑tailed prices.

 (ii) noisy HTML/marketing fluff and occasional price mentions in text.

 (iii) strong signal from compact textual summaries (titles + top bullets).

**Key Observations:**
- Remove explicit price lines (“MRP/discount/price”) to avoid leakage.
- Normalize whitespace, strip HTML, and collapse bullets to short, information‑dense sentences.
- Log‑transform the target stabilizes gradients and improves convergence.

### 2.2 Solution Strategy
**Approach Type:** Single Model (text‑only baseline)  
**Core Innovation:** Attention pooling over token embeddings + **multi‑sample dropout** for better generalization; training on **log(price)** with a numerically safe SMAPE surrogate.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Text → Tokenizer → Transformer Encoder → Attention Pooling → MLP Head → Pred(log‑price) → expm1() at inference.

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: HTML strip (BeautifulSoup); regex to remove bullets/section labels and price lines; whitespace normalization; field concatenation into `catalog_content_clean`.
- [x] Model type: `sentence-transformers/all-MiniLM-L6-v2` (384‑d)
- [x] Key parameters: dropout=0.30, msd=5, attn_heads=8, AdamW (enc_lr=2e‑5, head_lr=1e‑3, weight_decay=1e‑2), cosine schedule + warmup, AMP, grad‑clip=1.0, batch_size 256–350.

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** less than 43


## 5. Conclusion
A compact text‑only model with attention pooling and MSD yields strong SMAPE while remaining fast and reproducible. Future additions include an image branch and longer‑context encoders to further improve difficult, sparse listings.

---

## Appendix

### A. Code artefacts
https://drive.google.com/drive/folders/1j9ET7JwEYgKenWU3WSabQ9H6HqVGpoUT

