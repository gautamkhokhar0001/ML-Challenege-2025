import os, re, json, random, warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
@dataclass
class CFG:
    seed: int = 42
    train_path: str = "student_resources/dataset/train.csv"
    test_path: str  = "student_resources/dataset/test.csv"

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # hidden=384
    hidden_dim: int = 384
    dropout: float = 0.30
    msd: int = 5
    attn_heads: int = 8

    batch_size: int = 256
    num_workers: int = 2
    max_epochs: int = 10000

    enc_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0

    use_log_price: bool = True          # if True: predict log1p(price)
    combo_alpha: float = 0.6
    smape_eps: float = 1e-4

    out_csv: str = "test_predictions.csv"
    final_ckpt: str = "final_full_train.pth"

CFG = CFG()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALER = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(CFG.seed)

# ============================== PREPROCESS ==============================
LABEL_RE = re.compile(r'^\s*(item\s*name|item|items|bullet\s*point\s*\d*|bullet|specs?|features?|description|details?)\s*:\s*', re.I)
PRICE_LINE_RE = re.compile(r'\b(price|mrp|list\s*price|selling\s*price|discount)\b\s*[:\-]?', re.I)
WHITESPACE_RE = re.compile(r'\s+')

def clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    line = re.sub(r'^[•\-–\*\u2022]+\s*', '', line)
    line = LABEL_RE.sub('', line)
    return line.strip()

def preprocess_catalog(text: str) -> str:
    if pd.isna(text):
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text(separator="\n")
    seen, keep = set(), []
    for raw in str(text).splitlines():
        ln = clean_line(raw)
        if not ln: continue
        if PRICE_LINE_RE.search(ln): continue
        lower_ln = ln.lower()
        if lower_ln.startswith("value:"):
            ln = ln.split(":", 1)[1].strip()
        elif lower_ln.startswith("unit:"):
            ln = ln.split(":", 1)[1].strip()
        key = ln.lower()
        if key in seen or not ln: continue
        seen.add(key)
        keep.append(ln)
    txt = " [SEP] ".join(keep)
    txt = WHITESPACE_RE.sub(' ', txt).strip().lower()
    return txt[:4000]

# ============================== DATA ==============================
df_train = pd.read_csv(CFG.train_path)
if "price" not in df_train.columns:
    raise ValueError("train.csv must contain a 'price' column.")
df_train = df_train.drop(columns=[c for c in ["sample_id", "image_link"] if c in df_train.columns])

df_test = pd.read_csv(CFG.test_path)
test_has_sample_id = "sample_id" in df_test.columns
if "image_link" in df_test.columns:
    df_test = df_test.drop(columns=["image_link"])

df_train["catalog_content_clean"] = df_train["catalog_content"].astype(str).map(preprocess_catalog)
df_test["catalog_content_clean"]  = df_test["catalog_content"].astype(str).map(preprocess_catalog)

y_all = df_train["price"].values.astype(np.float32)
if CFG.use_log_price:
    y_all = np.log1p(y_all)  # targets >= 0

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
encoder   = AutoModel.from_pretrained(CFG.model_name)

class HistoryDataset(Dataset):
    def __init__(self, texts: List[str], targets: np.ndarray | None):
        self.texts = texts
        self.targets = targets
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item

def collate_train(batch):
    texts = [b["text"] for b in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.float)
    return enc, targets

def collate_test(batch):
    texts = [b["text"] for b in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    return enc

train_ds = HistoryDataset(df_train["catalog_content_clean"].tolist(), y_all)
test_ds  = HistoryDataset(df_test["catalog_content_clean"].tolist(), None)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_train)
test_loader  = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_test)

# ============================== MODEL ==============================
class MultiHeadAttnPool(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = max(1, int(num_heads))
        self.score = nn.Linear(d_model, self.num_heads, bias=False)
        self.out   = nn.Linear(d_model * self.num_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B, T, D], mask: [B, T]
        scores = self.score(x)                           # [B, T, H]
        mask_bool = mask.to(dtype=torch.bool)
        # Use a safe large negative in the CURRENT dtype (fp16-safe)
        if scores.dtype == torch.float16:
            neg_large = scores.new_tensor(-1e4)          # >= -65504
        else:
            neg_large = torch.finfo(scores.dtype).min * 0.5
        scores = scores.masked_fill(~mask_bool.unsqueeze(-1), neg_large)
        w = torch.softmax(scores, dim=1)                 # [B, T, H]
        pooled_heads = torch.matmul(w.transpose(1, 2), x)  # [B, H, D]
        pooled = pooled_heads.reshape(x.size(0), -1)     # [B, H*D]
        return self.out(pooled)                          # [B, D]

class SimpleHistoryBasedModel(nn.Module):
    def __init__(self, encoder, hidden_dim=384, dropout=0.15, msd: int = 5, num_heads: int = 8, use_log_price=True):
        super().__init__()
        self.encoder = encoder
        self.pool = MultiHeadAttnPool(hidden_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 8, hidden_dim // 16), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 16, 1),
        )
        self.msd = max(1, int(msd))
        self.use_log_price = use_log_price

    def forward(self, enc_inputs):
        outputs = self.encoder(**enc_inputs)
        last = outputs.last_hidden_state                 # [B, T, D]
        pooled = self.pool(last, enc_inputs["attention_mask"])
        h = self.norm(pooled)
        if self.training and self.msd > 1:
            pred = 0.0
            for _ in range(self.msd):
                pred = pred + self.ff(h)
            pred = pred / self.msd
        else:
            pred = self.ff(h)

        pred = pred.squeeze(-1)
        # If predicting log(price), do NOT force positivity (keep full real range ≥ 0 naturally)
        if not self.use_log_price:
            pred = torch.nn.functional.softplus(pred, beta=1.0)
        return pred

class SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-4, as_percent: bool = False):
        super().__init__()
        self.eps = eps
        self.scale = 100.0 if as_percent else 1.0
    def forward(self, y_pred, y_true):
        y_pred = y_pred.float(); y_true = y_true.float()
        denom = (y_true.abs() + y_pred.abs()).clamp_min(self.eps) * 0.5
        smape = (y_pred - y_true).abs() / denom
        return smape.mean() * self.scale

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.7, eps=1e-4):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.smape = SMAPELoss(eps=eps, as_percent=False)
    def forward(self, y_pred, y_true):
        return self.alpha * self.mae(y_pred, y_true) + (1 - self.alpha) * self.smape(y_pred, y_true)

def compute_smape_numpy(y_pred, y_true, eps=1e-4):
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps) * 0.5
    return np.mean(np.abs(y_pred - y_true) / denom)

# Build
model = SimpleHistoryBasedModel(encoder, hidden_dim=CFG.hidden_dim,
                                dropout=CFG.dropout, msd=CFG.msd,
                                num_heads=CFG.attn_heads, use_log_price=CFG.use_log_price)
model = nn.DataParallel(model).to(DEVICE)

enc_params, head_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    if n.startswith('module.encoder.'): enc_params.append(p)
    else: head_params.append(p)

optimizer = torch.optim.AdamW([
    {"params": enc_params, "lr": CFG.enc_lr, "weight_decay": CFG.weight_decay},
    {"params": head_params, "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
])

total_steps  = len(train_loader) * CFG.max_epochs
warmup_steps = int(CFG.warmup_ratio * total_steps)
scheduler    = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                               num_training_steps=total_steps)
criterion = ComboLoss(alpha=CFG.combo_alpha, eps=CFG.smape_eps)

print(f"Using device: {DEVICE}")
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

best_loss = float('inf')
best_model_path = "best_model.pth"

# ============================== TRAIN ==============================
for epoch in range(1, CFG.max_epochs + 1):
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    prog = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.max_epochs} [train]")

    for enc_inputs, targets in prog:
        enc_inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in enc_inputs.items()}
        targets    = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = model(enc_inputs)            # <- attention mask fix prevents fp16 overflow
            loss  = criterion(preds, targets)

        SCALER.scale(loss).backward()

        if CFG.grad_clip_norm is not None:
            SCALER.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)

        SCALER.step(optimizer)
        SCALER.update()
        scheduler.step()

        loss_val = float(loss.detach().cpu())
        epoch_loss += loss_val; batch_count += 1
        prog.set_postfix(loss=loss_val)

    avg_loss = epoch_loss / max(1, batch_count)
    print(f"\nEpoch {epoch} Average Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch} (loss={best_loss:.6f}) → {best_model_path}")

    # Save a submission every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        test_preds = []
        with torch.no_grad():
            for enc_inputs in tqdm(test_loader, desc=f"Predict [epoch {epoch}]"):
                enc_inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in enc_inputs.items()}
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred = model(enc_inputs)
                test_preds.append(pred.float().cpu().numpy())
        test_preds = np.concatenate(test_preds).reshape(-1)

        # Map back to price space if trained on log
        if CFG.use_log_price:
            test_preds = np.expm1(test_preds)

        if test_has_sample_id and "sample_id" in df_test.columns:
            out_df = pd.DataFrame({"sample_id": df_test["sample_id"].values, "Predicted": test_preds})
        else:
            out_df = pd.DataFrame({"Predicted": test_preds})
        out_path = f"submission_epoch{epoch}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved submission → {out_path}")
        model.train()

print("\nTraining complete.")
print(f"Best epoch loss: {best_loss:.6f}")
