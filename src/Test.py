# %% [markdown]
# Protein function prediction - Notebook cells
#
# This file contains cells (markers `# %%`) so you can open it in VS Code or Jupyter
# and run step-by-step. It implements a practical baseline using a pretrained
# ProtBert encoder (via HuggingFace), taxonomy embedding, multi-label head,
# training loop and inference. Edit paths if your repo layout differs.

# %%
# ## Optional: install dependencies
# # Run only if dependencies missing. On Kaggle, many are preinstalled.
# # Uncomment to run in a notebook environment.
# # !pip install -q transformers torch biopython pandas scikit-learn tqdm

# %%
import os
from pathlib import Path
import json
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# %%
# Data paths - adjust if needed
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'Train'
TEST_DIR = DATA_DIR / 'Test'

TRAIN_SEQS = TRAIN_DIR / 'train_sequences.fasta'
TRAIN_TERMS = TRAIN_DIR / 'train_terms.tsv'
TRAIN_TAX = TRAIN_DIR / 'train_taxonomy.tsv'
GO_OBO = TRAIN_DIR / 'go-basic.obo'
TEST_FASTA = TEST_DIR / 'testsuperset.fasta'

print('paths set - TRAIN_SEQS exists?', TRAIN_SEQS.exists())

# %%
# Load tabular files
train_terms_df = pd.read_csv(TRAIN_TERMS, sep='\t', header=None, names=['EntryID','term','aspect'])
train_tax_df = pd.read_csv(TRAIN_TAX, sep='\t', header=None)
if train_tax_df.shape[1] == 2:
    train_tax_df.columns = ['EntryID', 'taxon']
else:
    # fallback: keep first two columns
    train_tax_df = train_tax_df.iloc[:, :2]
    train_tax_df.columns = ['EntryID','taxon']

print('train_terms', train_terms_df.shape)
print('train_tax', train_tax_df.shape)

# %%
# Parse fasta -> dataframe
records = []
for rec in SeqIO.parse(str(TRAIN_SEQS), 'fasta'):
    records.append({'EntryID': rec.id, 'sequence': str(rec.seq)})
train_seq_df = pd.DataFrame(records)
print('train_seq_df', train_seq_df.shape)

# %%
# Merge sequences, terms and taxonomy
labels_grouped = train_terms_df.groupby('EntryID')['term'].apply(list).rename('terms').reset_index()
df = train_seq_df.merge(labels_grouped, on='EntryID', how='left')
df = df.merge(train_tax_df, on='EntryID', how='left')
df['terms'] = df['terms'].apply(lambda x: x if isinstance(x, list) else [])
print('final df', df.shape)
df.head()

# %%
# Build multilabel binarizer and show distribution
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['terms'])
print('num classes:', len(mlb.classes_))
term_counts = Counter([t for terms in df['terms'] for t in terms])
print('top 10 terms:', term_counts.most_common(10))

# %%
# Train/validation split
train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42,
                                       stratify=(Y.sum(axis=1)>0))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
y_train = Y[train_idx]
y_val = Y[val_idx]
print('train/val sizes', train_df.shape, val_df.shape)

# %%
# Tokenizer helper - ProtBert (transformers). If you prefer a lighter model,
# replace PRETRAINED with another model or implement a simple k-mer embedding.
from transformers import AutoTokenizer, AutoModel

PRETRAINED = 'Rostlab/prot_bert'
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, do_lower_case=False)

def prot_tokenize_sequence(seq, max_len=512):
    seq = ' '.join(list(seq.replace('U','X').upper()))
    toks = tokenizer(seq, padding='max_length', truncation=True, max_length=max_len, return_tensors=None)
    return toks

# %%
# Dataset + collate
class ProteinDataset(Dataset):
    def __init__(self, df, y=None, tokenizer=None, max_len=512, taxon_map=None):
        self.df = df
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.taxon_map = taxon_map or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence']
        toks = self.tokenizer(' '.join(list(seq.upper())), padding='max_length', truncation=True, max_length=self.max_len)
        item = {k: torch.tensor(v) for k,v in toks.items()}
        taxon = str(row.get('taxon','NA'))
        item['taxon'] = torch.tensor(self.taxon_map.get(taxon, 0), dtype=torch.long)
        if self.y is not None:
            item['labels'] = torch.tensor(self.y[idx], dtype=torch.float32)
        item['EntryID'] = row['EntryID']
        return item

def collate_fn(batch):
    out = {}
    out['input_ids'] = torch.stack([b['input_ids'] for b in batch])
    out['attention_mask'] = torch.stack([b['attention_mask'] for b in batch])
    out['taxon'] = torch.stack([b['taxon'] for b in batch])
    if 'labels' in batch[0]:
        out['labels'] = torch.stack([b['labels'] for b in batch])
    out['EntryID'] = [b['EntryID'] for b in batch]
    return out

# %%
# Taxonomy encoding (keep top-N taxa)
taxa = df['taxon'].fillna('NA').astype(str)
taxon_counts = taxa.value_counts()
TOP_TAXA = 200
top_taxa = list(taxon_counts.index[:TOP_TAXA])
taxon_map = {t:i+1 for i,t in enumerate(top_taxa)}
print('taxon classes', len(taxon_map)+1)

# %%
# Model definition: ProtBert encoder + taxon embedding + classifier
class ProtBertForMultiLabel(nn.Module):
    def __init__(self, pretrained, num_labels, taxon_vocab, taxon_embed_dim=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        hidden = self.encoder.config.hidden_size
        self.taxon_emb = nn.Embedding(taxon_vocab, taxon_embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + taxon_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, taxon):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # B,L,H
        mask = attention_mask.unsqueeze(-1)
        summed = (last * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        pooled = summed / counts
        tax_e = self.taxon_emb(taxon)
        x = torch.cat([pooled, tax_e], dim=1)
        logits = self.classifier(x)
        return logits

# %%
# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_LABELS = len(mlb.classes_)
BATCH_SIZE = 8
MAX_LEN = 512

train_ds = ProteinDataset(train_df, y=y_train, tokenizer=lambda s,**k: tokenizer(s,**k), max_len=MAX_LEN, taxon_map=taxon_map)
val_ds = ProteinDataset(val_df, y=y_val, tokenizer=lambda s,**k: tokenizer(s,**k), max_len=MAX_LEN, taxon_map=taxon_map)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = ProtBertForMultiLabel(PRETRAINED, num_labels=NUM_LABELS, taxon_vocab=len(taxon_map)+1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

# %%
# Training loop (simple). For real runs, increase epochs and add gradient clipping / AMP.
from sklearn.metrics import f1_score

def evaluate(model, loader, device, thr=0.5):
    model.eval()
    ys = []
    ys_pred = []
    with torch.no_grad():
        for b in tqdm(loader, desc='Eval'):
            input_ids = b['input_ids'].to(device)
            attention_mask = b['attention_mask'].to(device)
            taxon = b['taxon'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, taxon=taxon)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(b['labels'].cpu().numpy())
            ys_pred.append((probs >= thr).astype(int))
    ys = np.vstack(ys)
    ys_pred = np.vstack(ys_pred)
    micro = f1_score(ys.flatten(), ys_pred.flatten(), average='micro', zero_division=0)
    return micro

EPOCHS = 3
best_val = 0.0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        taxon = batch['taxon'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, taxon=taxon)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({'loss': float(loss.item())})
    val_score = evaluate(model, val_loader, device, thr=0.5)
    print(f'Val micro-F1: {val_score:.4f}')
    if val_score > best_val:
        best_val = val_score
        torch.save(model.state_dict(), 'best_model.pt')
        print('Saved best model')

# %%
# Threshold sweep to find best global threshold
def find_best_threshold(model, loader, device, thrs=np.linspace(0.1,0.9,17)):
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for b in tqdm(loader, desc='Collect preds'):
            input_ids = b['input_ids'].to(device)
            attention_mask = b['attention_mask'].to(device)
            taxon = b['taxon'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, taxon=taxon)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_y.append(b['labels'].cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_y = np.vstack(all_y)
    best_t = 0.5
    best_f = 0.0
    for t in thrs:
        preds = (all_probs >= t).astype(int)
        f = f1_score(all_y.flatten(), preds.flatten(), average='micro', zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t
    return best_t, best_f

# %%
# Inference on test set and prepare submission
test_records = []
for rec in SeqIO.parse(str(TEST_FASTA), 'fasta'):
    test_records.append({'EntryID': rec.id, 'sequence': str(rec.seq)})
test_df = pd.DataFrame(test_records)
test_ds = ProteinDataset(test_df, y=None, tokenizer=lambda s,**k: tokenizer(s,**k), max_len=MAX_LEN, taxon_map=taxon_map)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()
preds = []
eids = []
with torch.no_grad():
    for b in tqdm(test_loader, desc='Predict'):
        input_ids = b['input_ids'].to(device)
        attention_mask = b['attention_mask'].to(device)
        taxon = b['taxon'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, taxon=taxon)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        eids.extend(b['EntryID'])
preds = np.vstack(preds)

# choose threshold found earlier or 0.5 fallback
try:
    best_t, best_f = find_best_threshold(model, val_loader, device)
except Exception:
    best_t = 0.5

submission = []
for eid, probs in zip(eids, preds):
    idxs = np.where(probs >= best_t)[0]
    terms = [mlb.classes_[i] for i in idxs]
    if len(terms) == 0:
        # fallback top-k
        idxs = np.argsort(-probs)[:5]
        terms = [mlb.classes_[i] for i in idxs]
    submission.append({'EntryID': eid, 'predicted_terms': ' '.join(terms)})

submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.tsv', sep='\t', index=False)
