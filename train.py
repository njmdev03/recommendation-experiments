import config as conf

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from data_wrappers.mind import MINDDataset
import models

import utils
# from models.simple_transformer import SimpleTransformer

# Training dataset
data = MINDDataset()

print("Training Tokenizers")
# Tokenizers
tok = utils.get_tokenizer([data.get_news_text(nid, tokenize=False) for nid in data.get_news_ids()])

pad_token_id = tok.stoi["<pad>"]
data.tok = tok

print("Build Embeddings")
emb = utils.get_embedding(len(tok), conf.EMBEDDING_SIZE)

quit()

def collate_fn(batch):
    hist_id_batch, cand_id_batch, label_id_batch = zip(*batch)

    # ---- Flatten everything ----
    hist_flat = []
    cand_flat = []
    hist_sizes = []
    cand_sizes = []

    for h, i in zip(hist_batch, cand_batch):
        hist_sizes.append(len(h))
        cand_sizes.append(len(i))

        hist_flat.extend([torch.tensor(x) for x in h])
        cand_flat.extend([torch.tensor(x) for x in i])

    # ---- Pad ----
    hist_padded = pad_sequence(hist_flat, batch_first=True, padding_value=pad_token_id)
    imp_padded = pad_sequence(cand_flat, batch_first=True, padding_value=pad_token_id)

    # ---- Masks ----
    hist_mask = (hist_padded != pad_token_id).long()
    imp_mask = (imp_padded != pad_token_id).long()

    # ---- Reshape back ----
    max_hist = max(hist_sizes)
    max_imp = max(cand_sizes)

    def reshape(flat, sizes, max_len):
        out = []
        idx = 0
        for s in sizes:
            out.append(flat[idx:idx+s])
            idx += s

        padded = []
        for seq in out:
            if len(seq) < max_len:
                pad = torch.zeros(max_len - len(seq), *seq[0].shape, dtype=seq[0].dtype)
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)

        return torch.stack(padded)

    hist_ids = reshape(hist_padded, hist_sizes, max_hist)
    hist_attn = reshape(hist_mask, hist_sizes, max_hist)

    imp_ids = reshape(imp_padded, cand_sizes, max_imp)
    imp_attn = reshape(imp_mask, cand_sizes, max_imp)

    # ---- Labels (index of positive) ----
    labels = torch.tensor([s.index(1) for s in label_batch])

    return {
        "hist_ids": hist_ids,
        "hist_mask": hist_attn,
        "imp_ids": imp_ids,
        "imp_mask": imp_attn,
        "labels": labels
    }

dataloader = DataLoader(
    data,
    batch_size=conf.BATCH_SIZE,
    # shuffle=True,
    collate_fn=collate_fn
)

# Model
model = models.NewsRecModel(vocab_size=len(tok)).to(conf.DEVICE)

# Training
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)

def score(user_vec, candidate_vecs):
    # dot product
    return torch.matmul(candidate_vecs, user_vec.unsqueeze(-1)).squeeze(-1)

try:
    for epoch in range(conf.EPOCHS):
        print(f"Starting epoch {epoch}")
        model.train()

        for batch in tqdm(dataloader):
            for k in batch:
                batch[k] = batch[k].to(conf.DEVICE)

            scores = model(batch)
            loss = criterion(scores, batch["labels"])

            # TODO: Optional testing metrics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: Checkpoint
        # TODO: Eval + Record
except KeyboardInterrupt:
    print("Training interrupted by user")
