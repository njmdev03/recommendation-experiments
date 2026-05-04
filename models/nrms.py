import torch
import torch.nn as nn

from models.utils import (
    PositionalEncoding,
    AdditiveAttention,
    MultiHeadSelfAttention,
)


# ---------------------------
# News Encoder (NRMS style)
# ---------------------------
class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()

        self.word_embed = embedding_matrix

        d_model = num_heads * head_dim

        self.proj = nn.Linear(embedding_matrix.embedding_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.pool = AdditiveAttention(d_model)

    def forward(self, x, mask):
        # x: (B, T)
        x = self.word_embed(x)
        x = self.proj(x)

        x = self.self_attn(x, mask)
        x = self.dropout(x)

        return self.pool(x, mask)


# ---------------------------
# User Encoder (NRMS style)
# ---------------------------
class UserEncoder(nn.Module):
    def __init__(self, d_model, num_heads=16, dropout=0.2):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.pool = AdditiveAttention(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hist_vecs, mask=None):
        x = self.self_attn(hist_vecs, mask)
        x = self.dropout(x)
        return self.pool(x, mask)


# ---------------------------
# Full NRMS Model
# ---------------------------
class NRMSModel(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16, head_dim=16):
        super().__init__()

        self.news_dim = num_heads * head_dim

        self.news_encoder = NewsEncoder(
            embedding_matrix, num_heads, head_dim
        )

        self.user_encoder = UserEncoder(self.news_dim, num_heads)

    def encode_news(self, ids, mask):
        B, N, T = ids.shape
        ids = ids.view(B * N, T)
        mask = mask.view(B * N, T)

        out = self.news_encoder(ids, mask)
        return out.view(B, N, -1)

    def forward(self, batch):
        hist = self.encode_news(batch["history"], batch["history_word_mask"])
        cand = self.encode_news(batch["candidate"], batch["candidate_word_mask"])

        user = self.user_encoder(hist, batch["history_mask"])

        scores = torch.bmm(cand, user.unsqueeze(-1)).squeeze(-1)

        if "candidate_mask" in batch:
            mask = batch["candidate_mask"]
            # use a value that fits in Half if needed
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, fill_val)

        return scores