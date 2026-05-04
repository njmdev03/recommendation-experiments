import torch
import torch.nn as nn

from models.utils import (
    PositionalEncoding,
    TransformerBlock,
    AdditiveAttention,
)


# ---------------------------
# News Encoder (Deep Transformer)
# ---------------------------
class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, d_model=256, num_heads=8, num_layers=3, positional=True):
        super().__init__()

        self.embed = embedding_matrix

        self.proj = nn.Linear(embedding_matrix.shape[1], d_model)

        if positional:
            self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        self.pool = AdditiveAttention(d_model)

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.proj(x)
        if self.pos:
            x = self.pos(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.pool(x, mask)


# ---------------------------
# User Encoder (Transformer-based)
# ---------------------------
class UserEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        self.pool = AdditiveAttention(d_model)

    def forward(self, hist_vecs, mask):
        x, _ = self.self_attn(hist_vecs, hist_vecs, hist_vecs)

        if mask is not None:
            mask = mask.bool()

        return self.pool(x, mask)


# ---------------------------
# Full Transformer Model
# ---------------------------
class NewsRecModel(nn.Module):
    def __init__(self, embedding_matrix, d_model=256, num_heads=8, num_layers=3, positional=True):
        super().__init__()

        self.news_encoder = NewsEncoder(
            embedding_matrix, d_model, num_heads, num_layers, positional=positional
        )

        self.user_encoder = UserEncoder(d_model, num_heads)

    def encode_news(self, ids, mask):
        B, N, T = ids.shape
        ids = ids.view(B * N, T)
        mask = mask.view(B * N, T)

        out = self.news_encoder(ids, mask)
        return out.view(B, N, -1)

    def forward(self, batch):
        hist = self.encode_news(batch["history"], batch["history_mask"])
        cand = self.encode_news(batch["candidate"], batch["candidate_mask"])

        user = self.user_encoder(hist, batch["history_mask"])

        scores = torch.bmm(cand, user.unsqueeze(-1)).squeeze(-1)
        return scores