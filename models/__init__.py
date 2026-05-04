import torch
import torch.nn as nn

from models.utils import PositionalEncoding, TransformerBlock, AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(self, embedding, d_model, num_heads=8, num_layers=2, positional=True):
        super().__init__()

        self.embedding = embedding
        if positional:
            self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        self.pool = AdditiveAttention(d_model)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids.long())
        if self.pos:
            x = self.pos(x)

        for layer in self.layers:
            x = layer(x, mask)

        # mean pooling (mask-aware)
        x = self.pool(x, mask)

        return self.pool(x)  # (N, D)

class UserEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.pool = AdditiveAttention(dim)

    def forward(self, clicked_news_vecs, mask=None):
        x, _ = self.attn(clicked_news_vecs, clicked_news_vecs, clicked_news_vecs)

        if mask is not None:
            mask = mask.bool()

        return self.pool(x, mask)

class NewsRecModel(nn.Module):
    def __init__(self, embedding, d_model, num_heads=8, num_layers=2):
        super().__init__()

        self.news_encoder = NewsEncoder(
            embedding=embedding,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.user_encoder = UserEncoder()

    def encode_batch(self, ids, mask):
        B, N, L = ids.shape

        ids = ids.view(B * N, L)
        mask = mask.view(B * N, L)

        emb = self.news_encoder(ids, mask)
        return emb.view(B, N, -1)

    def forward(self, batch):
        hist_emb = self.encode_batch(batch["history"], batch["history_mask"])
        user_vec = self.user_encoder(hist_emb)

        cand_emb = self.encode_batch(batch["candidate"], batch["candidate_mask"])

        scores = torch.bmm(
            cand_emb,
            user_vec.unsqueeze(-1)
        ).squeeze(-1)

        return scores