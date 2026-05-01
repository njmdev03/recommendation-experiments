import torch
import torch.nn as nn

from models.utils import PositionalEncoding, TransformerBlock


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        self.pool = nn.Linear(d_model, d_model)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x, mask)

        # mean pooling (mask-aware)
        mask = mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        return self.pool(x)  # (N, D)

class UserEncoder(nn.Module):
    def forward(self, hist_emb):
        return hist_emb.mean(dim=1)

class NewsRecModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()

        self.news_encoder = NewsEncoder(vocab_size, d_model)
        self.user_encoder = UserEncoder()

    def encode_batch(self, ids, mask):
        B, N, L = ids.shape

        ids = ids.view(B * N, L)
        mask = mask.view(B * N, L)

        emb = self.news_encoder(ids, mask)
        return emb.view(B, N, -1)

    def forward(self, batch):
        hist_emb = self.encode_batch(batch["hist_ids"], batch["hist_mask"])
        user_vec = self.user_encoder(hist_emb)

        imp_emb = self.encode_batch(batch["imp_ids"], batch["imp_mask"])

        scores = torch.bmm(
            imp_emb,
            user_vec.unsqueeze(-1)
        ).squeeze(-1)

        return scores