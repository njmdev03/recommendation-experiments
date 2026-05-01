import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.pe = pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, L, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(x, mask)
        x = self.norm1(x)

        x = x + self.ff(x)
        x = self.norm2(x)

        return x

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