import torch
import torch.nn as nn
import math


# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------
# Additive Attention (NRMS core)
# ---------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden=200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        h = torch.tanh(self.proj(x))
        scores = self.query(h).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


# ---------------------------
# Multi-Head Self Attention
# ---------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.h = num_heads
        self.dh = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)

        if mask is not None:
            mask = mask.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# ---------------------------
# Transformer Block
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask)
        x = self.norm1(x)

        x = x + self.ff(x)
        x = self.norm2(x)
        return x