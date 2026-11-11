"""A deliberately small GPT (decoder-only transformer), written to be read.

    tokens -> embeddings (+ positions) -> N x [attention -> MLP] -> logits over the vocabulary

Every block lets each position look at earlier positions only (causal attention), so the model learns
to predict the next token. Training minimizes cross-entropy between that prediction and the real next token.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    ctx: int = 256        # how many tokens the model can see at once
    d: int = 256          # width of every hidden vector
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)   # queries, keys and values in one projection
        self.proj = nn.Linear(c.d, c.d)
        self.mlp = nn.Sequential(nn.Linear(c.d, 4 * c.d), nn.GELU(), nn.Linear(4 * c.d, c.d))
        self.heads, self.drop = c.heads, nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        # causal attention: each token mixes in information from itself and earlier tokens only
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TinyGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying: input and output share the token table

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        # sum of -log p(next token): divide by characters later to get bits per character
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
