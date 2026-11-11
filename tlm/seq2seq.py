"""Character-level encoder-decoder transformer: romanized word -> Telugu-script word.

    "yenti"  --encoder-->  one vector per Latin letter
             --decoder-->  writes Telugu characters one at a time:  ఏ ం ట ి <end>

The ENCODER reads the whole romanized word at once (it may look both left and right).
The DECODER writes the output left to right; at every step it attends to its own output so far
(causal) and to the encoder's vectors (cross-attention), which is how it "looks at" the input spelling.
"""
import torch
import torch.nn as nn

PAD, BOS, EOS = 0, 1, 2


class Vocab:
    def __init__(self, texts):
        self.chars = sorted({c for t in texts for c in t})
        self.idx = {c: i + 3 for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars) + 3

    def encode(self, s, maxlen):
        return [BOS] + [self.idx.get(c, PAD) for c in s][: maxlen - 2] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= 3:
                out.append(self.chars[i - 3])
        return "".join(out)


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d=256, layers=3, heads=4, maxlen=40, dropout=0.1):
        super().__init__()
        self.src_emb, self.tgt_emb = nn.Embedding(src_vocab, d, padding_idx=PAD), nn.Embedding(tgt_vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(maxlen, d)
        self.t = nn.Transformer(d, heads, layers, layers, dim_feedforward=4 * d, dropout=dropout, batch_first=True,
                                norm_first=True)
        self.out = nn.Linear(d, tgt_vocab)
        self.maxlen = maxlen

    def _emb(self, emb, x):
        return emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))

    def forward(self, src, tgt_in):
        mask = self.t.generate_square_subsequent_mask(tgt_in.shape[1], device=src.device, dtype=torch.bool)
        h = self.t(self._emb(self.src_emb, src), self._emb(self.tgt_emb, tgt_in), tgt_mask=mask,
                   src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD,
                   memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src, maxlen=None):
        """Write the output one character at a time, always taking the most likely next character."""
        maxlen = maxlen or self.maxlen
        mem = self.t.encoder(self._emb(self.src_emb, src), src_key_padding_mask=src == PAD)
        ys = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(maxlen - 1):
            mask = self.t.generate_square_subsequent_mask(ys.shape[1], device=src.device, dtype=torch.bool)
            h = self.t.decoder(self._emb(self.tgt_emb, ys), mem, tgt_mask=mask, memory_key_padding_mask=src == PAD)
            nxt = self.out(h[:, -1]).argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
