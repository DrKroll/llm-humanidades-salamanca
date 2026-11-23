"""Un GPT diminuto (inspirado en nanoGPT de Andrej Karpathy y en minimind).

No hace falta entender este archivo para el taller, pero está escrito para
que se pueda leer. Las piezas son las mismas que en los grandes modelos:

  tokens -> embeddings -> [bloque Transformer] x N -> probabilidades del siguiente token

Cada bloque Transformer tiene:
  1. Atención (self-attention): cada token «mira» a los tokens anteriores.
  2. Una red neuronal pequeña (MLP) que procesa cada posición.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4000
    block_size: int = 128   # cuántos tokens puede «ver» a la vez
    n_layer: int = 4        # número de bloques Transformer
    n_head: int = 4         # cabezas de atención por bloque
    n_embd: int = 192       # tamaño de cada vector (embedding)
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Atención «causal»: un token solo puede mirar hacia atrás, nunca al futuro."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd debe ser divisible por n_head"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # consulta, clave, valor
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, cabezas, T, C/cabezas)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def attention_weights(self, x):
        """Devuelve los pesos de atención (para visualizarlos)."""
        B, T, C = x.shape
        q, k, _ = self.qkv(x).split(C, dim=2)
        q, k = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float("-inf"))
        return att.softmax(dim=-1)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # conexión residual
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # posición -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # compartir pesos (como GPT-2)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Secuencia demasiado larga ({T} > {self.cfg.block_size})"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            # «pérdida» (loss): qué tan sorprendido está el modelo por el token real
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
