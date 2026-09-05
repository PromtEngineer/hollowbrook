"""TinyLM: a small decoder-only Transformer, written to be read.

Chapters 5, 6 and 12 walk through this file. The architecture is the 2024–2026
"standard recipe" (Llama / Qwen / DeepSeek family):

    token embedding
    N × [ RMSNorm -> attention (GQA + RoPE, optional sliding window) -> residual add
          RMSNorm -> MLP (SwiGLU) or Mixture-of-Experts        -> residual add ]
    RMSNorm -> output projection (tied to the embedding)

Shapes use the notation from the course: B batch, T sequence length, d = d_model,
h = n_heads, hd = head_dim, V = vocab_size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TinyLMConfig


# ------------------------------------------------------------------ RMSNorm
class RMSNorm(nn.Module):
    """Normalise each token's vector to unit RMS, then scale by a learned gain.

    Unlike LayerNorm there is no mean subtraction and no bias; cheaper and just as good.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:            # x: (B, T, d)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


# --------------------------------------------------------------------- RoPE
def rope_tables(head_dim: int, max_seq_len: int, theta: float = 10_000.0,
                device=None) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for Rotary Position Embeddings.

    Returns (cos, sin), each of shape (max_seq_len, head_dim // 2).
    Pair i of the head vector is rotated by angle  pos * theta^(-2i/head_dim):
    low i -> fast-spinning (local detail), high i -> slow (long-range order).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(pos, inv_freq)                # (T_max, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate pairs of channels of x (B, h, T, hd) by position-dependent angles.

    cos/sin are (T, hd/2) for the positions of x. Uses the "half-split" pairing
    (channel i with channel i + hd/2), which is what Llama does.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ------------------------------------------------------------------ KV cache
@dataclass
class LayerCache:
    k: Optional[Tensor] = None      # (B, kv_heads, T_cached, hd)
    v: Optional[Tensor] = None

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class KVCache:
    """One LayerCache per layer. ``cache.pos`` is how many tokens are stored."""

    def __init__(self, n_layers: int) -> None:
        self.layers = [LayerCache() for _ in range(n_layers)]

    @property
    def pos(self) -> int:
        return self.layers[0].length

    def truncate(self, length: int) -> None:
        for lc in self.layers:
            if lc.k is not None:
                lc.k = lc.k[:, :, :length]
                lc.v = lc.v[:, :, :length]


# ---------------------------------------------------------------- attention
class Attention(nn.Module):
    """Causal multi-head attention with grouped-query heads and RoPE."""

    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.last_attn: Optional[Tensor] = None     # filled when record_attention=True

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: Optional[LayerCache] = None,
                record_attention: bool = False) -> Tensor:
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)      # (B, h, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)   # (B, kv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        past = 0
        if cache is not None:
            past = cache.length
            k, v = cache.append(k, v)                                        # (B, kv, past+T, hd)
        S = k.shape[2]                                                       # keys available

        # Grouped-query attention: each kv head serves n_heads // n_kv_heads query heads.
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # scores[b, h, i, j] = how much query i attends to key j
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                  # (B, h, T, S)
        mask = causal_mask(T, S, past, self.cfg.sliding_window, x.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        if record_attention:
            self.last_attn = attn.detach()
        if self.dropout and self.training:
            attn = F.dropout(attn, self.dropout)
        out = attn @ v                                                       # (B, h, T, hd)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * hd)
        return self.o_proj(out)


def causal_mask(T: int, S: int, past: int, window: Optional[int], device) -> Tensor:
    """Boolean (T, S) mask: query i (absolute position past+i) may see key j iff j <= past+i,
    and, with a sliding window, j > past+i-window."""
    qpos = torch.arange(past, past + T, device=device)[:, None]
    kpos = torch.arange(S, device=device)[None, :]
    allowed = kpos <= qpos
    if window is not None:
        allowed &= kpos > qpos - window
    return allowed


# ---------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------- MoE
class MoE(nn.Module):
    """Mixture of Experts: a router sends each token to its top-k experts.

    Chapter 12. Each expert is a small SwiGLU MLP. Optional *shared* experts see every
    token (DeepSeekMoE). The auxiliary loss pushes the router toward using all experts
    evenly (Switch Transformer style); the model adds it to the main loss.
    """

    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.n_experts, self.k = cfg.n_experts, cfg.n_experts_active
        d_ff = max(64, cfg.d_ff // max(1, cfg.n_experts_active))  # keep active params ≈ dense MLP
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(cfg.d_model, d_ff) for _ in range(cfg.n_experts)])
        self.shared = nn.ModuleList([MLP(cfg.d_model, d_ff) for _ in range(cfg.n_shared_experts)])
        self.aux_coef = cfg.moe_aux_loss_coef
        self.aux_loss: Tensor = torch.zeros(())
        self.last_expert_counts: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:                    # x: (B, T, d)
        B, T, d = x.shape
        flat = x.reshape(-1, d)                                # (B*T, d)
        logits = self.router(flat)                             # (B*T, E)
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(self.k, dim=-1)            # (B*T, k)
        topk_p = topk_p / topk_p.sum(-1, keepdim=True)         # renormalise chosen weights

        out = torch.zeros_like(flat)
        counts = torch.zeros(self.n_experts, device=x.device)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (topk_i == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            counts[e] = token_idx.numel()
            out.index_add_(0, token_idx, expert(flat[token_idx]) * topk_p[token_idx, slot, None])
        for expert in self.shared:
            out = out + expert(flat)

        # Load-balancing loss: E * sum_e (fraction of tokens routed to e) * (mean router prob of e)
        frac = counts / max(1, flat.shape[0] * self.k)
        mean_p = probs.mean(0)
        self.aux_loss = self.aux_coef * self.n_experts * (frac * mean_p).sum()
        self.last_expert_counts = counts.detach()
        return out.view(B, T, d)


# -------------------------------------------------------------------- Block
class Block(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MoE(cfg) if cfg.use_moe else MLP(cfg.d_model, cfg.d_ff)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: Optional[LayerCache] = None,
                record_attention: bool = False) -> Tensor:
        # The residual stream x is read by each sub-layer and written back to by addition.
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, record_attention)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ------------------------------------------------------------------- TinyLM
class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        # Multi-token prediction: extra heads predict token t+2, t+3, ... (simplified; Chapter 12)
        self.mtp_heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                                        for _ in range(cfg.mtp_heads)])
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 / nanoGPT trick: shrink the residual-writing projections so the stream's
        # variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ----------------------------------------------------------------- info
    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Training FLOPs per token ≈ 6·N (forward + backward) + attention term."""
        cfg = self.cfg
        attn = 12 * cfg.n_layers * cfg.d_model * cfg.max_seq_len   # score + value matmuls, fwd+bwd
        return 6 * self.num_params(non_embedding=False) + attn

    def extend_context(self, new_max_seq_len: int, theta: Optional[float] = None) -> None:
        """Long-context extension (Chapter 13): rebuild RoPE tables, optionally with a
        larger base frequency (as in Llama 3 / 'ABF') so existing positions keep their meaning."""
        self.cfg.max_seq_len = new_max_seq_len
        if theta is not None:
            self.cfg.rope_theta = theta
        cos, sin = rope_tables(self.cfg.head_dim, new_max_seq_len, self.cfg.rope_theta,
                               device=self.embed.weight.device)
        self.rope_cos, self.rope_sin = cos, sin

    # -------------------------------------------------------------- forward
    def forward(self, idx: Tensor, targets: Optional[Tensor] = None,
                cache: Optional[KVCache] = None, record_attention: bool = False,
                loss_mask: Optional[Tensor] = None, return_hidden: bool = False):
        """idx: (B, T) token ids. Returns (logits, loss). loss is None without targets.

        With ``cache`` the model processes only the *new* tokens in idx and appends to
        the cache. ``loss_mask`` (B, T) zeroes out positions that should not be trained
        on (e.g. the prompt during SFT; Chapter 15). ``targets`` may use -100 to ignore.
        """
        B, T = idx.shape
        past = cache.pos if cache is not None else 0
        assert past + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        cos = self.rope_cos[past:past + T]
        sin = self.rope_sin[past:past + T]

        x = self.embed(idx)                                            # (B, T, d)
        for i, block in enumerate(self.blocks):
            layer_cache = cache.layers[i] if cache is not None else None
            x = block(x, cos, sin, layer_cache, record_attention)
        x = self.final_norm(x)
        logits = self.lm_head(x)                                       # (B, T, V)

        loss = None
        if targets is not None:
            loss = self.loss_fn(logits, targets, loss_mask)
            for k, head in enumerate(self.mtp_heads, start=2):        # predict t+k
                if T > k:
                    mtp_logits = head(x[:, :-(k - 1)])
                    mtp_targets = targets[:, k - 1:]
                    loss = loss + 0.3 * self.loss_fn(mtp_logits, mtp_targets, None)
            if self.cfg.use_moe:
                loss = loss + sum(b.mlp.aux_loss for b in self.blocks)
        if return_hidden:
            return logits, loss, x
        return logits, loss

    @staticmethod
    def loss_fn(logits: Tensor, targets: Tensor, loss_mask: Optional[Tensor]) -> Tensor:
        V = logits.shape[-1]
        if loss_mask is None:
            return F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), ignore_index=-100)
        per_token = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1),
                                    ignore_index=-100, reduction="none")
        m = loss_mask.reshape(-1).float()
        return (per_token * m).sum() / m.sum().clamp(min=1.0)

    # ---------------------------------------------------------- convenience
    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.n_layers)

    @torch.no_grad()
    def attention_maps(self, idx: Tensor) -> list[Tensor]:
        """Run a forward pass and return each layer's attention (B, h, T, T)."""
        self.forward(idx, record_attention=True)
        return [b.attn.last_attn for b in self.blocks]

    def save(self, path: str, tokenizer_path: Optional[str] = None, extra: Optional[dict] = None) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict(),
                    "tokenizer_path": tokenizer_path, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TinyLM":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(TinyLMConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
