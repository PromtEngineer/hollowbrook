"""Model configuration for TinyLM.

Chapter 6 explains every field. The presets are sized so that:

* ``nano``  trains to readable text in ~1 minute on a laptop CPU (used by ``--quick`` labs)
* ``small`` trains in ~5–10 minutes on a laptop CPU (the default course model)
* ``medium`` is what you would use with a single GPU

Parameter counts are printed by ``TinyLM.num_params()``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class TinyLMConfig:
    # --- sizes -----------------------------------------------------------
    vocab_size: int = 4096          # V: number of distinct tokens the tokenizer produces
    d_model: int = 192              # d: width of the residual stream
    n_layers: int = 6               # number of Transformer blocks stacked
    n_heads: int = 6                # h: query heads
    n_kv_heads: int = 2             # key/value heads (GQA when < n_heads, MQA when 1)
    d_ff: Optional[int] = None      # hidden width of the MLP; default = 8/3 * d_model rounded to 64
    max_seq_len: int = 256          # T_max: longest sequence RoPE tables are built for

    # --- architecture switches --------------------------------------------
    rope_theta: float = 10_000.0    # RoPE base frequency (raised for long-context extension)
    tie_embeddings: bool = True     # share input embedding and output projection weights
    dropout: float = 0.0            # modern LLM pretraining uses no dropout
    norm_eps: float = 1e-5
    use_moe: bool = False           # replace the MLP with a mixture of experts (Chapter 12)
    n_experts: int = 4
    n_experts_active: int = 1
    n_shared_experts: int = 0       # DeepSeekMoE-style always-on experts
    moe_aux_loss_coef: float = 0.01 # load-balancing auxiliary loss weight
    mtp_heads: int = 0              # multi-token prediction heads (0 = off; Chapter 12)
    sliding_window: Optional[int] = None  # local attention window (None = full causal)

    # --- misc -------------------------------------------------------------
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_ff is None:
            # SwiGLU uses three matrices; 8/3*d keeps parameter count equal to a 4d ReLU MLP
            self.d_ff = int(round(8 * self.d_model / 3 / 64) * 64) or 64
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly into heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TinyLMConfig":
        return cls(**d)


PRESETS: dict[str, dict] = {
    # ~0.9M non-embedding params; quick smoke tests
    "nano": dict(d_model=96, n_layers=3, n_heads=3, n_kv_heads=1, max_seq_len=128),
    # ~2.7M non-embedding params; the default course model
    "small": dict(d_model=192, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=256),
    # ~19M non-embedding params; for a GPU
    "medium": dict(d_model=384, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=512),
}


def preset(name: str, **overrides) -> TinyLMConfig:
    """Return a TinyLMConfig for a named preset, with optional field overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    d = dict(PRESETS[name])
    d.update(overrides)
    return TinyLMConfig(**d)
