"""Lab 12: 2026 architecture pieces on a CPU.

1. dense vs Mixture-of-Experts TinyLM trained for the same steps (val loss, tokens/s,
   total vs active parameters, expert utilisation per layer)
2. the same MoE with the load-balancing loss switched off (does the router collapse?)
3. multi-token prediction: an extra head that predicts token t+2
4. sliding-window attention on the pretrained base model (perplexity vs window)
5. KV-cache bytes per token: dense MHA vs GQA vs an MLA-style compressed latent

Run:  python3 labs/lab12_moe.py            (quick)
      python3 labs/lab12_moe.py --full
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import time

import torch

from llm.config import preset
from llm.model import TinyLM
from llm.train import TrainConfig, train, estimate_loss, get_batch
from llm.pipeline import get_tokenizer, get_tokens, get_base_model

args = setup("Lab 12: dense vs MoE, MTP, sliding windows, KV-cache sizes")

STEPS = 120 if args.quick else 250
B = 16 if args.quick else 32
T = 64 if args.quick else 128
MOE = dict(use_moe=True, n_experts=4, n_experts_active=1, n_shared_experts=1)

tok = get_tokenizer()
train_tokens, val_tokens = get_tokens(tok)


# ------------------------------------------------------------------ helpers
def active_params(model: TinyLM) -> int:
    """Parameters that touch ONE token: total minus the experts a token does not visit."""
    n = model.num_params(non_embedding=False)
    if model.cfg.use_moe:
        for blk in model.blocks:
            idle = blk.mlp.experts[model.cfg.n_experts_active:]
            n -= sum(p.numel() for e in idle for p in e.parameters())
    return n


def train_variant(name: str, **cfg_kw):
    torch.manual_seed(args.seed)
    cfg = preset("nano", vocab_size=tok.vocab_size, **cfg_kw)
    model = TinyLM(cfg)
    tc = TrainConfig(steps=STEPS, batch_size=B, seq_len=T, optimizer="adamw", lr=1e-3, warmup_steps=10,
                     schedule="cosine", eval_every=STEPS, eval_batches=8, log_every=max(1, STEPS // 4),
                     seed=args.seed)
    print(f"[{name}] total params {model.num_params(non_embedding=False):,} | active {active_params(model):,}")
    t0 = time.perf_counter()
    hist = train(model, train_tokens, val_tokens, tc, verbose=True)
    wall = time.perf_counter() - t0
    tps = STEPS * B * T / wall
    return dict(name=name, model=model, hist=hist, val=hist.val_loss[-1], tps=tps, wall=wall)


@torch.no_grad()
def expert_fractions(model: TinyLM, n_batches: int = 4) -> torch.Tensor:
    """(n_layers, n_experts) fraction of routed tokens per expert over a few val batches."""
    model.eval()
    g = torch.Generator().manual_seed(7)
    counts = torch.zeros(model.cfg.n_layers, model.cfg.n_experts)
    for _ in range(n_batches):
        x, _ = get_batch(val_tokens, B, T, g)
        model(x)
        for i, blk in enumerate(model.blocks):
            counts[i] += blk.mlp.last_expert_counts
    return counts / counts.sum(1, keepdim=True)


@torch.no_grad()
def main_loss(model: TinyLM, n_batches: int = 8) -> float:
    """Next-token loss only (ignores MTP and aux terms), on fixed val windows."""
    model.eval()
    g = torch.Generator().manual_seed(123)
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(val_tokens, B, T, g)
        logits, _ = model(x)
        tot += TinyLM.loss_fn(logits, y, None).item()
    return tot / n_batches


def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes        # K and V


def mla_bytes_per_token(n_layers, d_latent, d_rope, dtype_bytes=2):
    return n_layers * (d_latent + d_rope) * dtype_bytes              # one shared latent per layer


# ------------------------------------------------------------------ 1. dense vs MoE
section(f"1. dense vs MoE (nano, {STEPS} steps, batch {B}x{T})")
dense = train_variant("dense")
moe = train_variant("moe", **MOE)
print(f"\n   {'variant':<8} {'total':>10} {'active':>10} {'val loss':>9} {'val ppl':>8} {'tok/s':>8}")
for r in (dense, moe):
    print(f"   {r['name']:<8} {r['model'].num_params(False):>10,} {active_params(r['model']):>10,} "
          f"{r['val']:>9.3f} {math.exp(r['val']):>8.1f} {r['tps']:>8,.0f}")
ratio = moe["model"].num_params(False) / active_params(moe["model"])
print(f"   MoE stores {ratio:.2f}x more parameters than it uses per token")
check(moe["model"].num_params(False) > 2 * dense["model"].num_params(False),
      "MoE has >2x the TOTAL parameters of the dense model")
check(abs(moe["val"] - dense["val"]) < 1.0, f"dense and MoE reach comparable val loss ({dense['val']:.3f} vs {moe['val']:.3f})")

# ------------------------------------------------------------------ 2. expert utilisation, with and without aux loss
section("2. expert utilisation per layer, with and without the load-balancing loss")
moe_noaux = train_variant("moe-noaux", moe_aux_loss_coef=0.0, **MOE)
frac_on = expert_fractions(moe["model"])
frac_off = expert_fractions(moe_noaux["model"])
E = moe["model"].cfg.n_experts
print(f"   ideal share per expert = 1/{E} = {1 / E:.2f}")
for i in range(frac_on.shape[0]):
    print(f"   layer {i}: aux=0.01 {['%.2f' % v for v in frac_on[i].tolist()]}   "
          f"aux=0 {['%.2f' % v for v in frac_off[i].tolist()]}")
min_on, min_off = frac_on.min().item(), frac_off.min().item()
print(f"   least-used expert share: with aux {min_on:.3f}, without aux {min_off:.3f}")
print(f"   val loss: with aux {moe['val']:.3f}, without aux {moe_noaux['val']:.3f}")
check(min_on > 0.05, f"with the aux loss every expert gets >5% of tokens in every layer (min {min_on:.3f})")
if min_off < 0.05:
    print("   -> without the aux loss at least one expert is (nearly) dead: router collapse")
else:
    print("   -> without the aux loss the router still spread tokens out at this scale (collapse is a risk, not a certainty)")

# ------------------------------------------------------------------ 3. multi-token prediction
section("3. multi-token prediction (mtp_heads=1: an extra head predicts token t+2)")
mtp = train_variant("mtp", mtp_heads=1)
dense_main, mtp_main = main_loss(dense["model"]), main_loss(mtp["model"])
print(f"   next-token val loss: dense {dense_main:.3f} | with MTP head {mtp_main:.3f} "
      f"(the model's training loss also includes 0.3 x the t+2 loss)")
with torch.no_grad():
    g = torch.Generator().manual_seed(123)
    x, y = get_batch(val_tokens, B, T, g)
    _, _, h = mtp["model"](x, return_hidden=True)
    l2 = TinyLM.loss_fn(mtp["model"].mtp_heads[0](h[:, :-1]), y[:, 1:], None).item()
print(f"   the t+2 head's own loss: {l2:.3f} (predicting two tokens ahead is harder than one)")
check(l2 > mtp_main, "predicting t+2 is harder than predicting t+1")
check(abs(mtp_main - dense_main) < 0.5, "the MTP head does not wreck next-token prediction")

# ------------------------------------------------------------------ 4. sliding window on the base model
section("4. sliding-window attention on the pretrained base model (trained with FULL attention)")
base, _ = get_base_model(quick=args.quick, verbose=True)
T_eval = min(128, base.cfg.max_seq_len)
rows = []
for w in [None, 32, 8]:
    base.cfg.sliding_window = w
    loss = estimate_loss(base, val_tokens, 16, T_eval, n_batches=8)
    rows.append((w, loss))
    print(f"   window {str(w):>4}: val loss {loss:.3f} | perplexity {math.exp(loss):6.2f}")
base.cfg.sliding_window = None
check(rows[2][1] > rows[0][1], "a window of 8 tokens hurts a model trained with full attention")
check(rows[1][1] >= rows[0][1] - 1e-3, "a window of 32 is no better than full attention")

# ------------------------------------------------------------------ 5. KV-cache table
section("5. KV-cache bytes per token (bf16): dense MHA vs GQA vs MLA-style latent")
cfgs = [("TinyLM small", 6, 6, 2, 32, 64, 16),
        ("Llama-3-70B-like", 80, 64, 8, 128, 512, 64),
        ("DeepSeek-V3-like", 61, 128, 128, 128, 512, 64)]
print(f"   {'model':<18} {'MHA':>10} {'GQA':>10} {'MLA-ish':>10}   at 128k tokens: MHA / GQA / MLA")
for name, L, h, kvh, hd, d_lat, d_rope in cfgs:
    mha = kv_bytes_per_token(L, h, hd)
    gqa = kv_bytes_per_token(L, kvh, hd)
    mla = mla_bytes_per_token(L, d_lat, d_rope)
    ctx = 131_072
    print(f"   {name:<18} {mha / 1e3:8.1f}KB {gqa / 1e3:8.1f}KB {mla / 1e3:8.1f}KB   "
          f"{mha * ctx / 1e9:6.1f} / {gqa * ctx / 1e9:5.1f} / {mla * ctx / 1e9:4.1f} GB")
check(kv_bytes_per_token(80, 8, 128) * 8 == kv_bytes_per_token(80, 64, 128), "GQA with 8 kv-heads is 8x smaller than 64-head MHA")

# ------------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 3, figsize=(14, 4))
ax = axes[0]
for r in (dense, moe, moe_noaux, mtp):
    ax.plot(r["hist"].step, r["hist"].train_loss, "o-", ms=3, label=r["name"])
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_title("training curves (nano)"); ax.legend()
for ax, frac, title in [(axes[1], frac_on, "expert share, aux=0.01"), (axes[2], frac_off, "expert share, aux=0")]:
    L, E_ = frac.shape
    width = 0.8 / E_
    for e in range(E_):
        ax.bar([i + e * width for i in range(L)], frac[:, e].tolist(), width, label=f"expert {e}")
    ax.axhline(1 / E_, color="k", ls=":", lw=1)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(L)]); ax.set_xticklabels([f"layer {i}" for i in range(L)])
    ax.set_ylim(0, 1); ax.set_title(title); ax.set_ylabel("fraction of tokens")
axes[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "lab12_moe.png")
done()
