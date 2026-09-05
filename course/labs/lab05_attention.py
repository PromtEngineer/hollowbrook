"""Lab 5: attention — a soft, learned lookup.

What you will see:
  1. Attention built line by line on a toy (T=5, d=8): scores, scaling, mask, softmax, weighted sum.
  2. Why we divide by sqrt(d).
  3. llm.model.Attention verified against torch's scaled_dot_product_attention (MHA and GQA).
  4. RoPE: scores depend only on the *distance* between query and key positions.
  5. GQA parameter savings and the KV-cache memory formula (TinyLM and a 70B-class model).
  6. Attention maps of the trained TinyLM on a Storyland sentence.
  7. A sliding-window mask.

Run:  python3 labs/lab05_attention.py            (--quick: nano base model)
      python3 labs/lab05_attention.py --full     (small base model)
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from llm.config import preset
from llm.model import Attention, TinyLM, apply_rope, causal_mask, rope_tables
from llm.pipeline import get_base_model, BASE_FULL, BASE_QUICK, COURSE_DIR

args = setup("Lab 5: attention")
plt = plt()
torch.manual_seed(args.seed)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)


def wait_for_checkpoint(path: str, every: int = 30, max_wait: int = 600) -> bool:
    """Another lab may be pretraining the base model right now; poll before training our own."""
    if os.path.exists(path):
        return True
    rel = os.path.relpath(path, COURSE_DIR)
    print(f"[lab05] {rel} not found; polling every {every}s for up to {max_wait // 60} min before training one here")
    waited = 0
    while waited < max_wait and not os.path.exists(path):
        time.sleep(every)
        waited += every
        print(f"   ... waited {waited}s; runs/ contains {sorted(os.listdir(os.path.dirname(path)))}")
    if os.path.exists(path):
        time.sleep(3)
        return True
    return False


# ----------------------------------------------------------------------------
section("1. Attention from scratch on a toy: T=5 tokens, d=8 channels")
T, d = 5, 8
x = torch.randn(T, d)                                # (T, d): one row per token
Wq, Wk, Wv = torch.randn(d, d) / math.sqrt(d), torch.randn(d, d) / math.sqrt(d), torch.randn(d, d) / math.sqrt(d)
q, k, v = x @ Wq, x @ Wk, x @ Wv                      # (T, d) each: what I look for / what I offer / what I carry
scores = q @ k.T                                      # (T, T): scores[i, j] = q_i · k_j
scaled = scores / math.sqrt(d)                        # keep the numbers O(1) regardless of d
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # query i may look at keys 0..i only
masked = scaled.masked_fill(~mask, float("-inf"))     # -inf -> exactly 0 after softmax
weights = F.softmax(masked, dim=-1)                   # (T, T): each row sums to 1
out = weights @ v                                     # (T, d): row i = weighted average of v_0..v_i
print("scores / sqrt(d)  (before the mask):\n", scaled)
print("attention weights (after mask + softmax):\n", weights)
print("output shape", tuple(out.shape))
check(torch.allclose(weights.sum(-1), torch.ones(T)), "every row of the attention matrix sums to 1")
check(bool((weights.triu(1) == 0).all()), "causal mask: no weight on future tokens (upper triangle is 0)")
check(torch.allclose(out[0], v[0]), "token 0 can only see itself, so its output is exactly v_0")
ref = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=True)[0, 0]
check(torch.allclose(out, ref, atol=1e-6), "our 6 lines match torch.nn.functional.scaled_dot_product_attention")

# ----------------------------------------------------------------------------
section("2. Why divide by sqrt(d)?  (dot products grow with d; softmax saturates)")
for dd in (8, 64, 512):
    qq, kk = torch.randn(1000, dd), torch.randn(1000, dd)
    dots = (qq * kk).sum(-1)
    peak_raw = F.softmax((qq[:8] @ kk[:8].T), dim=-1).max(-1).values.mean()
    peak_scaled = F.softmax((qq[:8] @ kk[:8].T) / math.sqrt(dd), dim=-1).max(-1).values.mean()
    print(f"  d={dd:4d}: std of q·k = {dots.std():6.2f} (≈ sqrt(d) = {math.sqrt(dd):5.2f});"
          f"  mean max softmax weight: raw {peak_raw:.2f}  scaled {peak_scaled:.2f}")
check(abs(torch.randn(4000, 64).mul(torch.randn(4000, 64)).sum(-1).std() - 8) < 1.0,
      "std of a random dot product in d=64 is about sqrt(64) = 8")

# ----------------------------------------------------------------------------
section("3. llm.model.Attention vs torch's reference, with RoPE on both paths")


def reference(attn: Attention, x: torch.Tensor, cos, sin) -> torch.Tensor:
    """Rebuild q/k/v with the module's own projections, apply the same RoPE, call torch SDPA."""
    B, T, _ = x.shape
    h, kv, hd = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = attn.q_proj(x).view(B, T, h, hd).transpose(1, 2)        # (B, h, T, hd)
    k = attn.k_proj(x).view(B, T, kv, hd).transpose(1, 2)       # (B, kv, T, hd)
    v = attn.v_proj(x).view(B, T, kv, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if kv < h:                                                  # GQA: copy each kv head h/kv times
        k = k.repeat_interleave(h // kv, dim=1)
        v = v.repeat_interleave(h // kv, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, h, T, hd)
    return attn.o_proj(o.transpose(1, 2).reshape(B, T, h * hd))


B, T = 2, 16 if args.quick else 64
for n_kv in (6, 2, 1):
    cfg = preset("small", vocab_size=871, n_kv_heads=n_kv)
    attn = Attention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len)
    ours = attn(x, cos[:T], sin[:T])
    ref = reference(attn, x, cos[:T], sin[:T])
    diff = (ours - ref).abs().max().item()
    kind = {6: "MHA", 2: "GQA", 1: "MQA"}[n_kv]
    print(f"  n_heads=6, n_kv_heads={n_kv} ({kind}): output {tuple(ours.shape)}, max |ours - torch| = {diff:.1e}")
    check(diff < 1e-5, f"{kind} matches torch.scaled_dot_product_attention")

# ----------------------------------------------------------------------------
section("4. RoPE: the score depends only on the distance between positions")
hd = 32
cos, sin = rope_tables(hd, 256)                       # (256, 16) each
qv, kv_ = torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd)


def rope_score(i: int, j: int) -> float:
    qi = apply_rope(qv, cos[i:i + 1], sin[i:i + 1])
    kj = apply_rope(kv_, cos[j:j + 1], sin[j:j + 1])
    return float((qi * kj).sum())


pairs = [(5, 2), (15, 12), (105, 102), (200, 197)]
vals = [rope_score(i, j) for i, j in pairs]
print("  same q, same k, positions (i, j) with i - j = 3:", "  ".join(f"({i},{j}) -> {s:.4f}" for (i, j), s in zip(pairs, vals)))
print(f"  but (i, j) = (5, 3), distance 2 -> {rope_score(5, 3):.4f};  (5, 5), distance 0 -> {rope_score(5, 5):.4f}")
check(max(vals) - min(vals) < 1e-4, "q·k is identical for every pair with the same relative offset")
check(abs(apply_rope(qv, cos[7:8], sin[7:8]).norm() - qv.norm()) < 1e-5, "RoPE is a rotation: it does not change vector length")
inv_freq = 1.0 / (10_000 ** (torch.arange(0, hd, 2).float() / hd))
print("  rotation speed per channel pair (radians per position), first 4 and last 2:",
      [round(f, 4) for f in inv_freq[:4].tolist()], "...", [round(f, 5) for f in inv_freq[-2:].tolist()])

# ----------------------------------------------------------------------------
section("5. GQA parameter savings and KV-cache memory")
print("  TinyLM small (d_model=192, 6 query heads, head_dim=32):")
counts = {}
for n_kv in (6, 2, 1):
    a = Attention(preset("small", vocab_size=871, n_kv_heads=n_kv))
    kv_params = a.k_proj.weight.numel() + a.v_proj.weight.numel()
    total = sum(p.numel() for p in a.parameters())
    counts[n_kv] = kv_params
    print(f"    n_kv_heads={n_kv}: k_proj+v_proj = {kv_params:7,} params, whole attention layer = {total:8,}")
check(counts[2] == counts[6] // 3 and counts[1] == counts[6] // 6, "k/v parameters scale with n_kv_heads / n_heads")


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, T, bytes_per_number, batch=1):
    # 2 (K and V) x layers x batch x tokens x kv heads x head_dim x bytes
    return 2 * n_layers * batch * T * n_kv_heads * head_dim * bytes_per_number


cfg = preset("small", vocab_size=871)
formula = kv_cache_bytes(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 256, 4)
m = TinyLM(cfg)
cache = m.new_cache()
with torch.no_grad():
    m(torch.randint(0, 871, (1, 256)), cache=cache)
measured = sum(lc.k.numel() * lc.k.element_size() + lc.v.numel() * lc.v.element_size() for lc in cache.layers)
print(f"  TinyLM small, T=256, fp32: formula {formula:,} bytes = {formula / 1024:.0f} KiB; measured cache {measured:,} bytes")
check(formula == measured, "KV-cache formula matches the bytes actually stored in model.new_cache()")

big = dict(n_layers=80, n_kv_heads=8, head_dim=128)
gib = 2 ** 30
for T_ctx in (8_192, 131_072):
    gqa = kv_cache_bytes(**big, T=T_ctx, bytes_per_number=2) / gib
    mha = kv_cache_bytes(80, 64, 128, T_ctx, 2) / gib
    print(f"  Llama-3-70B-like (80 layers, 8 kv heads x 128, bf16), T={T_ctx:>7,}: GQA {gqa:6.1f} GiB  vs MHA (64 kv heads) {mha:6.1f} GiB")
check(abs(kv_cache_bytes(**big, T=131_072, bytes_per_number=2) / gib - 40.0) < 1e-6, "70B-class KV cache at 128k context = 40 GiB per sequence")
# MLA (DeepSeek-V2/V3): cache one compressed latent (d_c=512) + one rope key (64) per token per layer
mla = 2 * 0 + 61 * 131_072 * (512 + 64) * 2 / gib
print(f"  DeepSeek-V3-like MLA (61 layers, latent 512 + rope 64, bf16), T=131,072: {mla:5.1f} GiB "
      f"(vs {kv_cache_bytes(61, 128, 128, 131_072, 2) / gib:.0f} GiB if it used MHA with 128 heads)")

print("  score matrix size per head, T x T entries:", ", ".join(f"T={T_:>7,}: {T_ * T_:>15,}" for T_ in (256, 4096, 131_072)))

# ----------------------------------------------------------------------------
section("6. Attention maps of the trained TinyLM")
path = BASE_QUICK if args.quick else BASE_FULL
wait_for_checkpoint(path)
model, tok = get_base_model(quick=args.quick)
sentence = "Mia had a red kite. One sunny day Mia took the kite to the park."
ids = torch.tensor([tok.encode(sentence)])                       # (1, T)
toks = [tok.token_str(i) for i in ids[0].tolist()]
print(f"  {len(toks)} tokens:", toks)
maps = model.attention_maps(ids)                                  # list of n_layers x (1, h, T, T)
L, H = len(maps), maps[0].shape[1]
print(f"  {L} layers x {H} heads, each map {tuple(maps[0].shape[-2:])}")
check(all(torch.allclose(mp.sum(-1), torch.ones_like(mp.sum(-1)), atol=1e-4) for mp in maps), "every attention row sums to 1")
check(all(bool((mp.triu(1) == 0).all()) for mp in maps), "trained model is causal: zero weight on future tokens")

# which head looks most at the previous token? at itself? at the first token (attention sink)?
Tn = ids.shape[1]
stats = []
for l, mp in enumerate(maps):
    for h in range(H):
        A = mp[0, h]
        prev = A.diagonal(-1).mean().item()
        self_ = A.diagonal(0).mean().item()
        first = A[1:, 0].mean().item()
        stats.append((l, h, prev, self_, first))
best_prev = max(stats, key=lambda s: s[2])
best_first = max(stats, key=lambda s: s[4])
print(f"  strongest 'previous token' head: layer {best_prev[0]} head {best_prev[1]} (mean weight on t-1 = {best_prev[2]:.2f})")
print(f"  strongest 'first token' head:    layer {best_first[0]} head {best_first[1]} (mean weight on token 0 = {best_first[4]:.2f})")
# where does the second ' kite' look?
second_kite = [i for i, t in enumerate(toks) if t == " kite"][-1]
row = maps[0][0, 0, second_kite]
top = row.topk(3)
print(f"  layer 0 head 0, query = second ' kite' (pos {second_kite}): top keys "
      + ", ".join(f"{toks[j]!r}@{j} {w:.2f}" for w, j in zip(top.values.tolist(), top.indices.tolist())))

fig, axes = plt.subplots(L, H, figsize=(2.1 * H, 2.0 * L), squeeze=False)
for l in range(L):
    for h in range(H):
        ax = axes[l, h]
        ax.imshow(maps[l][0, h].numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if h == 0:
            ax.set_ylabel(f"layer {l}", fontsize=8)
        if l == 0:
            ax.set_title(f"head {h}", fontsize=8)
fig.suptitle(f"attention maps, TinyLM {'nano' if args.quick else 'small'}: rows = query token, cols = key token\n\"{sentence}\"", fontsize=9)
fig.tight_layout()
savefig(fig, "lab05_attention_maps.png")

# a bigger view of the strongest 'previous token' head, with token labels
lp, hp = best_prev[0], best_prev[1]
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.imshow(maps[lp][0, hp].numpy(), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(Tn), toks, rotation=90, fontsize=7); ax.set_yticks(range(Tn), toks, fontsize=7)
ax.set_title(f"layer {lp} head {hp}: the 'previous token' head", fontsize=10)
savefig(fig, "lab05_attention_head.png")

# ----------------------------------------------------------------------------
section("7. Sliding-window attention: the mask alone changes the model")
print("  causal_mask(T=6, S=6, window=None):\n", causal_mask(6, 6, 0, None, "cpu").int().numpy())
print("  causal_mask(T=6, S=6, window=4):\n", causal_mask(6, 6, 0, 4, "cpu").int().numpy())
swa = TinyLM(preset("nano", vocab_size=871, sliding_window=4))
maps_swa = swa.attention_maps(torch.randint(0, 871, (1, 10)))
beyond = max(mp[0].tril(-4).abs().max().item() for mp in maps_swa)
print(f"  with sliding_window=4, max weight on keys more than 3 back = {beyond:.1e}")
check(beyond == 0.0, "sliding window: zero weight beyond the window")
print(f"  KV cache needed per layer with a window of 4: 4 tokens, however long the text (vs T={10} here)")

done()
