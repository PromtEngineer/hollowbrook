"""Lab 7: from logits to text — sampling knobs, the KV cache, speed, cost.

    python3 labs/lab07_generate.py            # --quick: nano base model
    python3 labs/lab07_generate.py --full     # small base model, longer benchmarks, speculative decoding
"""
from _common import setup, check, banner, section, savefig, done, plt
import math
import os
import time
import torch
import torch.nn.functional as F
from llm.config import TinyLMConfig
from llm.model import TinyLM
from llm.pipeline import get_base_model, run_path
from llm.generate import generate, generate_ids, sample_next, apply_repetition_penalty, benchmark_decode
from llm.chat import render, END

args = setup("Lab 7: inference")
model, tok = get_base_model(quick=args.quick)
cfg = model.cfg
PROMPT = "At the park, Mia met"
ids = torch.tensor([tok.encode(PROMPT)])                       # (1, T0)
print(f"model: {'nano' if args.quick else 'small'} base, {model.num_params():,} non-embedding params; "
      f"prompt {PROMPT!r} = {ids.shape[1]} tokens")

# ----------------------------------------------------------------------------- 1
section("1. logits -> softmax -> a distribution over the next token")
with torch.no_grad():
    logits, _ = model(ids)                                     # (1, T0, V)
last = logits[0, -1]                                           # (V,)
probs = F.softmax(last, dim=-1)
entropy_bits = -(probs * probs.clamp_min(1e-12).log2()).sum().item()
top = probs.topk(10)
print(f"logits shape {tuple(logits.shape)}; last-position logits range [{last.min():.2f}, {last.max():.2f}]")
print(f"{'token':>12s} {'logit':>8s} {'prob':>7s}")
for p, i in zip(top.values, top.indices):
    print(f"{tok.token_str(int(i))!r:>12s} {last[i]:8.2f} {p:7.3f}")
print(f"entropy of this distribution: {entropy_bits:.2f} bits  (uniform over V={cfg.vocab_size} would be {math.log2(cfg.vocab_size):.2f})")
check(abs(probs.sum().item() - 1) < 1e-5, "softmax output sums to 1")
check(int(sample_next(last[None], temperature=0.0)) == int(top.indices[0]), "temperature 0 == argmax")

# ----------------------------------------------------------------------------- 2
section("2. The sampling knobs on the SAME prompt and seed")
settings = [
    ("greedy (T=0)",        dict(temperature=0.0)),
    ("T=0.5",               dict(temperature=0.5)),
    ("T=1.0",               dict(temperature=1.0)),
    ("T=1.5",               dict(temperature=1.5)),
    ("T=1.0 top-k=5",       dict(temperature=1.0, top_k=5)),
    ("T=1.0 top-p=0.9",     dict(temperature=1.0, top_p=0.9)),
    ("T=1.0 min-p=0.1",     dict(temperature=1.0, min_p=0.1)),
    ("T=1.0 rep-pen=1.5",   dict(temperature=1.0, repetition_penalty=1.5)),
]
outs = {}
for name, kw in settings:
    outs[name] = generate(model, tok, PROMPT, max_new_tokens=24, seed=args.seed, **kw)
    print(f"{name:20s} | {outs[name]!r}")
g2 = generate(model, tok, PROMPT, max_new_tokens=24, temperature=0.0, seed=args.seed + 1)
check(g2 == outs["greedy (T=0)"], "greedy decoding is deterministic (seed does not matter)")
pen = apply_repetition_penalty(last[None].clone(), ids, 1.5)[0]
seen = ids[0].unique()
check(bool((pen[seen] <= last[seen]).all()), "repetition penalty never raises the logit of a token already seen")
check(len(set(outs.values())) >= 4, "different knobs give different texts from one seed")

# ----------------------------------------------------------------------------- 3
section("3. Temperature sweep: entropy of the next-token distribution")
temps = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
ent = []
for t in temps:
    p = F.softmax(logits[0] / t, dim=-1)                       # every prompt position, (T0, V)
    ent.append(-(p * p.clamp_min(1e-12).log2()).sum(-1).mean().item())
    print(f"T={t:<5} mean entropy {ent[-1]:6.2f} bits   top-1 prob at last position {F.softmax(last / t, -1).max():.3f}")
fig, ax = plt().subplots(figsize=(5.5, 3.2))
ax.plot(temps, ent, marker="o"); ax.axhline(math.log2(cfg.vocab_size), ls="--", c="gray", label="uniform (log2 V)")
ax.set_xscale("log"); ax.set_xlabel("temperature"); ax.set_ylabel("entropy (bits)"); ax.legend()
ax.set_title("higher temperature -> flatter distribution")
savefig(fig, "lab07_temperature_entropy.png")
check(all(a < b for a, b in zip(ent, ent[1:])), "entropy increases monotonically with temperature")

# ----------------------------------------------------------------------------- 4
section("4. KV cache: same numbers, fewer FLOPs")
n_new = 32 if args.quick else 96
t0 = time.perf_counter()
with_cache = generate_ids(model, ids, n_new, temperature=0.0, use_cache=True)
t_cache = time.perf_counter() - t0
t0 = time.perf_counter()
no_cache = generate_ids(model, ids, n_new, temperature=0.0, use_cache=False)
t_nocache = time.perf_counter() - t0
check(torch.equal(with_cache, no_cache), f"greedy ids identical with and without cache ({n_new} tokens)")
# logits: full forward vs prefill + one-token steps
seq = with_cache                                               # (1, T0+n_new)
with torch.no_grad():
    full_logits, _ = model(seq)
    cache = model.new_cache()
    inc = [model(seq[:, :4], cache=cache)[0]]                  # prefill 4 tokens
    for t in range(4, seq.shape[1]):
        inc.append(model(seq[:, t:t + 1], cache=cache)[0])     # decode one token at a time
    inc_logits = torch.cat(inc, dim=1)
print(f"cache.pos after prefill+decode = {cache.pos} (= sequence length {seq.shape[1]})")
print(f"max |logits_full - logits_incremental| = {(full_logits - inc_logits).abs().max().item():.2e}")
check(torch.allclose(full_logits, inc_logits, atol=1e-4), "cached incremental logits == full-sequence logits (allclose)")
print(f"greedy {n_new} tokens: cache {n_new/t_cache:6.1f} tok/s   no cache {n_new/t_nocache:6.1f} tok/s   "
      f"speed-up {t_nocache/t_cache:.2f}x")
bench = benchmark_decode(model, tok, PROMPT, max_new_tokens=n_new)
print(f"benchmark_decode({n_new} tokens): {bench}")
check(bench["cache"] > bench["no_cache"], "the KV cache is faster than recomputing")

# ----------------------------------------------------------------------------- 5
section("5. How big is the cache?")
def kv_bytes(c: TinyLMConfig, n_tokens: int, bytes_per: int = 4) -> int:
    return 2 * c.n_layers * c.n_kv_heads * c.head_dim * bytes_per * n_tokens

actual = sum(lc.k.numel() * lc.k.element_size() + lc.v.numel() * lc.v.element_size() for lc in cache.layers)
print(f"TinyLM cache after {cache.pos} tokens: formula {kv_bytes(cfg, cache.pos):,} B, measured {actual:,} B "
      f"({kv_bytes(cfg, 1)} B per token; params take {model.num_params(False)*4:,} B)")
check(kv_bytes(cfg, cache.pos) == actual, "2 * L * kv_heads * head_dim * bytes * T matches the tensors")
big = {  # name: (layers, kv_heads, head_dim, params)
    "Llama-3-8B-like (GQA, 8 kv)":   (32, 8, 128, 8e9),
    "same but MHA (32 kv heads)":     (32, 32, 128, 8e9),
    "Llama-3-70B-like (GQA, 8 kv)":  (80, 8, 128, 70e9),
}
print(f"{'model':32s} {'B/token':>9s} {'8k ctx':>9s} {'128k ctx':>9s} {'weights':>9s}")
for name, (Lb, kvb, hdb, Np) in big.items():
    per = 2 * Lb * kvb * hdb * 2                                # bf16
    print(f"{name:32s} {per/1024:7.0f} KB {per*8192/1e9:7.1f} GB {per*131072/1e9:7.1f} GB {Np*2/1e9:7.0f} GB")

# ----------------------------------------------------------------------------- 6
section("6. Why decode is memory-bound: arithmetic intensity vs the H100 ridge point")
PEAK, BW = 1000e12, 3.35e12                                    # H100 SXM: ~1000 TFLOP/s bf16 dense, 3.35 TB/s
ridge = PEAK / BW
print(f"H100 ridge point = {PEAK:.0e} / {BW:.2e} = {ridge:.0f} FLOP per byte")
Np = 70e9
for Bsz in (1, 8, 64, 512):
    flops = 2 * Np * Bsz                                       # one decode step for the batch
    byts = 2 * Np                                              # read every bf16 weight once
    inten = flops / byts
    t_mem, t_comp = byts / BW, flops / PEAK
    print(f"70B model, batch {Bsz:4d}: intensity {inten:5.0f} FLOP/B  -> "
          f"{'memory' if inten < ridge else 'compute'}-bound; step >= {max(t_mem, t_comp)*1e3:5.1f} ms "
          f"-> {Bsz/max(t_mem, t_comp):8,.0f} tok/s aggregate")
check(2 * Np * 1 / (2 * Np) < ridge, "batch-1 decode intensity (1 FLOP/byte) is far below the ridge point")

# ----------------------------------------------------------------------------- 7
section("7. Speculative decoding: nano drafts, small verifies (greedy acceptance rule)")
def speculative_greedy(target: TinyLM, draft: TinyLM, idx: torch.Tensor, n_new: int, k: int = 4):
    """Greedy speculative decoding without caches (clarity over speed). Returns (ids, accepted, drafted)."""
    accepted = drafted = 0
    T0 = idx.shape[1]
    with torch.no_grad():
        while idx.shape[1] < T0 + n_new:
            d = idx
            for _ in range(k):                                 # 1. draft k tokens greedily with the small model
                d = torch.cat([d, draft(d)[0][:, -1].argmax(-1, keepdim=True)], dim=1)
            proposed = d[0, idx.shape[1]:]                     # (k,)
            tgt = target(d)[0][0, idx.shape[1] - 1:].argmax(-1) # 2. one target pass scores all k (+1 bonus)
            n_ok = 0
            while n_ok < k and proposed[n_ok] == tgt[n_ok]:    # 3. accept the longest matching prefix
                n_ok += 1
            new = torch.cat([proposed[:n_ok], tgt[n_ok:n_ok + 1]])   # + the target's own next token
            idx = torch.cat([idx, new[None]], dim=1)
            accepted += n_ok; drafted += k
    return idx[:, :T0 + n_new], accepted, drafted

nano_path, small_path = run_path("base_nano.pt"), run_path("base_small.pt")
if os.path.exists(nano_path) and os.path.exists(small_path):
    draft, target = TinyLM.load(nano_path), TinyLM.load(small_path)
    n_spec = 40
    spec_ids, acc, drafted = speculative_greedy(target, draft, ids, n_spec, k=4)
    plain = generate_ids(target, ids, n_spec, temperature=0.0)
    print(f"draft nano ({draft.num_params():,}) -> target small ({target.num_params():,})")
    print(f"acceptance rate {acc}/{drafted} = {acc/drafted:.0%}; target calls {drafted//4} instead of {n_spec}")
    print("speculative:", repr(tok.decode(spec_ids[0, ids.shape[1]:].tolist())))
    print("plain greedy:", repr(tok.decode(plain[0, ids.shape[1]:].tolist())))
    check(torch.equal(spec_ids, plain), "speculative output == target's plain greedy output (lossless)")
else:
    print("skipped: needs both runs/base_nano.pt and runs/base_small.pt (run lab06 --quick and --full)")

# ----------------------------------------------------------------------------- 8
section("8. Stop tokens and the chat format (preview of Chapter 14)")
chat = render([{"role": "user", "content": "What is 2 + 3?"}])
chat_ids = tok.encode(chat)
print(repr(chat))
print("tokens:", [tok.token_str(i) for i in chat_ids][:12], "...")
print(f"stop token {END!r} has id {tok.special_tokens[END]}; base model generation with stop={END!r}:")
out = generate(model, tok, chat, max_new_tokens=30, temperature=0.0, stop=(END,))
print(repr(out))
check(tok.special_tokens[END] in chat_ids, "the chat template is a single special token in the vocabulary")
print("(the base model emits <|eos|>, the pretraining document separator, but never <|end|>; SFT in Chapter 15 teaches it to end a turn)")

# ----------------------------------------------------------------------------- 9
section("9. The cost model: $ per million tokens")
def usd_per_million(gpu_usd_per_hour: float, tokens_per_sec: float) -> float:
    return gpu_usd_per_hour / 3600 / tokens_per_sec * 1e6
for name, price, tps in [("70B, batch 1 (memory-bound)", 3.0, 24), ("70B, batch 64", 3.0, 1500),
                          ("70B, batch 512 (near compute-bound)", 3.0, 7000)]:
    print(f"{name:38s} ${price}/h at {tps:5,} tok/s -> ${usd_per_million(price, tps):7.2f} per M tokens")
done()
