"""Lab 0: is your machine ready? A 30-second smoke test of the whole TinyLM stack.

    python3 labs/lab00_setup.py            # --quick: nano model (trains ~1 min the first time, then cached)
    python3 labs/lab00_setup.py --full     # the 'small' course model (trains ~5-10 min the first time)

Checks Python / PyTorch / CPU, the tokenizer round trip, that a base model can be
loaded (or trained), and that it generates text. Nothing here needs a GPU.
"""
from _common import setup, check, banner, section, savefig, done, plt

import os
import platform
import sys
import time

args = setup("Lab 00: setup and smoke test")

# --------------------------------------------------------------- environment
section("environment")
print(f"python  {platform.python_version()}  ({sys.executable})")
check(sys.version_info >= (3, 10), "Python 3.10 or newer")
try:
    import torch
    import numpy
    import matplotlib
    print(f"numpy {numpy.__version__} | matplotlib {matplotlib.__version__}")
    check(True, "torch, numpy and matplotlib import")
except ImportError as e:  # pragma: no cover
    check(False, f"missing package: {e} -> pip install torch numpy matplotlib regex")
    done()
try:
    import regex  # noqa: F401
    print("regex   installed (the tokenizer uses the full GPT-4-style pattern)")
except ImportError:  # pragma: no cover
    print("regex   NOT installed: the tokenizer falls back to a simpler ASCII pattern (pip install regex)")

from llm.pipeline import device_summary, get_corpus, get_tokenizer, get_base_model, RUNS_DIR
from llm.generate import generate

print(device_summary())
print(f"cpu     {platform.machine()} | {os.cpu_count()} logical cores | torch using {torch.get_num_threads()} threads")
check(torch.get_num_threads() >= 1, "PyTorch can run on this CPU")
x = torch.randn(256, 256)
t0 = time.perf_counter()
for _ in range(20):
    y = x @ x
dt = (time.perf_counter() - t0) / 20
print(f"a 256x256 matmul takes {dt * 1e6:.0f} µs  ({2 * 256 ** 3 / dt / 1e9:.1f} GFLOP/s)")
print(f"artifacts folder: {RUNS_DIR}")

# ---------------------------------------------------------------- corpus
section("corpus: Storyland")
docs = get_corpus()
n_chars = sum(len(d["text"]) for d in docs)
print(f"{len(docs)} documents, {n_chars:,} characters; sources: "
      + ", ".join(f"{s}={sum(d['source'] == s for d in docs)}" for s in ("stories", "math")))
print("doc 0:", docs[0]["text"][:120].replace("\n", " ") + "...")
check(len(docs) == 6000, "6000 Storyland documents generated deterministically (seed 0)")

# ------------------------------------------------------------- tokenizer
section("tokenizer: text -> integers -> text")
tok = get_tokenizer(docs)
s = "Mia had a red kite. What is 12 + 7?\nAnswer: 12 + 7 = 19."
ids = tok.encode(s)
print(f"vocab size {tok.vocab_size} | {len(s)} chars -> {len(ids)} tokens")
print("tokens:", "|".join(tok.token_str(i) for i in ids[:16]) + "|...")
check(tok.decode(ids) == s, "tokenizer round trip: decode(encode(s)) == s")
check(all(0 <= i < tok.vocab_size for i in ids), "every id is inside the vocabulary")

# ------------------------------------------------------------ base model
section(f"base model ({'nano' if args.quick else 'small'})")
ckpt = os.path.join(RUNS_DIR, "base_nano.pt" if args.quick else "base_small.pt")
if os.path.exists(ckpt):
    print(f"loading cached checkpoint {os.path.relpath(ckpt)}")
else:
    print(f"no checkpoint at {os.path.relpath(ckpt)} -> pretraining one now (this is the slow part, once)")
t0 = time.perf_counter()
model, tok = get_base_model(quick=args.quick)
print(f"ready in {time.perf_counter() - t0:.1f}s | {model.num_params():,} non-embedding params, "
      f"{model.num_params(non_embedding=False):,} total | config: d_model={model.cfg.d_model} "
      f"layers={model.cfg.n_layers} heads={model.cfg.n_heads} kv_heads={model.cfg.n_kv_heads}")
check(os.path.exists(ckpt), "checkpoint saved to runs/")

# one forward pass: the model's whole job is to produce next-token probabilities
x = torch.tensor([tok.encode("Mia had a")])                  # (B=1, T) token ids
with torch.no_grad():
    logits, _ = model(x)                                       # (B, T, V) scores for every next token
probs = torch.softmax(logits[0, -1], dim=-1)                   # (V,) probabilities for the token after 'a'
top = torch.topk(probs, 5)
print(f"logits shape {tuple(logits.shape)} = (batch, tokens, vocab)")
print("P(next | 'Mia had a'):", ", ".join(f"{tok.token_str(i)!r}={p:.3f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))
check(abs(float(probs.sum()) - 1.0) < 1e-4, "next-token probabilities sum to 1")
check(tuple(logits.shape) == (1, x.shape[1], tok.vocab_size), "logits have shape (B, T, V)")

# ------------------------------------------------------------- generation
section("generate")
t0 = time.perf_counter()
out = generate(model, tok, "Mia had a", max_new_tokens=40, temperature=0.8, top_k=40, seed=args.seed)
dt = time.perf_counter() - t0
n_new = len(tok.encode(out))
print(f"prompt: 'Mia had a'\ncompletion: {out!r}")
print(f"{n_new} tokens in {dt:.2f}s = {n_new / dt:.0f} tokens/s")
check(len(out) > 0, "model generated text")
out2 = generate(model, tok, "What is 12 + 7?\nAnswer:", max_new_tokens=12, temperature=0.0)
print(f"prompt: 'What is 12 + 7?\\nAnswer:' -> {out2!r}   (a base model may or may not get this right yet)")

# --------------------------------------------------------------- figure
p = plt()
fig, ax = p.subplots(figsize=(7, 3))
top = torch.topk(probs, 12)
ax.bar([tok.token_str(i) for i in top.indices.tolist()], top.values.tolist(), color="#2563eb")
ax.set_title("TinyLM: P(next token | 'Mia had a')")
ax.set_ylabel("probability")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
savefig(fig, "lab00_next_token.png")
done()
