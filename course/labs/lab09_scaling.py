"""Lab 09: fit a tiny scaling law.

Train three TinyLM sizes on the same token budget, record the final validation loss
against parameter count N, fit  L(N) = E + A / N^alpha  with numpy, and look at how
loss falls with tokens for one size. Then convert real compute budgets into
"what model and how many tokens" with the 6·N·D rule.

    python3 labs/lab09_scaling.py            # --quick (default): ~1 minute
    python3 labs/lab09_scaling.py --full     # a few minutes
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import time

import numpy as np
import torch

from llm.config import TinyLMConfig
from llm.model import TinyLM
from llm.pipeline import get_tokenizer, get_tokens
from llm.train import TrainConfig, train, estimate_loss

args = setup("Lab 09: fit a tiny scaling law")
torch.manual_seed(args.seed)

# ------------------------------------------------------------ 1. the budget
section("1. one token budget, three model sizes")
tok = get_tokenizer()
train_tokens, val_tokens = get_tokens(tok)
V = tok.vocab_size
if args.quick:
    STEPS, BS, SL, WARMUP, LR = 60, 16, 64, 6, 3e-3
else:
    STEPS, BS, SL, WARMUP, LR = 300, 32, 128, 30, 2e-3
D = STEPS * BS * SL                                   # tokens every model will see
print(f"train stream {len(train_tokens):,} tokens | budget D = {STEPS} steps × {BS} × {SL} = {D:,} tokens per model")

SIZES = [  # name, d_model, n_layers, n_heads
    ("xs", 64, 2, 2),
    ("s", 96, 3, 3),
    ("m", 160, 4, 4),
]
results = []
for name, d, L, h in SIZES:
    cfg = TinyLMConfig(vocab_size=V, d_model=d, n_layers=L, n_heads=h, n_kv_heads=1, max_seq_len=SL)
    model = TinyLM(cfg)
    N = model.num_params()                            # non-embedding parameters (Kaplan's convention)
    N_all = model.num_params(non_embedding=False)
    tc = TrainConfig(steps=STEPS, batch_size=BS, seq_len=SL, optimizer="adamw", lr=LR, warmup_steps=WARMUP,
                     schedule="cosine", eval_every=max(10, STEPS // 5), eval_batches=8, log_every=max(10, STEPS // 6),
                     seed=args.seed)
    t0 = time.perf_counter()
    hist = train(model, train_tokens, val_tokens, tc, verbose=False)
    dt = time.perf_counter() - t0
    val = estimate_loss(model, val_tokens, BS, SL, n_batches=20)
    flops = model.flops_per_token() * D
    results.append(dict(name=name, d=d, L=L, N=N, N_all=N_all, val=val, hist=hist, secs=dt, flops=flops))
    print(f"  {name}: d={d:>3} layers={L}  N={N:>9,} (+{N_all - N:,} embedding)  "
          f"val loss {val:.3f}  {dt:5.1f}s  {D / dt:,.0f} tok/s  C = {flops:.2e} FLOPs")

# The same validation windows were scored every STEPS//5 steps, so we also get L(N, D) for free.
print("\nval loss L(N, D) at intermediate token counts (rows: D, columns: model size):")
print(f"{'D (tokens)':>12}" + "".join(f"{r['name']:>9}" for r in results) + "   (biggest − smallest)")
for i, st in enumerate(results[0]["hist"].val_step):
    row = [r["hist"].val_loss[i] for r in results]
    print(f"{(st + 1) * BS * SL:>12,}" + "".join(f"{v:9.3f}" for v in row) + f"   {row[-1] - row[0]:+.3f}")
print("as D grows every size approaches the same floor: the irreducible loss E of Storyland")

Ns = np.array([r["N"] for r in results], dtype=float)
Ls = np.array([r["val"] for r in results])
check(Ls[0] > Ls[1] > Ls[2], "bigger model -> lower loss at a fixed token budget")
check(all(r["val"] < math.log(V) - 1.0 for r in results), "every size learned (loss well below uniform ln V)")

# ---------------------------------------------------------- 2. fit L(N)
section("2. fit L(N) = E + A / N^alpha")


def fit_power_law(N, L, alphas=np.linspace(0.05, 2.0, 400)):
    """Grid over alpha; for each alpha (E, A) is a linear least-squares problem."""
    best = None
    for a in alphas:
        X = np.stack([np.ones_like(N), N ** (-a)], axis=1)      # (n_points, 2)
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        E, A = coef
        if E < 0 or A <= 0:
            continue
        resid = float(((X @ coef - L) ** 2).sum())
        if best is None or resid < best[0]:
            best = (resid, E, A, a)
    return best[1:]


E, A, alpha = fit_power_law(Ns, Ls)
pred = E + A * Ns ** (-alpha)
print(f"fit: L(N) = {E:.3f} + {A:.3g} / N^{alpha:.3f}")
print(f"     E is the loss no amount of parameters removes (at this D); alpha is how fast the rest shrinks")
for r, p in zip(results, pred):
    print(f"     N={r['N']:>9,}  measured {r['val']:.3f}  fitted {p:.3f}")
check(np.abs(pred - Ls).max() < 0.05, "fit passes within 0.05 of every point (3 points, 3 parameters: an exact fit)")
N10 = Ns[-1] * 10
print(f"extrapolation to N = {N10:,.0f} (10× the largest): predicted loss {E + A * N10 ** (-alpha):.3f}")
print("(Three points cannot validate a three-parameter law — add a fourth size in the exercises before trusting alpha.)")

fig, axes = plt().subplots(1, 2, figsize=(10, 4))
ax = axes[0]
grid = np.logspace(np.log10(Ns[0] / 2), np.log10(Ns[-1] * 20), 100)
ax.plot(grid, E + A * grid ** (-alpha), color="#64748b", lw=1.5, label=f"fit: {E:.2f} + {A:.2g}·N^-{alpha:.2f}")
ax.scatter(Ns, Ls, color="#2563eb", zorder=3, label="measured (final val loss)")
for r in results:
    ax.annotate(r["name"], (r["N"], r["val"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.axhline(E, color="#dc2626", ls=":", lw=1, label=f"E = {E:.2f}")
ax.set_xscale("log")
ax.set_xlabel("non-embedding parameters N")
ax.set_ylabel(f"val loss after D = {D:,} tokens")
ax.set_title("Lab 09: loss vs model size")
ax.legend(fontsize=8)

# ------------------------------------------------ 3. the loss curve for one size
section("3. loss vs tokens for the largest model")
r = results[-1]
h = r["hist"]
toks = np.array(h.step) * BS * SL + BS * SL
ax = axes[1]
ax.plot(toks, h.train_loss, color="#2563eb", lw=1.2, label="train loss (one batch)")
ax.plot(np.array(h.val_step) * BS * SL + BS * SL, h.val_loss, "o-", color="#f59e0b", label="val loss")
ax.set_xscale("log")
ax.set_xlabel("tokens seen")
ax.set_ylabel("loss")
ax.set_title(f"Lab 09: loss vs tokens, model '{r['name']}' (N={r['N']:,})")
ax.legend(fontsize=8)
savefig(fig, "lab09_scaling.png")
first, last = h.train_loss[0], h.train_loss[-1]
print(f"model '{r['name']}': train loss {first:.2f} at {toks[0]:,} tokens -> {last:.2f} at {toks[-1]:,} tokens")
print("shape: steep at first (easy statistics: which tokens are common), then a long slow tail")
check(last < first - 1.0, "loss curve dropped by more than 1 nat over the run")

# ------------------------------------------------- 4. FLOPs: 6ND vs the model
section("4. compute used: C ≈ 6·N·D")
for r in results:
    six_nd = 6 * r["N_all"] * D
    print(f"  {r['name']}: 6·N·D = {six_nd:.2e}   model.flops_per_token()·D = {r['flops']:.2e}   "
          f"(attention adds {100 * (r['flops'] / six_nd - 1):.0f}%)   achieved {r['flops'] / r['secs'] / 1e9:.1f} GFLOP/s")
total_flops = sum(r["flops"] for r in results)
print(f"  this lab used {total_flops:.2e} FLOPs in total — a frontier run is ~10^25–10^26, i.e. 10^12× more")

# ------------------------------------------------ 5. what a budget buys
section("5. what a budget buys (Chinchilla vs 2026 practice)")
H100_BF16 = 990e12          # dense bf16 peak, FLOP/s
MFU = 0.40                  # a good large-run utilisation
PRICE_PER_GPU_HOUR = 3.0    # USD, cloud H100 in 2025–26 (order of magnitude)
flops_per_gpu_hour = H100_BF16 * MFU * 3600
print(f"one H100-hour at {MFU:.0%} MFU = {flops_per_gpu_hour:.2e} FLOPs;  8×H100 for 4 h (nanochat) = {8 * 4 * flops_per_gpu_hour:.2e}")
print(f"\n{'budget':<22}{'GPU-hours':>10}{'FLOPs':>10}{'N (20 tok/param)':>18}{'D':>9}{'N (200 tok/param)':>19}{'D':>9}")


def fmt(x):
    for unit, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= unit:
            return f"{x / unit:.3g}{s}"
    return f"{x:.0f}"


for label, dollars in (("$100 (nanochat)", 100), ("$10k", 10_000), ("$10M", 10_000_000)):
    gpu_h = dollars / PRICE_PER_GPU_HOUR
    C = gpu_h * flops_per_gpu_hour
    N20 = math.sqrt(C / (6 * 20)); D20 = 20 * N20              # C = 6·N·D with D = 20·N
    N200 = math.sqrt(C / (6 * 200)); D200 = 200 * N200
    print(f"{label:<22}{gpu_h:>10,.0f}{C:>10.1e}{fmt(N20):>18}{fmt(D20):>9}{fmt(N200):>19}{fmt(D200):>9}")
nano_C = 8 * 4 * flops_per_gpu_hour
N_nano = math.sqrt(nano_C / 120)
print(f"\nsanity check: nanochat's 8×H100×4h -> Chinchilla-optimal N ≈ {N_nano / 1e6:.0f}M, D ≈ {20 * N_nano / 1e9:.1f}B tokens;")
print("nanochat actually trains a ~560M-parameter model on ~11B tokens (d20 preset) — the rule lands in the right place.")
check(4e8 < N_nano < 8e8, "6ND + 20 tokens/param puts the $100 run near nanochat's real 560M model")

done()
