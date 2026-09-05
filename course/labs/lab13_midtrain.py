"""Lab 13: mid-training on a CPU — anneal on a re-weighted mix, then extend the context.

(a) Anneal: continue training the pretrained base for a few hundred steps on a mix that
    up-weights math 4:1, with a learning rate that decays linearly to zero. We measure
    the loss on held-out MATH-only and STORIES-only token streams before and after.
(b) Long-context extension: raise the RoPE base frequency (ABF, as Llama 3 does), rebuild
    the tables for 512 positions, and fine-tune briefly at seq_len 512. We measure the loss
    per position before and after.

The result is saved as runs/lab13_annealed.pt: the course's "mid-trained" checkpoint.

Run:  python3 labs/lab13_midtrain.py            (quick)
      python3 labs/lab13_midtrain.py --full
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import time

import torch
import torch.nn.functional as F

from llm.data import make_corpus, mix_sources, tokenize_and_pack
from llm.model import TinyLM
from llm.optim import build_optimizer, lr_at, set_lr
from llm.train import get_batch, estimate_loss
from llm.pipeline import get_corpus, get_tokenizer, get_base_model, run_path, TOKENIZER_PATH

args = setup("Lab 13: mid-training — anneal on a math-heavy mix, then extend the context")

ANNEAL_STEPS = 60 if args.quick else 250
ANNEAL_B = 8 if args.quick else 32
T_SHORT = 128
LONG_T = 512
LONG_STEPS = 20 if args.quick else 60
LONG_B = 2 if args.quick else 8
EVAL_BATCHES = 6 if args.quick else 10
NEW_THETA = 50_000.0

tok = get_tokenizer()
model, _ = get_base_model(quick=args.quick, verbose=True)
model.train()
print(f"base model: {model.num_params():,} params, max_seq_len {model.cfg.max_seq_len}, rope_theta {model.cfg.rope_theta:.0f}")

# ------------------------------------------------------------------ data
section("data: the anneal mix and two held-out streams")
train_docs = get_corpus()                                    # what the base was pretrained on (seed 0)
eval_docs = make_corpus(2000, seed=12345)                    # brand-new documents, never seen
math_eval = tokenize_and_pack([d for d in eval_docs if d["source"] == "math"], tok)
story_eval = tokenize_and_pack([d for d in eval_docs if d["source"] == "stories"], tok)
mix_docs = mix_sources(train_docs, {"stories": 1, "math": 4}, n_out=4000, seed=args.seed)
mix_tokens = tokenize_and_pack(mix_docs, tok)
n_math_docs = sum(d["source"] == "math" for d in mix_docs)
math_tok = sum(len(tok.encode(d["text"])) for d in mix_docs if d["source"] == "math")
print(f"   pretraining corpus: {sum(d['source'] == 'math' for d in train_docs)} math / "
      f"{sum(d['source'] == 'stories' for d in train_docs)} story docs")
print(f"   anneal mix (weights stories:1, math:4): {n_math_docs}/{len(mix_docs)} docs are math "
      f"= {100 * n_math_docs / len(mix_docs):.0f}% of docs but {100 * math_tok / len(mix_tokens):.0f}% of tokens "
      f"(math docs are short)")
print(f"   held-out streams: math {len(math_eval):,} tokens, stories {len(story_eval):,} tokens")


def eval_both(m: TinyLM) -> tuple[float, float]:
    m.eval()
    a = estimate_loss(m, math_eval, 16, T_SHORT, EVAL_BATCHES)
    b = estimate_loss(m, story_eval, 16, T_SHORT, EVAL_BATCHES)
    m.train()
    return a, b


# ------------------------------------------------------------------ (a) anneal
section(f"(a) anneal: {ANNEAL_STEPS} steps on the math-heavy mix, LR decaying linearly to 0")
math_before, story_before = eval_both(model)
print(f"   BEFORE: math loss {math_before:.3f} (ppl {math.exp(math_before):.1f}) | "
      f"stories loss {story_before:.3f} (ppl {math.exp(story_before):.1f})")

optimizers = build_optimizer(model, "adamw", lr=3e-4, weight_decay=0.1)   # lower LR than pretraining's 1e-3
g = torch.Generator().manual_seed(args.seed)
anneal_curve, lr_curve, t0 = [], [], time.perf_counter()
for step in range(ANNEAL_STEPS):
    scale = lr_at(step, ANNEAL_STEPS, 1.0, warmup_steps=0, kind="wsd", decay_frac=1.0)  # 1 -> 0 linearly
    set_lr(optimizers, scale)
    x, y = get_batch(mix_tokens, ANNEAL_B, T_SHORT, g)
    _, loss = model(x, y)
    for o in optimizers:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in optimizers:
        o.step()
    anneal_curve.append(loss.item()); lr_curve.append(scale)
    if step % max(1, ANNEAL_STEPS // 5) == 0 or step == ANNEAL_STEPS - 1:
        print(f"   step {step:4d} | mix loss {loss.item():.4f} | lr x{scale:.2f}")
anneal_wall = time.perf_counter() - t0
math_after, story_after = eval_both(model)
print(f"   AFTER:  math loss {math_after:.3f} (ppl {math.exp(math_after):.1f}) | "
      f"stories loss {story_after:.3f} (ppl {math.exp(story_after):.1f})   [{anneal_wall:.0f}s]")
print(f"   change: math {math_after - math_before:+.3f} | stories {story_after - story_before:+.3f}")
check(math_after < math_before, "annealing on a math-heavy mix lowers the held-out MATH loss")
check(story_after - story_before < 0.15,
      f"...without a large regression on stories (change {story_after - story_before:+.3f}; replay keeps it small)")


# ------------------------------------------------------------------ (b) long-context extension
section(f"(b) long context: {model.cfg.max_seq_len} -> {LONG_T} positions with RoPE theta {NEW_THETA:.0f}")
long_eval = torch.cat([story_eval, math_eval])


@torch.no_grad()
def loss_per_position(m: TinyLM, n_windows: int = 8) -> torch.Tensor:
    """Mean cross-entropy at each of the LONG_T positions over fixed held-out windows."""
    m.eval()
    g = torch.Generator().manual_seed(99)
    acc = torch.zeros(LONG_T)
    for _ in range(n_windows):
        x, y = get_batch(long_eval, 1, LONG_T, g)
        logits, _ = m(x)                                                     # (1, T, V)
        ce = F.cross_entropy(logits[0].float(), y[0], reduction="none")      # (T,)
        acc += ce
    m.train()
    return acc / n_windows


def summarise(curve: torch.Tensor, label: str) -> tuple[float, float]:
    short, long = curve[:T_SHORT].mean().item(), curve[T_SHORT:].mean().item()
    bins = [f"{curve[i:i + 64].mean().item():.2f}" for i in range(0, LONG_T, 64)]
    print(f"   {label:<34} pos 0-{T_SHORT}: {short:.3f} | pos {T_SHORT}-{LONG_T}: {long:.3f} | per-64 bins {bins}")
    return short, long


orig_theta = model.cfg.rope_theta
model.extend_context(LONG_T)                                  # same theta, just longer tables
curve_naive = loss_per_position(model)
s_naive, l_naive = summarise(curve_naive, f"tables extended, theta={orig_theta:.0f}")
model.extend_context(LONG_T, theta=NEW_THETA)                 # ABF: raise the base frequency
curve_abf = loss_per_position(model)
s_abf, l_abf = summarise(curve_abf, f"theta={NEW_THETA:.0f}, before fine-tune")
check(l_naive > s_naive, "before extension training, positions the model never saw are worse (RoPE is relative and Storyland docs are short, so expect a small gap)")

optimizers = build_optimizer(model, "adamw", lr=1e-4, weight_decay=0.1)
g = torch.Generator().manual_seed(args.seed + 1)
t0 = time.perf_counter()
for step in range(LONG_STEPS):
    set_lr(optimizers, lr_at(step, LONG_STEPS, 1.0, warmup_steps=0, kind="wsd", decay_frac=1.0))
    x, y = get_batch(mix_tokens, LONG_B, LONG_T, g)           # long windows from the packed mix
    _, loss = model(x, y)
    for o in optimizers:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in optimizers:
        o.step()
    if step % max(1, LONG_STEPS // 4) == 0 or step == LONG_STEPS - 1:
        print(f"   step {step:4d} | loss at seq_len {LONG_T}: {loss.item():.4f}")
long_wall = time.perf_counter() - t0
curve_ft = loss_per_position(model)
s_ft, l_ft = summarise(curve_ft, f"theta={NEW_THETA:.0f}, after {LONG_STEPS} steps")
print(f"   long-position loss: {l_naive:.3f} (naive) -> {l_abf:.3f} (ABF) -> {l_ft:.3f} (ABF + fine-tune)   [{long_wall:.0f}s]")
check(l_ft < l_naive, "fine-tuning at 512 improves the loss on positions 128-512")
check(l_ft - s_ft < (l_naive - s_naive), "the gap between short and long positions shrinks")

# ------------------------------------------------------------------ save & figure
math_final, story_final = eval_both(model)
print(f"\n   final held-out (seq {T_SHORT}): math {math_final:.3f} | stories {story_final:.3f}")
model.save(run_path("lab13_annealed.pt"), TOKENIZER_PATH,
           extra={"stage": "mid-training", "anneal_steps": ANNEAL_STEPS, "long_steps": LONG_STEPS})
print(f"   saved {run_path('lab13_annealed.pt')} (max_seq_len {model.cfg.max_seq_len}, theta {model.cfg.rope_theta:.0f})")

fig, axes = plt().subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(anneal_curve, lw=1, label="train loss on mix")
ax2 = ax.twinx(); ax2.plot(lr_curve, "r--", label="LR multiplier"); ax2.set_ylabel("LR x", color="r")
ax.set_xlabel("anneal step"); ax.set_ylabel("loss"); ax.set_title("(a) anneal: linear decay to 0"); ax.legend(loc="upper right")
ax = axes[1]
ax.bar([0, 1], [math_before, math_after], color=["#94a3b8", "#2563eb"], width=0.6)
ax.bar([3, 4], [story_before, story_after], color=["#94a3b8", "#f59e0b"], width=0.6)
ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["math\nbefore", "math\nafter", "stories\nbefore", "stories\nafter"])
ax.set_ylabel("held-out loss"); ax.set_title("(a) what the anneal changed")
ax = axes[2]
k = 16
sm = lambda c: torch.nn.functional.avg_pool1d(c[None, None], k, stride=k)[0, 0]    # smooth in 16-token bins
xs = [i * k + k / 2 for i in range(LONG_T // k)]
ax.plot(xs, sm(curve_naive), label=f"theta {orig_theta:.0f}, no training")
ax.plot(xs, sm(curve_abf), label=f"theta {NEW_THETA:.0f}, no training")
ax.plot(xs, sm(curve_ft), lw=2, label=f"theta {NEW_THETA:.0f} + {LONG_STEPS} steps @512")
ax.axvline(T_SHORT, color="k", ls=":", lw=1); ax.text(T_SHORT + 4, ax.get_ylim()[1] * 0.98, "trained length", va="top", fontsize=8)
ax.set_xlabel("position in the window"); ax.set_ylabel("loss"); ax.set_title("(b) loss per position"); ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "lab13_midtrain.png")
done()
