"""Lab 10: the pretraining loop — AdamW vs Muon, schedules, throughput, resume.

    python3 labs/lab10_pretrain.py            # --quick (default): nano model, 150 steps each
    python3 labs/lab10_pretrain.py --full     # small model, 700 steps each (the real run)

The shared base checkpoint runs/base_small.pt belongs to the other labs; this lab
writes its own models to runs/lab10_*.pt and never touches it.
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import os
import time

import numpy as np
import torch

from llm.config import preset
from llm.generate import generate
from llm.model import TinyLM
from llm.optim import lr_at
from llm.pipeline import get_tokenizer, get_tokens, run_path
from llm.train import TrainConfig, train, estimate_loss, get_batch, load_checkpoint

args = setup("Lab 10: the pretraining loop (AdamW vs Muon)")

# --------------------------------------------------------------- 1. the data
section("1. the packed token stream and one batch")
tok = get_tokenizer()
train_tokens, val_tokens = get_tokens(tok)
V = tok.vocab_size
if args.quick:
    PRESET, STEPS, BS, SL, WARMUP, EVAL_EVERY = "nano", 150, 16, 64, 15, 50
else:
    PRESET, STEPS, BS, SL, WARMUP, EVAL_EVERY = "small", 700, 32, 128, 50, 100
print(f"train {len(train_tokens):,} tokens | val {len(val_tokens):,} tokens | vocab {V}")
x, y = get_batch(train_tokens, BS, SL, torch.Generator().manual_seed(0))
print(f"get_batch -> x {tuple(x.shape)}, y {tuple(y.shape)};  y[0,:6] = x[0,1:7]: {torch.equal(y[0, :6], x[0, 1:7])}")
print(f"x[0,:12] decodes to: {tok.decode(x[0, :12].tolist())!r}")
check(torch.equal(x[:, 1:], y[:, :-1]), "targets are inputs shifted one token left")

# ------------------------------------------------------ 2. LR schedules
section("2. learning-rate schedules from lr_at()")
steps = np.arange(STEPS)
sched = {k: [lr_at(s, STEPS, 1.0, WARMUP, k, 0.1) for s in steps] for k in ("cosine", "wsd", "constant")}
for k, v in sched.items():
    print(f"  {k:<9} step 0: {v[0]:.3f}  step {WARMUP}: {v[WARMUP]:.3f}  mid: {v[STEPS // 2]:.3f}  "
          f"at {int(0.85 * STEPS)}: {v[int(0.85 * STEPS)]:.3f}  last: {v[-1]:.3f}")
fig, ax = plt().subplots(figsize=(6, 3))
for k, c in (("cosine", "#2563eb"), ("wsd", "#f59e0b"), ("constant", "#64748b")):
    ax.plot(steps, sched[k], label=k, color=c, lw=1.8)
ax.axvspan(0, WARMUP, color="#16a34a", alpha=0.12, label=f"warmup ({WARMUP} steps)")
ax.set_xlabel("step"); ax.set_ylabel("LR multiplier"); ax.set_title("Lab 10: schedules (peak = 1.0)")
ax.legend(fontsize=8)
savefig(fig, "lab10_lr_schedules.png")
check(sched["wsd"][STEPS // 2] == 1.0 and sched["cosine"][STEPS // 2] < 0.7, "WSD stays flat where cosine has already decayed")

# ------------------------------------------------- 3. AdamW vs Muon
section(f"3. train '{PRESET}' twice: AdamW, then Muon  ({STEPS} steps × {BS} × {SL} = {STEPS * BS * SL:,} tokens each)")
runs = {}
for opt in ("adamw", "muon"):
    torch.manual_seed(args.seed)
    model = TinyLM(preset(PRESET, vocab_size=V, max_seq_len=max(SL, 128)))
    tc = TrainConfig(steps=STEPS, batch_size=BS, seq_len=SL, optimizer=opt, lr=1e-3, muon_lr=0.02,
                     weight_decay=0.1, warmup_steps=WARMUP, schedule="cosine", grad_clip=1.0,
                     eval_every=EVAL_EVERY, eval_batches=10, log_every=max(25, STEPS // 6), seed=args.seed,
                     ckpt_path=run_path(f"lab10_{opt}_ckpt.pt"))
    print(f"\n[{opt}] {model.num_params():,} non-embedding params, {model.flops_per_token():.2e} FLOPs/token")
    t0 = time.perf_counter()
    hist = train(model, train_tokens, val_tokens, tc)
    secs = time.perf_counter() - t0
    val = estimate_loss(model, val_tokens, BS, SL, n_batches=20)
    model.save(run_path(f"lab10_{opt}.pt"), extra={"stage": "lab10", "optimizer": opt})
    runs[opt] = dict(model=model, hist=hist, secs=secs, val=val, tps=hist.tokens_per_sec[-1])
    print(f"[{opt}] final val loss {val:.4f} (ppl {math.exp(val):.1f}) in {secs:.1f}s, {hist.tokens_per_sec[-1]:,.0f} tok/s"
          f" -> saved {os.path.relpath(run_path(f'lab10_{opt}.pt'))}")

a, m = runs["adamw"], runs["muon"]
print(f"\nAdamW {a['val']:.4f}  vs  Muon {m['val']:.4f}   (difference {a['val'] - m['val']:+.4f} nats, "
      f"{'Muon' if m['val'] < a['val'] else 'AdamW'} lower)")
uniform = math.log(V)
check(a["val"] < uniform - 2 and m["val"] < uniform - 2, "both optimizers cut the loss by > 2 nats from uniform")
check(abs(a["val"] - m["val"]) < 1.0, "AdamW and Muon land within 1 nat of each other at this scale")

fig, ax = plt().subplots(figsize=(7, 3.6))
for opt, c in (("adamw", "#2563eb"), ("muon", "#f59e0b")):
    h = runs[opt]["hist"]
    ax.plot(h.step, h.train_loss, color=c, lw=1.2, alpha=0.8, label=f"{opt} train")
    ax.plot(h.val_step, h.val_loss, "o--", color=c, lw=1.2, label=f"{opt} val")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(f"Lab 10: AdamW vs Muon, TinyLM '{PRESET}'")
ax.legend(fontsize=8)
savefig(fig, "lab10_adamw_vs_muon.png")

# ------------------------------------------------- 4. throughput and MFU
section("4. throughput, FLOP/s and a (rough) CPU MFU")
CPU_PEAK = 100e9     # placeholder: ~100 GFLOP/s for a laptop CPU in fp32; yours differs
for opt in ("adamw", "muon"):
    r = runs[opt]
    fpt = r["model"].flops_per_token()
    achieved = r["tps"] * fpt
    print(f"  {opt:<6} {r['tps']:>8,.0f} tok/s × {fpt:.2e} FLOP/token = {achieved / 1e9:6.1f} GFLOP/s "
          f"-> MFU ≈ {100 * achieved / CPU_PEAK:.0f}% of an assumed {CPU_PEAK / 1e9:.0f} GFLOP/s peak")
print("  (an H100 run reports 40–50% MFU against 990 TFLOP/s; the CPU 'peak' here is a rough placeholder)")
check(all(r["tps"] > 0 for r in runs.values()), "throughput was measured")

# ------------------------------------------- 5. checkpoint and resume
section("5. checkpoint at step 100, resume to 150 (nano model, so it is quick in both modes)")
ck = run_path("lab10_resume_ckpt.pt")
if os.path.exists(ck):
    os.remove(ck)
RB, RS = 16, 64
base_tc = dict(batch_size=RB, seq_len=RS, optimizer="adamw", lr=2e-3, warmup_steps=10, schedule="cosine",
               eval_every=50, eval_batches=5, log_every=10, seed=args.seed, ckpt_path=ck)
torch.manual_seed(args.seed)
m1 = TinyLM(preset("nano", vocab_size=V))
h1 = train(m1, train_tokens, val_tokens, TrainConfig(steps=100, **base_tc), verbose=False)
step_saved, _ = load_checkpoint(ck, TinyLM(preset("nano", vocab_size=V)))
print(f"run 1: trained 100 steps, loss {h1.train_loss[0]:.3f} -> {h1.train_loss[-1]:.3f}; checkpoint saved at step {step_saved}")
m2 = TinyLM(preset("nano", vocab_size=V))                    # fresh weights, will be overwritten by the checkpoint
h2 = train(m2, train_tokens, val_tokens, TrainConfig(steps=150, **base_tc), resume_from=ck, verbose=True)
i_resume = h2.step.index(100)
print(f"note: run 1 had steps=100 so its cosine had decayed to lr×0.1 by step 99; run 2 has steps=150, so at step 100 "
      f"lr_at() gives ×{lr_at(100, 150, 1.0, 10, 'cosine', 0.1):.3f} — the schedule depends on the planned length (WSD avoids this)")
print(f"run 2: resumed at step {step_saved}; first logged loss after resume {h2.train_loss[i_resume]:.3f} "
      f"(before the checkpoint: {h1.train_loss[-1]:.3f}; a fresh model would be at ~{math.log(V):.2f})")
print(f"       history now spans steps {h2.step[0]}..{h2.step[-1]} with {len(h2.step)} log points")
check(step_saved == 100, "checkpoint records step 100")
check(h2.step[0] == 0 and h2.step[-1] == 149, "resumed history continues the original one to step 149")
check(abs(h2.train_loss[i_resume] - h1.train_loss[-1]) < 0.6, "loss after resume continues from where it stopped")
check(h2.train_loss[i_resume] < math.log(V) - 1.5, "the resumed model is far from a fresh one")
# the two histories agree on the first 100 steps because the first 100 were loaded from the checkpoint
check(h2.train_loss[:len(h1.train_loss)] == h1.train_loss, "loaded history matches the original run exactly")

# ------------------------------------------------------ 6. generate
section("6. generate from the better model")
best = "muon" if m["val"] < a["val"] else "adamw"
model = TinyLM.load(run_path(f"lab10_{best}.pt"))
for prompt in ("Mia had a", "What is 12 + 7?"):
    out = generate(model, tok, prompt, max_new_tokens=40, temperature=0.7, top_k=40, seed=args.seed)
    print(f"  [{best}] {prompt!r} -> {out!r}")
check(len(out) > 0, "the final model generates text")

print(f"\nwall-clock: AdamW {a['secs']:.0f}s, Muon {m['secs']:.0f}s")
done()
