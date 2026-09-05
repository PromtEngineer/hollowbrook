"""Lab 15: supervised fine-tuning — turn the base model into an instruction follower.

    python3 labs/lab15_sft.py            # --quick: TinyLM-nano, 300 steps, saves runs/sft_nano.pt
    python3 labs/lab15_sft.py --full     # TinyLM-small, 400 steps, saves runs/sft_small.pt (later labs load it)

What you will see:
  1. the instruction set (tasks.make_examples) rendered through the chat template, with the loss mask;
  2. task accuracy of the BASE model (evals.eval_tasks, with a bootstrap CI): zero;
  3. full fine-tuning with sft.sft_train: the masked loss falling, accuracy measured every 100 steps;
  4. the same questions answered after SFT, and a per-task accuracy table;
  5. what SFT cost: Storyland perplexity before and after (catastrophic forgetting, measured);
  6. LoRA vs full fine-tuning: trainable parameters, wall-clock, accuracy; the adapter is a no-op at
     initialisation and merges back into an ordinary TinyLM.
"""
from _common import setup, check, banner, section, savefig, done, plt

import copy
import time

import torch

from llm.chat import build_sft_example
from llm.evals import eval_tasks, perplexity
from llm.model import TinyLM
from llm.pipeline import get_tokenizer, get_base_model, get_tokens, run_path, TOKENIZER_PATH
from llm.sft import SFTConfig, sft_train, describe_mask, apply_lora, merge_lora, trainable_params, LoRALinear, respond
from llm.tasks import make_examples

args = setup("Lab 15: supervised fine-tuning (full FT vs LoRA)")

TASKS = ["upper", "reverse", "add", "count"]
N_TRAIN = 2000
N_VAL = 32 if args.quick else 48
STEPS = 300 if args.quick else 400
LR = 1e-3                      # sft.SFTConfig's default is 3e-4; TinyLM is tiny, and 1e-3 converges 3x faster here
LORA_STEPS = 150 if args.quick else 200
LORA_RANK = 8
LORA_LR = 2e-3                 # adapters tolerate (and need) a higher LR than the full weights
EVAL_EVERY = 100
MAX_NEW = 16                   # every answer in these tasks fits in 12 tokens + <|end|>

tok = get_tokenizer()
base, _ = get_base_model(quick=args.quick, verbose=False)
n_all = sum(p.numel() for p in base.parameters())
print(f"base model: {base.num_params():,} non-embedding params ({n_all:,} total), max_seq_len {base.cfg.max_seq_len}")

# ------------------------------------------------------------ 1. data
section(f"1. instruction data: {N_TRAIN} train / {N_VAL} val examples on {TASKS}")
train = make_examples(N_TRAIN, seed=args.seed + 1, tasks=TASKS)
val = make_examples(N_VAL, seed=args.seed + 2, tasks=TASKS)
for ex in train[:4]:
    print(f"  [{ex.task:<7}] {ex.prompt!r:<40} -> {ex.answer!r}")
ids, mask = build_sft_example(tok, train[0].messages())
print("one example through the template:\n  " + describe_mask(tok, ids, mask))
n_tok = sum(len(build_sft_example(tok, ex.messages())[0]) for ex in train[:200])
n_tr = sum(sum(build_sft_example(tok, ex.messages())[1]) for ex in train[:200])
print(f"first 200 examples: {n_tok:,} tokens, {n_tr:,} trainable ({100 * n_tr / n_tok:.0f}%); "
      f"{STEPS} steps x 16 = {STEPS * 16:,} examples = {STEPS * 16 / N_TRAIN:.1f} epochs")
overlap = {ex.prompt for ex in train} & {ex.prompt for ex in val}
print(f"val prompts that also appear in train: {len(overlap)}/{N_VAL} (a 55-word pool repeats; 'add' prompts are fresh)")

# ------------------------------------------------------------ 2. before
section("2. the base model on the tasks (greedy, max 16 new tokens)")
_, val_tokens = get_tokens(tok)
t0 = time.perf_counter()
before = eval_tasks(base, tok, val, max_new_tokens=MAX_NEW)
print(before.table())
print(f"  ({time.perf_counter() - t0:.1f}s; every answer runs to the token limit because the base never emits <|end|>)")
for prompt, completion, ok in before.samples[:3]:
    print(f"  {prompt!r:<40} -> {completion!r}")
ppl_before = perplexity(base, val_tokens, batch_size=8, seq_len=128, n_batches=8)
print(f"Storyland val perplexity before SFT: {ppl_before:.2f}")
check(before.accuracy == 0.0, "the base model scores 0.00: it does not know the format")

# ------------------------------------------------------------ 3. full fine-tuning
section(f"3. full fine-tuning: {STEPS} steps, batch 16, AdamW lr {LR:.0e}, cosine, loss masked to assistant tokens")
model = copy.deepcopy(base)
cfg = SFTConfig(steps=STEPS, batch_size=16, lr=LR, warmup_steps=20, schedule="cosine",
                log_every=50, eval_every=EVAL_EVERY, seed=args.seed)
t0 = time.perf_counter()
hist = sft_train(model, tok, train, cfg, val_examples=val, verbose=True)
full_time = time.perf_counter() - t0
print(f"full FT: {full_time:.0f}s for {STEPS} steps ({full_time / STEPS:.2f}s/step) | "
      f"loss {hist.train_loss[0]:.2f} -> {hist.train_loss[-1]:.2f} | val acc by step: "
      + ", ".join(f"{s}:{a:.2f}" for s, a in zip(hist.val_step, hist.val_acc)))
check(hist.train_loss[-1] < hist.train_loss[0] / 3, "the masked loss fell by more than 3x")
check(hist.val_acc[-1] > 0.0, "accuracy rose above zero: the model now answers in the format")

# ------------------------------------------------------------ 4. after
section("4. after SFT: the same tasks")
after = eval_tasks(model, tok, val, max_new_tokens=MAX_NEW)
print(after.table())
shown = set()
for prompt, completion, ok in after.samples:
    task = next(ex.task for ex in val if ex.prompt == prompt)
    if task not in shown or (ok and task + "ok" not in shown):
        want = next(ex.answer for ex in val if ex.prompt == prompt)
        print(f"  [{task:<7}] {prompt!r:<40} -> {completion!r:<22} {'✓' if ok else '✗'} (want {want!r})")
        shown.add(task); shown.add(task + "ok" if ok else task)
    if len(shown) >= 8:
        break
easy = (after.per_task.get("upper", 0) + after.per_task.get("reverse", 0)) / 2
print(f"mean accuracy on upper+reverse (copy tasks): {easy:.2f} | add: {after.per_task.get('add', 0):.2f} | count: {after.per_task.get('count', 0):.2f}")
check(easy >= (0.5 if args.quick else 0.8), f"clear accuracy on the copy tasks upper/reverse ({easy:.2f})")
ends = [respond(model, tok, ex.prompt, max_new_tokens=MAX_NEW) for ex in val[:8]]
check(all(len(tok.encode(a, allowed_special=False)) <= 12 for a in ends), "answers are short: the model learned to emit <|end|>")
ppl_after = perplexity(model, val_tokens, batch_size=8, seq_len=128, n_batches=8)
print(f"Storyland val perplexity: {ppl_before:.2f} before -> {ppl_after:.2f} after SFT "
      f"({100 * (ppl_after / ppl_before - 1):+.0f}%): this is catastrophic forgetting, measured")
check(ppl_after > ppl_before, "SFT on four narrow tasks made the model WORSE at Storyland prose (no replay data)")

save_path = run_path("sft_nano.pt" if args.quick else "sft_small.pt")
model.save(save_path, TOKENIZER_PATH, extra={"stage": "sft", "tasks": TASKS, "steps": STEPS, "lr": LR,
                                              "accuracy": after.accuracy, "per_task": after.per_task})
print(f"saved {save_path} (accuracy {after.accuracy:.2f}); Labs 14, 16, 17, 19 and 20 load it")

# ------------------------------------------------------------ 5. LoRA
section(f"5. LoRA rank {LORA_RANK}: freeze the base, train small adapters, merge them back")
demo = copy.deepcopy(base)                       # a throwaway copy to look at the adapters themselves
x_probe = torch.tensor([ids[:32]])
with torch.no_grad():
    logits_before, _ = demo(x_probe)
apply_lora(demo, LORA_RANK)
n_lora = trainable_params(demo)
n_layers_wrapped = sum(isinstance(m, LoRALinear) for m in demo.modules())
with torch.no_grad():
    logits_after, _ = demo(x_probe)
q = demo.blocks[0].attn.q_proj
print(f"wrapped {n_layers_wrapped} linear layers; block 0 q_proj: W {tuple(q.base.weight.shape)} frozen, "
      f"A {tuple(q.lora_A.shape)} + B {tuple(q.lora_B.shape)} trainable, scale alpha/r = {q.scale:.1f}")
print(f"trainable params: {n_lora:,} (LoRA) vs {n_all:,} (full) = {100 * n_lora / n_all:.1f}%")
check(torch.allclose(logits_before, logits_after, atol=1e-5), "at initialisation B = 0, so the LoRA model computes exactly what the base did")
check(n_lora < 0.1 * n_all, "LoRA trains fewer than 10% of the parameters")

# sft_train applies LoRA itself when cfg.lora_rank > 0 (and merges it at the end), so hand it a FRESH copy:
# calling apply_lora twice on one model freezes the first adapters and leaves nothing trainable.
lora_model = copy.deepcopy(base)
lcfg = SFTConfig(steps=LORA_STEPS, batch_size=16, lr=LORA_LR, warmup_steps=20, schedule="cosine",
                 log_every=50, eval_every=EVAL_EVERY, seed=args.seed, lora_rank=LORA_RANK)
t0 = time.perf_counter()
lhist = sft_train(lora_model, tok, train, lcfg, val_examples=val, verbose=True)
lora_time = time.perf_counter() - t0
check(not any(isinstance(m, LoRALinear) for m in lora_model.modules()), "sft_train merged the adapters: the result is a plain TinyLM again")
check(set(k for k, _ in lora_model.named_parameters()) == set(k for k, _ in base.named_parameters()),
      "...with the same parameter names as the base (so save/load work)")
lora_after = eval_tasks(lora_model, tok, val, max_new_tokens=MAX_NEW)
ppl_lora = perplexity(lora_model, val_tokens, batch_size=8, seq_len=128, n_batches=8)
print(f"\n{'method':<12}{'trainable':>12}{'steps':>7}{'time':>8}{'s/step':>8}{'accuracy':>10}{'upper':>7}{'reverse':>9}{'add':>6}{'count':>7}{'ppl':>7}")
for name, n, st, t, r, pp in (("full FT", n_all, STEPS, full_time, after, ppl_after),
                              (f"LoRA r={LORA_RANK}", n_lora, LORA_STEPS, lora_time, lora_after, ppl_lora)):
    print(f"{name:<12}{n:>12,}{st:>7}{t:>7.0f}s{t / st:>8.2f}{r.accuracy:>10.2f}{r.per_task.get('upper', 0):>7.2f}"
          f"{r.per_task.get('reverse', 0):>9.2f}{r.per_task.get('add', 0):>6.2f}{r.per_task.get('count', 0):>7.2f}{pp:>7.2f}")
check(lora_after.accuracy > 0.0, "LoRA also learns the format")
check(lora_time / LORA_STEPS < full_time / STEPS * 1.05, "a LoRA step is no slower than a full step (no optimizer state for frozen weights)")

# ------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(hist.step, hist.train_loss, color="#2563eb", label="full FT")
ax.plot(lhist.step, lhist.train_loss, color="#f59e0b", label=f"LoRA r={LORA_RANK}")
ax.set_xlabel("step"); ax.set_ylabel("masked train loss"); ax.set_title("SFT loss (assistant tokens only)"); ax.legend()
ax = axes[1]
ax.plot(hist.val_step, hist.val_acc, marker="o", color="#2563eb", label="full FT")
ax.plot(lhist.val_step, lhist.val_acc, marker="s", color="#f59e0b", label=f"LoRA r={LORA_RANK}")
ax.set_ylim(0, 1); ax.set_xlabel("step"); ax.set_ylabel("val accuracy (exact match)"); ax.set_title("accuracy during training"); ax.legend()
ax = axes[2]
xs = range(len(TASKS)); w = 0.27
ax.bar([x - w for x in xs], [before.per_task.get(t, 0) for t in TASKS], w, color="#94a3b8", label="base")
ax.bar(list(xs), [after.per_task.get(t, 0) for t in TASKS], w, color="#2563eb", label="full FT")
ax.bar([x + w for x in xs], [lora_after.per_task.get(t, 0) for t in TASKS], w, color="#f59e0b", label="LoRA")
ax.set_xticks(list(xs)); ax.set_xticklabels(TASKS); ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title("per task"); ax.legend()
fig.tight_layout()
savefig(fig, "lab15_sft.png")
done()
