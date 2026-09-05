"""Lab 23: evaluate every TinyLM checkpoint you have, and learn what the numbers can and cannot say.

(a) every loadable checkpoint in runs/: task accuracy (exact match, greedy) and perplexity, as one
    markdown table saved to runs/eval_report.md; bootstrap confidence intervals per task;
(b) how many eval items you need: CI width vs n;
(c) contamination: plant eval items in a training set, show contamination_check catches them, fine-tune
    on the contaminated set, and watch the score on the leaked items inflate while a clean set does not;
(d) LLM-as-judge scaffolding: a position-bias check on a fair judge and on a biased one.

Run:  python3 labs/lab23_evals.py            (quick)
      python3 labs/lab23_evals.py --full
"""
from _common import setup, check, banner, section, savefig, done, plt

import glob
import os
import time

import torch

from llm import tasks
from llm.data import decontaminate, ngrams
from llm.evals import (bootstrap_ci, compare_checkpoints, contamination_check, eval_tasks, judge_pairwise,
                       perplexity, position_bias_check, rule_based_judge)
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_tokenizer, get_tokens, run_path
from llm.sft import SFTConfig, sft_train
from llm.tasks import TaskExample

args = setup("Lab 23: evaluation — every checkpoint, with error bars, contamination and judge checks")

N_EVAL = 28 if args.quick else 84            # eval items across the 7 task types
N_LEAK = 12 if args.quick else 20            # eval items planted into the training set
LEAK_STEPS = 150 if args.quick else 200
LEAK_COPIES = 5                              # a popular benchmark appears on many web pages
PPL_BATCHES = 5 if args.quick else 10

tok = get_tokenizer()
eval_set = tasks.make_examples(N_EVAL, seed=2023, max_value=20)
_, val_tokens = get_tokens(tok)                                          # held-out Storyland stream


# ------------------------------------------------------------------ (a) every checkpoint
section("(a) every loadable checkpoint in runs/")
paths = []
for p in sorted(glob.glob(run_path("*.pt"))):
    if p.endswith("_ckpt.pt"):
        continue                                                        # optimizer checkpoints from Lab 10
    try:
        TinyLM.load(p)
        paths.append(p)
    except Exception as e:                                              # noqa: BLE001
        print(f"   skipping {os.path.basename(p)}: {type(e).__name__}")
print(f"   {len(paths)} checkpoints: {[os.path.basename(p) for p in paths]}")

t0 = time.perf_counter()
rows = []
per_ckpt: dict[str, list[float]] = {}
for p in paths:
    model = TinyLM.load(p)
    res = eval_tasks(model, tok, eval_set, max_new_tokens=24)
    ppl_story = perplexity(model, val_tokens, batch_size=8, seq_len=min(128, model.cfg.max_seq_len), n_batches=PPL_BATCHES)
    lo, hi = bootstrap_ci(res.correct)
    stage = torch.load(p, map_location="cpu").get("extra", {}).get("stage", "?")
    rows.append((os.path.basename(p), stage, model.num_params(), res.accuracy, lo, hi, ppl_story, res.per_task))
    per_ckpt[os.path.basename(p)] = res.correct
    print(f"   {os.path.basename(p):28s} {stage:14s} acc {res.accuracy:.2f} [{lo:.2f}, {hi:.2f}]  ppl(story) {ppl_story:6.1f}  "
          f"{time.perf_counter() - t0:5.0f}s")
task_names = sorted({t for r in rows for t in r[7]})
lines = [f"# TinyLM checkpoint evaluation ({N_EVAL} items, greedy, 95% bootstrap CI)", "",
         "| checkpoint | stage | params | accuracy | 95% CI | ppl (Storyland) | " + " | ".join(task_names) + " |",
         "|---|---|---:|---:|---|---:|" + "---:|" * len(task_names)]
for name, stage, n, acc, lo, hi, ppl, per_task in rows:
    lines.append(f"| {name} | {stage} | {n:,} | {acc:.2f} | [{lo:.2f}, {hi:.2f}] | {ppl:.1f} | "
                 + " | ".join(f"{per_task.get(t, float('nan')):.2f}" for t in task_names) + " |")
lines += ["", "Perplexity on the chat-formatted eval items (compare_checkpoints):", "",
          compare_checkpoints(paths, tok, eval_set[:8], max_new_tokens=24, n_batches=3)]
report = "\n".join(lines)
with open(run_path("eval_report.md"), "w") as f:
    f.write(report + "\n")
print("\n" + report)
print(f"\n   saved runs/eval_report.md")
best = max(rows, key=lambda r: r[3])
base_rows = [r for r in rows if r[1] == "base"]
check(len(paths) >= 2, "at least two checkpoints were evaluated")
if base_rows:
    check(best[3] >= max(r[3] for r in base_rows), "no base checkpoint beats the best post-trained one on task accuracy")
# perplexity vs capability
if len(rows) >= 2:
    by_ppl = min(rows, key=lambda r: r[6])
    print(f"   lowest Storyland perplexity: {by_ppl[0]} ({by_ppl[6]:.1f}); highest accuracy: {best[0]} ({best[3]:.2f})")
    check(True, "perplexity and task accuracy rank checkpoints " + ("the same way" if by_ppl[0] == best[0] else "differently"))

# ------------------------------------------------------------------ (b) how many items?
section("(b) how wide is the error bar? CI half-width vs number of eval items")
best_correct = per_ckpt[best[0]]
import random
rng = random.Random(0)
print(f"   {'n':>5s} {'acc':>6s} {'95% CI':>14s} {'half-width':>11s}")
for n in (10, 25, 50, 100, 400):
    sample = [rng.choice(best_correct) for _ in range(n)]                  # resample the best model's scores
    lo, hi = bootstrap_ci(sample)
    print(f"   {n:5d} {sum(sample) / n:6.2f} [{lo:.2f}, {hi:.2f}]  {(hi - lo) / 2:11.2f}")
print("   rule of thumb: half-width ≈ 1 / sqrt(n) for accuracies near 0.5; a 0.05 gap needs ~400 items")

# ------------------------------------------------------------------ (c) contamination
section("(c) contamination: plant eval items in the training set and watch the score inflate")
# story_qa items: a 3-sentence story plus a question, ~45 words, so the production n = 13 applies
leaked = tasks.make_examples(N_LEAK, seed=31, tasks=["story_qa"])
clean = tasks.make_examples(N_LEAK, seed=32, tasks=["story_qa"])
clean_train = tasks.make_examples(100, seed=33, tasks=["story_qa"])
train_docs = [{"text": f"{ex.prompt} {ex.answer}", "source": "sft"} for ex in clean_train]
contaminated_docs = train_docs + [{"text": f"{ex.prompt} {ex.answer}", "source": "leak"} for ex in leaked] * LEAK_COPIES
n_gram = 13                                                                # the GPT-3 / Llama report value
print(f"   contamination_check (n={n_gram}): clean train set vs leaked eval  {contamination_check(train_docs, leaked, n=n_gram):.2f} | "
      f"contaminated train set vs leaked eval  {contamination_check(contaminated_docs, leaked, n=n_gram):.2f} | "
      f"contaminated vs the clean eval  {contamination_check(contaminated_docs, clean, n=n_gram):.2f}")
kept = decontaminate(contaminated_docs, [ex.prompt for ex in leaked], n=n_gram)
print(f"   decontaminate() drops {len(contaminated_docs) - len(kept)} of {len(contaminated_docs)} documents")
check(contamination_check(contaminated_docs, leaked, n=n_gram) > 0.9, "contamination_check flags the planted items")
check(contamination_check(train_docs, leaked, n=n_gram) < 0.2, "...and does not flag them against the clean training set")

student_path = next((p for p in paths if "student_sft_nano" in p), run_path("base_nano.pt"))
model = TinyLM.load(student_path)
res_leak0, res_clean0 = eval_tasks(model, tok, leaked, max_new_tokens=16), eval_tasks(model, tok, clean, max_new_tokens=16)
train_examples = clean_train + leaked * LEAK_COPIES
t0 = time.perf_counter()
sft_train(model, tok, train_examples, SFTConfig(steps=LEAK_STEPS, batch_size=16, lr=1e-3, log_every=1000), verbose=False)
res_leak1, res_clean1 = eval_tasks(model, tok, leaked, max_new_tokens=16), eval_tasks(model, tok, clean, max_new_tokens=16)
print(f"   fine-tuned {os.path.basename(student_path)} for {LEAK_STEPS} steps on 100 clean + {N_LEAK} leaked items x{LEAK_COPIES} ({time.perf_counter() - t0:.0f}s)")
print(f"   {'':22s} {'before':>8s} {'after':>8s}")
print(f"   {'leaked eval items':22s} {res_leak0.accuracy:8.2f} {res_leak1.accuracy:8.2f}   <- inflated: the model saw these exact prompts")
print(f"   {'clean eval items':22s} {res_clean0.accuracy:8.2f} {res_clean1.accuracy:8.2f}   <- the honest number")
check(res_leak1.accuracy - res_leak0.accuracy > res_clean1.accuracy - res_clean0.accuracy,
      "the leaked items improve more than the clean ones: the score is contaminated")

# ------------------------------------------------------------------ (d) judge position bias
section("(d) LLM-as-judge scaffolding: position bias")
ex = tasks.make_examples(1, seed=5, tasks=["add"], max_value=20)[0]
good, bad = ex.answer, f"{ex.prompt.split()[2]} + {ex.prompt.split()[4].rstrip('?')} = {ex.meta['answer'] + 1}"
print(f"   prompt {ex.prompt!r}: A = {good!r} (correct), B = {bad!r} (off by one)")
fair = position_bias_check(rule_based_judge, ex, good, bad)
first_wins = lambda p, a, b: "A"                                          # a judge that always picks the first answer
longer_wins = lambda p, a, b: "A" if len(a) >= len(b) else "B"            # a length-biased judge
biased = position_bias_check(first_wins, ex, good, bad)
print(f"   rule-based judge : forward {fair['forward']}, swapped {fair['swapped']}  -> consistent={fair['consistent']}")
print(f"   'first wins'     : forward {biased['forward']}, swapped {biased['swapped']}  -> position_bias={biased['position_bias']}")
verbose = "Well, let me think carefully about this. " + bad
print(f"   'longer wins'    : {judge_pairwise(ex, good, verbose, longer_wins)} picks the verbose wrong answer over {good!r}")
n_bias = 0
prompts = tasks.make_examples(20, seed=6, tasks=["add", "upper"], max_value=20)
for p in prompts:
    n_bias += position_bias_check(first_wins, p, p.answer, p.answer + " and more")["position_bias"]
print(f"   over 20 prompts the 'first wins' judge shows position bias on {n_bias}/20; the rule-based judge on "
      f"{sum(position_bias_check(rule_based_judge, p, p.answer, p.answer + ' and more')['position_bias'] for p in prompts)}/20")
check(fair["consistent"] and biased["position_bias"], "the swap test passes a fair judge and catches a position-biased one")

# ------------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 2, figsize=(12, 4))
names = [r[0].replace(".pt", "") for r in rows]
accs = [r[3] for r in rows]; err = [[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]]
axes[0].barh(range(len(rows)), accs, xerr=err, color="#2563eb", alpha=0.8, capsize=3)
axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(names, fontsize=8); axes[0].set_xlim(0, 1)
axes[0].set_xlabel(f"task accuracy ({N_EVAL} items, 95% CI)"); axes[0].set_title("every checkpoint")
axes[1].bar([0, 1], [res_leak0.accuracy, res_leak1.accuracy], width=0.35, color="#dc2626", label="leaked items")
axes[1].bar([0.4, 1.4], [res_clean0.accuracy, res_clean1.accuracy], width=0.35, color="#16a34a", label="clean items")
axes[1].set_xticks([0.2, 1.2]); axes[1].set_xticklabels(["before fine-tune", "after fine-tune"]); axes[1].set_ylim(0, 1)
axes[1].legend(); axes[1].set_title("contamination inflates the score")
fig.tight_layout()
savefig(fig, "lab23_evals.png")
done()
