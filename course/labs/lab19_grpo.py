"""Lab 19: GRPO with a verifiable reward — TinyLM learns to add (a, b <= 20).

(a) Warm start (the SFT model from Lab 17) and its greedy accuracy on 100 held-out sums.
(b) The pieces: a verifiable reward, one rollout group, group-relative advantages.
(c) The GRPO run (DAPO defaults: clip-higher, dynamic sampling, token-level loss, no KL),
    logging reward / accuracy / response length / clip fraction / entropy every step.
(d) Before vs after: greedy accuracy with a bootstrap CI, plus pass@8 at temperature 1
    (does RL add capability, or sharpen what the model could already sample?).
(e) Test-time compute: majority vote and best-of-N (oracle) accuracy vs the number of samples.
(f) --full only: ablations — sequence-level loss, symmetric clip, no dynamic sampling,
    Dr. GRPO (no std normalisation), GSPO (sequence ratio) — short runs from the same start.

Run:  python3 labs/lab19_grpo.py            (quick: nano model, ~20 steps)
      python3 labs/lab19_grpo.py --full     (small model, 40 steps + ablations)
"""
from _common import setup, check, banner, section, savefig, done, plt

import collections
import copy
import os
import random
import time

import numpy as np
import torch

from llm import chat, rl, tasks
from llm.evals import bootstrap_ci, eval_tasks
from llm.generate import sample_group
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_base_model, get_tokenizer, run_path
from llm.sft import SFTConfig, sft_train
from llm.tasks import TaskExample

args = setup("Lab 19: GRPO with verifiable rewards")
SIZE = "nano" if args.quick else "small"
STEPS = 20 if args.quick else 40
ABL_STEPS = 10
LR = 1e-4 if args.quick else 5e-5
PPO_EPOCHS = 1 if args.quick else 2     # 2 epochs make the clip real, but with lr 1e-4 they spike the KL; nano gets 1 epoch
MAX_NEW = 12
N_TTC = 40 if args.quick else 50            # prompts for the pass@k / majority-vote measurements
torch.manual_seed(args.seed)

# ----------------------------------------------------------------- (a) warm start
section("(a) warm start")
tok = get_tokenizer()
pad_id, end_id = tok.special_tokens["<|pad|>"], tok.special_tokens[chat.END]


# The task world is small: only 441 sums exist with a, b <= 20. SFT trains on 400 random draws
# (seed 1) and every evaluation uses 100 fresh draws (seed 999). Most evaluation prompts therefore
# also occur in the SFT set: these numbers measure accuracy on the task distribution, not
# generalisation to unseen sums (a strict 341/100 split is exercise 8 of Chapter 19: the small
# model memorises rather than generalises there).
TRAIN_SET = tasks.make_examples(400, seed=1, tasks=["add"], max_value=20)
EVAL_SET = tasks.make_examples(100, seed=999, tasks=["add"], max_value=20)
EVAL_OVERLAP = len({e.prompt for e in EVAL_SET} & {e.prompt for e in TRAIN_SET})


def warm_start(size: str) -> TinyLM:
    for name in (f"lab17_sft_{size}.pt", f"sft_{size}.pt"):
        if os.path.exists(run_path(name)):
            m = TinyLM.load(run_path(name))
            acc = eval_tasks(m, tok, EVAL_SET[:20], max_new_tokens=MAX_NEW).accuracy
            if acc >= 0.10 or name.startswith("lab17"):
                print(f"   loading SFT warm start {run_path(name)} (greedy accuracy on 20 sums: {acc:.2f})")
                return m
            print(f"   {run_path(name)} cannot add (accuracy {acc:.2f} on 20 sums; Lab 15's model may be multi-task) -> training our own")
    model, _ = get_base_model(quick=(size == "nano"), verbose=True)
    cfg = SFTConfig(steps=450 if size == "nano" else 700, batch_size=16, lr=1e-3 if size == "nano" else 3e-4,
                    log_every=100, eval_every=150)
    print(f"   no SFT checkpoint found: SFT on {len(TRAIN_SET)} 'add' examples, {cfg.steps} steps, lr {cfg.lr}")
    sft_train(model, tok, TRAIN_SET, cfg, val_examples=EVAL_SET[:30], verbose=True)
    model.save(run_path(f"lab17_sft_{size}.pt"), TOKENIZER_PATH, extra={"stage": "sft", "tasks": ["add"], "max_value": 20})
    return model


def greedy_eval(model: TinyLM, label: str, examples=EVAL_SET):
    res = eval_tasks(model, tok, examples, max_new_tokens=MAX_NEW)
    lo, hi = bootstrap_ci(res.correct)
    print(f"   greedy accuracy [{label}]: {res.accuracy:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (n={len(examples)})")
    return res.accuracy, (lo, hi)


@torch.no_grad()
def sample_answers(model: TinyLM, examples, n: int, seed: int) -> list[list[str]]:
    """n answers per prompt at temperature 1 (the same rollout primitive GRPO uses)."""
    out = []
    for i, ex in enumerate(examples):
        prompt_ids = tok.encode(chat.render(ex.messages(with_answer=False)))
        ids = sample_group(model, prompt_ids, n, MAX_NEW, temperature=1.0, stop_ids=[end_id], pad_id=pad_id, seed=seed + i)
        out.append([tok.decode(rl.split_completion(r, len(prompt_ids), pad_id, end_id)) for r in ids.tolist()])
    return out


def pass_at_k(examples, answers) -> tuple[float, float]:
    """(mean sample accuracy, fraction of prompts with at least one correct sample)."""
    ok = [[tasks.verify(ex, a) for a in ans] for ex, ans in zip(examples, answers)]
    return float(np.mean([np.mean(o) for o in ok])), float(np.mean([max(o) for o in ok]))


sft_model = warm_start(SIZE).eval()
print(f"   model: {sft_model.num_params():,} params ({SIZE}); {len(TRAIN_SET)} training prompts; eval = 100 fresh draws, {EVAL_OVERLAP} of which also occur in the SFT set")
acc0, ci0 = greedy_eval(sft_model, "SFT, before RL")
ttc_set = EVAL_SET[:N_TTC]
t0 = time.perf_counter()
samples_before = sample_answers(sft_model, ttc_set, 16, seed=100)
p1_before, p8_before = pass_at_k(ttc_set, [s[:8] for s in samples_before])
print(f"   at T=1 on {N_TTC} held-out prompts: sample accuracy {p1_before:.2f}, pass@8 {p8_before:.2f}   [{time.perf_counter() - t0:.0f}s]")

# ---------------------------------------------------------------- (b) the pieces
section("(b) the pieces: verifiable reward, one rollout group, group-relative advantages")
ex = TRAIN_SET[0]
for c in (ex.answer, ex.answer.replace(str(ex.meta["answer"]), str(ex.meta["answer"] + 1)), f"the answer is {ex.meta['answer']}", "kite kite"):
    print(f"   reward({c!r:28}) = {rl.default_reward(ex, c, [0]):.1f}   (verify {tasks.verify(ex, c):.0f} + 0.1 x format {tasks.format_reward(ex, c):.0f})")
cfg = rl.GRPOConfig(group_size=8, steps=STEPS, prompts_per_step=4, max_new_tokens=MAX_NEW, lr=LR,
                    clip_eps_low=0.2, clip_eps_high=0.28, dynamic_sampling=True, token_level_loss=True,
                    normalize_std=True, kl_coef=0.0, ppo_epochs=PPO_EPOCHS, seed=args.seed, log_every=max(1, STEPS // 10))
ro = rl.rollout_group(sft_model, tok, ex, cfg, seed=args.seed)
adv = rl.grpo_advantages(ro.rewards)
print(f"   prompt {ex.prompt!r}: G={cfg.group_size} rollouts, ids {tuple(ro.ids.shape)}, mask {tuple(ro.mask.shape)}, prompt_len {ro.prompt_len}")
for c, r, a, L in zip(ro.completions, ro.rewards.tolist(), adv.tolist(), ro.mask.sum(-1).tolist()):
    print(f"      {c!r:22} reward {r:.1f}  advantage {a:+.2f}  ({int(L)} answer tokens incl. <|end|>)")
print(f"   mean {ro.rewards.mean():.3f}, std {ro.rewards.std(unbiased=False):.3f}; advantages sum to {adv.sum():+.1e}; "
      f"Dr. GRPO would use {[round(x, 2) for x in rl.grpo_advantages(ro.rewards, normalize_std=False).tolist()]}")
check(abs(adv.sum().item()) < 1e-4, "group-relative advantages sum to zero")

# ------------------------------------------------------------------ (c) GRPO run
section(f"(c) GRPO: {STEPS} steps x {cfg.prompts_per_step} prompts x G={cfg.group_size}, lr {LR}, clip {cfg.clip_eps_low}/{cfg.clip_eps_high}, "
        f"ppo_epochs {cfg.ppo_epochs}, dynamic sampling on, token-level loss, no KL")
policy = copy.deepcopy(sft_model)
t0 = time.perf_counter()
hist = rl.grpo_train(policy, tok, TRAIN_SET, cfg, verbose=True)
grpo_wall = time.perf_counter() - t0
print(f"   {STEPS} steps in {grpo_wall:.0f}s ({grpo_wall / STEPS:.1f}s/step, of which rollouts {np.mean([h['t_rollout'] for h in hist]):.1f}s)")
k = max(3, STEPS // 6)
r_first, r_last = np.mean([h["reward"] for h in hist[:k]]), np.mean([h["reward"] for h in hist[-k:]])
H_first, H_last = np.nanmean([h["entropy"] for h in hist[:k]]), np.nanmean([h["entropy"] for h in hist[-k:]])
print(f"   training reward, first {k} steps {r_first:.3f} -> last {k} {r_last:.3f} | entropy {H_first:.2f} -> {H_last:.2f} | "
      f"mean clip frac {np.mean([h['clip_frac'] for h in hist]):.3f} | mean skipped {np.mean([h['skipped_frac'] for h in hist]):.2f} | "
      f"resp len {hist[0]['resp_len']:.1f} -> {hist[-1]['resp_len']:.1f}")
print(f"   training reward {'rose' if r_last > r_first else 'did not rise'} over the run (per-step values are noisy: 32 samples per step)")
if cfg.ppo_epochs > 1:
    check(np.mean([h["clip_frac"] for h in hist]) > 0, "with ppo_epochs=2 the clip is exercised (non-zero clip fraction)")
else:
    check(all(h["clip_frac"] == 0 for h in hist), "with ppo_epochs=1 every ratio is exactly 1, so the clip never fires (by construction)")
check(all(np.isfinite(h["loss"]) for h in hist), "every step produced a finite loss")

# ------------------------------------------------------------- (d) before vs after
section("(d) before vs after on the 100 held-out sums")
acc1, ci1 = greedy_eval(policy, "after GRPO")
samples_after = sample_answers(policy, ttc_set, 16, seed=100)
p1_after, p8_after = pass_at_k(ttc_set, [s[:8] for s in samples_after])
print(f"   {'':<10} {'greedy':>7} {'95% CI':>14} {'sample acc':>11} {'pass@8':>7}")
print(f"   {'before':<10} {acc0:>7.2f} [{ci0[0]:.2f}, {ci0[1]:.2f}] {p1_before:>11.2f} {p8_before:>7.2f}")
print(f"   {'after':<10} {acc1:>7.2f} [{ci1[0]:.2f}, {ci1[1]:.2f}] {p1_after:>11.2f} {p8_after:>7.2f}")
print(f"   greedy {acc0:.2f} -> {acc1:.2f}: {'the CIs overlap; call it suggestive, not proven' if ci1[0] <= ci0[1] else 'the CIs do not overlap'}")
print(f"   pass@8 {p8_before:.2f} -> {p8_after:.2f}: {'sharpening — RL moved mass onto answers the SFT model could already sample' if p8_after <= p8_before + 0.05 else 'more prompts became solvable at all'}")
check(acc1 >= acc0 - 0.10, "greedy accuracy after RL is within noise of, or above, the SFT start")
print(f"   (reported, not checked: greedy {acc0:.2f} -> {acc1:.2f}, sample accuracy {p1_before:.2f} -> {p1_after:.2f}, pass@8 {p8_before:.2f} -> {p8_after:.2f})")

# ----------------------------------------------------- (e) test-time compute
section("(e) test-time compute: majority vote vs best-of-N (oracle) vs N samples")


def majority(answers: list[str], task: str) -> str:
    votes = collections.Counter(tasks.extract_answer(task, a) for a in answers)
    votes.pop(None, None)
    return votes.most_common(1)[0][0] if votes else ""


def ttc_curve(examples, answers):
    rows = []
    for N in (1, 2, 4, 8, 16):
        maj = np.mean([float(majority(ans[:N], ex.task) == str(ex.meta["answer"])) for ex, ans in zip(examples, answers)])
        oracle = np.mean([max(tasks.verify(ex, a) for a in ans[:N]) for ex, ans in zip(examples, answers)])
        rows.append((N, maj, oracle))
    return rows


ttc_before, ttc_after = ttc_curve(ttc_set, samples_before), ttc_curve(ttc_set, samples_after)
print(f"   {'N':>3} | {'majority (before)':>17} {'majority (after)':>16} | {'pass@N (before)':>15} {'pass@N (after)':>14}")
for (N, mb, ob), (_, ma, oa) in zip(ttc_before, ttc_after):
    print(f"   {N:>3} | {mb:>17.2f} {ma:>16.2f} | {ob:>15.2f} {oa:>14.2f}")
check(all(ttc_after[i][2] <= ttc_after[i + 1][2] for i in range(4)), "pass@N never decreases with N (an oracle verifier only gains from more samples)")
check(ttc_after[-1][2] >= ttc_after[-1][1], "pass@N is an upper bound on majority voting")
print(f"   majority vote at N=16 vs N=1: {ttc_after[-1][1]:.2f} vs {ttc_after[0][1]:.2f} -> "
      f"{'voting helps' if ttc_after[-1][1] > ttc_after[0][1] else 'voting does not help here: it needs per-sample accuracy well above the most common wrong answer'}")

policy.save(run_path(f"grpo_{SIZE}.pt"), TOKENIZER_PATH, extra={"stage": "grpo", "steps": STEPS, "task": "add<=20"})
print(f"\n   saved {run_path(f'grpo_{SIZE}.pt')}")

# ------------------------------------------------------------------ (f) ablations
abl_results = []
if args.full:
    section(f"(f) ablations: {ABL_STEPS} steps each from the same SFT start (same seed); small-scale noise is expected")
    variants = [
        ("DAPO defaults (as above)", {}),
        ("sequence-level loss", {"token_level_loss": False}),
        ("symmetric clip 0.2/0.2", {"clip_eps_high": 0.2}),
        ("no dynamic sampling", {"dynamic_sampling": False}),
        ("Dr. GRPO (no std norm)", {"normalize_std": False}),
        ("GSPO (sequence ratio)", {"sequence_ratio": True, "clip_eps_low": 3e-4, "clip_eps_high": 4e-4}),
    ]
    abl_eval = EVAL_SET[:50]
    for name, over in variants:
        vcfg = rl.GRPOConfig(**{**cfg.__dict__, "steps": ABL_STEPS, "log_every": 10 ** 9, **over})
        m = copy.deepcopy(sft_model)
        t0 = time.perf_counter()
        h = rl.grpo_train(m, tok, TRAIN_SET, vcfg, verbose=False)
        acc = eval_tasks(m, tok, abl_eval, max_new_tokens=MAX_NEW).accuracy
        row = (name, np.mean([x["reward"] for x in h[:5]]), np.mean([x["reward"] for x in h[-5:]]), acc,
               np.mean([x["clip_frac"] for x in h]), np.nanmean([x["entropy"] for x in h[-5:]]), np.mean([x["skipped_frac"] for x in h]))
        abl_results.append(row)
        print(f"   {name:<26} reward {row[1]:.2f} -> {row[2]:.2f} | greedy acc (n=50) {acc:.2f} | clip {row[4]:.3f} | H {row[5]:.2f} | skipped {row[6]:.2f} | {time.perf_counter() - t0:.0f}s")
    print(f"   SFT start on the same 50: {eval_tasks(sft_model, tok, abl_eval, max_new_tokens=MAX_NEW).accuracy:.2f}; "
          f"a 50-prompt CI is about +-0.14, so only large gaps mean anything")
    check(all(np.isfinite(r[2]) for r in abl_results), "every variant trains without NaNs")

# ------------------------------------------------------------------- figures
fig, axes = plt().subplots(1, 4, figsize=(17, 3.8))
steps = [h["step"] for h in hist]
ax = axes[0]
ax.plot(steps, [h["reward"] for h in hist], color="#f59e0b", alpha=0.5, label="reward (per step)")
w = max(1, STEPS // 10)
ax.plot(steps, np.convolve([h["reward"] for h in hist], np.ones(w) / w, mode="same"), color="#f59e0b", lw=2, label=f"reward ({w}-step mean)")
ax.plot(steps, [h["accuracy"] for h in hist], color="#2563eb", alpha=0.4, label="sample accuracy")
ax.set_xlabel("GRPO step"); ax.set_title("(c) training reward"); ax.legend(fontsize=7)
ax = axes[1]
ax.plot(steps, [h["entropy"] for h in hist], color="#7c3aed", label="entropy (nats/token)")
ax.set_xlabel("GRPO step"); ax.set_title("(c) policy entropy at answer tokens"); ax.legend(fontsize=8)
ax = axes[2]
ax.plot(steps, [h["clip_frac"] for h in hist], color="#dc2626", label="clip fraction")
ax.plot(steps, [h["skipped_frac"] for h in hist], color="#64748b", ls="--", label="groups skipped")
ax.plot(steps, np.array([h["resp_len"] for h in hist]) / MAX_NEW, color="#16a34a", ls=":", label=f"resp len / {MAX_NEW}")
ax.set_xlabel("GRPO step"); ax.set_title("(c) clipping, skipping, length"); ax.legend(fontsize=7)
ax = axes[3]
Ns = [r[0] for r in ttc_after]
ax.plot(Ns, [r[1] for r in ttc_before], "o--", color="#94a3b8", label="majority, before RL")
ax.plot(Ns, [r[1] for r in ttc_after], "o-", color="#2563eb", label="majority, after RL")
ax.plot(Ns, [r[2] for r in ttc_before], "s--", color="#fbbf24", label="pass@N, before")
ax.plot(Ns, [r[2] for r in ttc_after], "s-", color="#f59e0b", label="pass@N, after")
ax.set_xscale("log", base=2); ax.set_xlabel("N samples"); ax.set_ylabel("accuracy"); ax.set_title("(e) test-time compute"); ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "lab19_grpo.png")
if abl_results:
    fig, ax = plt().subplots(figsize=(9, 3.5))
    names = [r[0] for r in abl_results]
    ax.bar(np.arange(len(names)) - 0.2, [r[2] for r in abl_results], 0.4, color="#f59e0b", label=f"train reward, last 5 of {ABL_STEPS} steps")
    ax.bar(np.arange(len(names)) + 0.2, [r[3] for r in abl_results], 0.4, color="#2563eb", label="greedy acc after (n=50)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8); ax.legend(fontsize=8)
    ax.set_title("(f) ablations from the same start; differences within ~0.14 are noise")
    fig.tight_layout()
    savefig(fig, "lab19_ablations.png")
done()
