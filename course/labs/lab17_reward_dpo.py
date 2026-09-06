"""Lab 17: reward models and preference optimisation on TinyLM.

(a) Warm start: an SFT'd TinyLM that answers "What is a + b?" (a, b <= 20) some of the time.
(b) Synthetic preference pairs -> a scalar reward model trained with the Bradley-Terry loss.
(c) Goodhart's law, measured: best-of-N against two proxies (the reward model, and a
    rubric that never checks the number) vs the gold verifier. Plus two static exploits.
(d) On-policy pairs: sample from the SFT model, grade with the verifier, pair right vs wrong.
(e) DPO on those pairs: pair accuracy, implicit-reward margin, greedy task accuracy.
(f) Likelihood displacement: many DPO steps on a handful of pairs -> pair accuracy 1.0 while
    the chosen answers themselves become LESS likely and greedy accuracy falls.
(g) SimPO vs DPO on the same pairs.

Run:  python3 labs/lab17_reward_dpo.py            (quick: nano model)
      python3 labs/lab17_reward_dpo.py --full     (small model)
"""
from _common import setup, check, banner, section, savefig, done, plt

import copy
import os
import random
import time

import torch

from llm import chat, tasks
from llm.dpo import DPOConfig, dpo_eval, dpo_train, make_reference, sequence_logprob, build_pair_batch
from llm.evals import bootstrap_ci, eval_tasks
from llm.generate import sample_group
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_base_model, get_tokenizer, run_path
from llm.reward import (ARITHMETIC_RUBRIC, RMConfig, RewardModel, make_preference_pairs,
                        make_preference_pairs_from_model, reward_accuracy, rubric_reward,
                        score_completions, train_reward_model)
from llm.sft import SFTConfig, sft_train
from llm.tasks import TaskExample

args = setup("Lab 17: reward model, Goodhart, DPO and its variants")
SIZE = "nano" if args.quick else "small"
RM_STEPS = 60 if args.quick else 120
DPO_STEPS = 40 if args.quick else 60
DISP_STEPS = 60 if args.quick else 100          # likelihood-displacement run
N_TRAIN_PROMPTS = 120 if args.quick else 200    # prompts used to build pairs
N_BON_PROMPTS = 16 if args.quick else 24        # prompts for the best-of-N demo
MAX_NEW = 12                                    # "13 + 8 = 21<|end|>" is 8 tokens
torch.manual_seed(args.seed)

# ----------------------------------------------------------------- (a) warm start
section("(a) warm start: an SFT'd TinyLM on 'add' (a, b <= 20)")
tok = get_tokenizer()


# The task world is small: only 441 sums exist with a, b <= 20. SFT trains on 400 random draws
# (seed 1) and every evaluation uses 100 fresh draws (seed 999). Most evaluation prompts therefore
# also occur in the SFT set: these numbers measure accuracy on the task distribution, not
# generalisation to unseen sums (a strict 341/100 split is exercise 8 of Chapter 19: the small
# model memorises rather than generalises there).
TRAIN_SET = tasks.make_examples(400, seed=1, tasks=["add"], max_value=20)
EVAL_SET = tasks.make_examples(100, seed=999, tasks=["add"], max_value=20)
EVAL_OVERLAP = len({e.prompt for e in EVAL_SET} & {e.prompt for e in TRAIN_SET})


def warm_start(size: str) -> TinyLM:
    """Load an SFT checkpoint (another lab's, or ours), or train one and save it."""
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
    print(f"   saved {run_path(f'lab17_sft_{size}.pt')}")
    return model


def greedy_accuracy(model: TinyLM, label: str) -> float:
    res = eval_tasks(model, tok, EVAL_SET, max_new_tokens=MAX_NEW)
    lo, hi = bootstrap_ci(res.correct)
    print(f"   greedy accuracy [{label}]: {res.accuracy:.2f}  (95% CI [{lo:.2f}, {hi:.2f}], n={len(EVAL_SET)})")
    return res.accuracy


sft_model = warm_start(SIZE).eval()
print(f"   model: {sft_model.num_params():,} params ({SIZE}); eval = 100 fresh draws, {EVAL_OVERLAP} of which also occur among the 400 SFT prompts")
acc_sft = greedy_accuracy(sft_model, "SFT warm start")
for ex in EVAL_SET[:3]:
    from llm.sft import respond
    print(f"      {ex.prompt!r:22} -> {respond(sft_model, tok, ex.prompt, max_new_tokens=MAX_NEW)!r}")

# ----------------------------------------------------------- (b) synthetic pairs + RM
section("(b) synthetic preference pairs and a Bradley-Terry reward model")
pair_prompts = TRAIN_SET[:N_TRAIN_PROMPTS]
syn_pairs = make_preference_pairs(pair_prompts, n_wrong_styles=2, seed=args.seed)
random.Random(args.seed).shuffle(syn_pairs)
n_val = len(syn_pairs) // 5
rm_val, rm_train = syn_pairs[:n_val], syn_pairs[n_val:]
print(f"   {len(syn_pairs)} pairs from {len(pair_prompts)} prompts (2 failure styles each): "
      f"{len(rm_train)} train / {len(rm_val)} val")
for p in syn_pairs[:4]:
    print(f"      [{p.meta['style']:>10}] chosen={p.chosen!r:18} rejected={p.rejected!r}")

rm = RewardModel(copy.deepcopy(sft_model))            # InstructGPT: RM initialised from the SFT model
acc_before = reward_accuracy(rm, tok, rm_val)
t0 = time.perf_counter()
rm_hist = train_reward_model(rm, tok, rm_train, RMConfig(steps=RM_STEPS, batch_size=8, lr=1e-4, warmup=5,
                                                          log_every=max(1, RM_STEPS // 6), seed=args.seed))
rm_wall = time.perf_counter() - t0
acc_after = reward_accuracy(rm, tok, rm_val)
print(f"   held-out pair accuracy: {acc_before:.2f} (zero head: every margin is 0) -> {acc_after:.2f}   [{rm_wall:.0f}s]")
check(acc_after >= 0.6, f"reward model ranks {acc_after:.0%} of held-out synthetic pairs correctly (chance = 50%)")

by_style: dict[str, list[float]] = {}
with torch.no_grad():
    for p in rm_val:
        s = score_completions(rm, tok, p.prompt_messages, [p.chosen, p.rejected])
        by_style.setdefault(p.meta["style"], []).append(float(s[0] > s[1]))
print("   accuracy by failure style: " + ", ".join(f"{k}={sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(by_style.items())))

# ---------------------------------------------------------- (c) Goodhart, measured
section("(c) reward hacking: best-of-N against a proxy vs the gold verifier")
end_id, pad_id = tok.special_tokens[chat.END], tok.special_tokens["<|pad|>"]
stop_ids = {end_id, pad_id, tok.special_tokens["<|eos|>"]}
N_MAX = 16
bon_prompts = EVAL_SET[:N_BON_PROMPTS]
samples: list[list[str]] = []
with torch.no_grad():
    for i, ex in enumerate(bon_prompts):
        prompt_ids = tok.encode(chat.render(ex.messages(with_answer=False)))
        rows = sample_group(sft_model, prompt_ids, N_MAX, MAX_NEW, temperature=1.0,
                            stop_ids=[end_id], pad_id=pad_id, seed=args.seed + i)
        comps = []
        for r in rows.tolist():
            ids = []
            for t in r[len(prompt_ids):]:
                if t in stop_ids:
                    break
                ids.append(t)
            comps.append(tok.decode(ids))
        samples.append(comps)
gold = [[tasks.verify(ex, c) for c in comps] for ex, comps in zip(bon_prompts, samples)]
rm_scores = [score_completions(rm, tok, ex.messages(with_answer=False), comps).tolist()
             for ex, comps in zip(bon_prompts, samples)]
rub_scores = [[rubric_reward(c, ARITHMETIC_RUBRIC)[0] for c in comps] for comps in samples]
print(f"   {N_BON_PROMPTS} held-out prompts x {N_MAX} samples at T=1: sample accuracy "
      f"{sum(map(sum, gold)) / (N_BON_PROMPTS * N_MAX):.2f}")
print(f"   {'N':>3} | {'RM pick: proxy':>14} {'gold':>5} | {'rubric pick: proxy':>18} {'gold':>5} | {'any correct':>11}")
bon_rows = []
for N in (1, 2, 4, 8, 16):
    rm_g, rm_p, ru_g, ru_p, cov = [], [], [], [], []
    for g, s_rm, s_ru in zip(gold, rm_scores, rub_scores):
        i_rm = max(range(N), key=lambda i: s_rm[i]); i_ru = max(range(N), key=lambda i: s_ru[i])
        rm_g.append(g[i_rm]); rm_p.append(s_rm[i_rm]); ru_g.append(g[i_ru]); ru_p.append(s_ru[i_ru])
        cov.append(float(any(g[:N])))
    row = (N, sum(rm_p) / len(rm_p), sum(rm_g) / len(rm_g), sum(ru_p) / len(ru_p), sum(ru_g) / len(ru_g), sum(cov) / len(cov))
    bon_rows.append(row)
    print(f"   {N:>3} | {row[1]:>14.2f} {row[2]:>5.2f} | {row[3]:>18.2f} {row[4]:>5.2f} | {row[5]:>11.2f}")
check(bon_rows[-1][3] >= bon_rows[0][3] and bon_rows[-1][4] <= bon_rows[-1][5],
      "the rubric proxy rises with N while gold accuracy stays below coverage: optimising a proxy is not optimising the goal")
# two static exploits of the graders themselves
ex = bon_prompts[0]
lenient = f"{ex.prompt.split()[2]} + {ex.prompt.split()[4].rstrip('?')} = 1 2 3 {ex.meta['answer']}"
print(f"   extract_answer leniency: verify({lenient!r}) = {tasks.verify(ex, lenient):.1f}  (the LAST number after '=' is graded)")
fake = "0 + 0 = 0"
print(f"   rubric exploit: rubric_reward({fake!r}) = {rubric_reward(fake, ARITHMETIC_RUBRIC)[0]:.2f} for {ex.prompt!r} "
      f"(the rubric checks shape, never the number); verify = {tasks.verify(ex, fake):.1f}")
check(rubric_reward(fake, ARITHMETIC_RUBRIC)[0] == 1.0 and tasks.verify(ex, lenient) == 1.0,
      "both graders can be satisfied by an answer that is not an answer")

# ---------------------------------------------------------------- (d) on-policy pairs
section("(d) on-policy pairs: sample from the SFT model, grade, pair right vs wrong")
t0 = time.perf_counter()
op_pairs, stats = make_preference_pairs_from_model(sft_model, tok, TRAIN_SET[:N_TRAIN_PROMPTS], n_samples=4,
                                                   seed=args.seed, max_new_tokens=MAX_NEW, temperature=1.0)
print(f"   {stats}   [{time.perf_counter() - t0:.0f}s]")
print(f"   {stats['n_pairs']}/{stats['n_prompts']} prompts yield a pair; {stats['n_all_correct']} were all-correct "
      f"(nothing to rank), {stats['n_all_wrong']} all-wrong (no positive example)")
for p in op_pairs[:3]:
    print(f"      chosen={p.chosen!r:16} rejected={p.rejected!r}")
random.Random(args.seed).shuffle(op_pairs)
n_val = max(4, len(op_pairs) // 5)
op_val, op_train = op_pairs[:n_val], op_pairs[n_val:]
check(len(op_train) >= 8, f"enough on-policy pairs to train on ({len(op_train)} train / {len(op_val)} val)")

# ------------------------------------------------------------------------ (e) DPO
section(f"(e) DPO on {len(op_train)} on-policy pairs, {DPO_STEPS} steps")
policy = copy.deepcopy(sft_model)
ref = make_reference(sft_model)
dcfg = DPOConfig(steps=DPO_STEPS, batch_size=8, lr=5e-5, beta=0.1, warmup=5, log_every=max(1, DPO_STEPS // 5), seed=args.seed)
before = dpo_eval(policy, ref, tok, op_val, dcfg)
print(f"   before: val pair acc {before['accuracy']:.2f} | margin {before['margin']:+.3f} | loss {before['loss']:.3f}  (= log 2: policy == reference)")
t0 = time.perf_counter()
dpo_hist = dpo_train(policy, ref, tok, op_train, dcfg)
dpo_wall = time.perf_counter() - t0
after = dpo_eval(policy, ref, tok, op_val, dcfg)
print(f"   after:  val pair acc {after['accuracy']:.2f} | margin {after['margin']:+.3f} | loss {after['loss']:.3f}   [{dpo_wall:.0f}s]")
train_after = dpo_eval(policy, ref, tok, op_train, dcfg)
print(f"   train pairs: acc {train_after['accuracy']:.2f} | margin {train_after['margin']:+.3f}   "
      f"(held-out: {len(op_val)} pairs only; treat a small move either way as noise)")
acc_dpo = greedy_accuracy(policy, "after DPO")
check(train_after["accuracy"] > 0.5 and train_after["margin"] > 0, "DPO separates the pairs it trained on (positive implicit-reward margin)")
print(f"   greedy accuracy: SFT {acc_sft:.2f} -> DPO {acc_dpo:.2f}  (100 held-out prompts; CI is about +-0.10, read small moves as noise)")


@torch.no_grad()
def mean_logps(model: TinyLM, pairs) -> tuple[float, float]:
    """Mean summed log-prob of the chosen and of the rejected answers over ``pairs``."""
    c_ids, c_mask, r_ids, r_mask = build_pair_batch(tok, pairs, pad_id)
    return (sequence_logprob(model, c_ids, c_mask).mean().item(),
            sequence_logprob(model, r_ids, r_mask).mean().item())


# ------------------------------------------------------ (f) likelihood displacement
section(f"(f) likelihood displacement: {DISP_STEPS} DPO steps at lr 5e-5 on 24 pairs")
disp_pairs = op_train[:24]
disp = copy.deepcopy(sft_model)
c0, r0 = mean_logps(disp, disp_pairs)
disp_cfg = DPOConfig(steps=DISP_STEPS, batch_size=8, lr=5e-5, beta=0.1, warmup=5, log_every=max(1, DISP_STEPS // 5), seed=args.seed)
disp_hist = dpo_train(disp, ref, tok, disp_pairs, disp_cfg)
c1, r1 = mean_logps(disp, disp_pairs)
d_eval = dpo_eval(disp, ref, tok, disp_pairs, disp_cfg)
print(f"   train pair accuracy {d_eval['accuracy']:.2f}, margin {d_eval['margin']:+.2f}")
print(f"   mean log pi(chosen):   {c0:7.2f} -> {c1:7.2f}   ({'DOWN' if c1 < c0 else 'up'} by {abs(c1 - c0):.2f} nats)")
print(f"   mean log pi(rejected): {r0:7.2f} -> {r1:7.2f}   (down by {r0 - r1:.2f} nats)")
acc_disp = greedy_accuracy(disp, "after displacement run")
print(f"   greedy accuracy: SFT {acc_sft:.2f} -> {acc_disp:.2f}")
check(d_eval["accuracy"] >= 0.9, "the pairs are (almost) perfectly separated")
check(r1 - r0 < c1 - c0, "the rejected answers fell further than the chosen ones (the margin is what DPO optimises)")
if c1 < c0:
    print("   -> likelihood displacement: the CHOSEN answers became less likely, the gap just grew faster on the rejected side")
else:
    print("   -> no displacement this run: the chosen answers rose; try more steps or a higher LR (exercise 4)")

# --------------------------------------------------------------- (g) SimPO vs DPO
section(f"(g) SimPO vs DPO: same {len(op_train)} pairs, same {DPO_STEPS} steps")
simpo = copy.deepcopy(sft_model)
scfg = DPOConfig(steps=DPO_STEPS, batch_size=8, lr=5e-5, beta=2.0, gamma=0.5, warmup=5, loss="simpo",
                 log_every=max(1, DPO_STEPS // 5), seed=args.seed)
simpo_hist = dpo_train(simpo, None, tok, op_train, scfg)
s_after = dpo_eval(simpo, None, tok, op_val, scfg)
s_train = dpo_eval(simpo, None, tok, op_train, scfg)
acc_simpo = greedy_accuracy(simpo, "after SimPO")
print(f"   {'method':<6} {'val pair acc':>12} {'margin':>8} {'greedy acc':>11}")
print(f"   {'DPO':<6} {after['accuracy']:>12.2f} {after['margin']:>+8.3f} {acc_dpo:>11.2f}")
print(f"   {'SimPO':<6} {s_after['accuracy']:>12.2f} {s_after['margin']:>+8.3f} {acc_simpo:>11.2f}")
print("   (margins are not comparable across methods: DPO's is beta x a log-ratio vs the reference, SimPO's beta x an average per-token log-prob)")
print(f"   train pairs: DPO acc {train_after['accuracy']:.2f} | SimPO acc {s_train['accuracy']:.2f}")
check(s_train["accuracy"] > 0.5, "SimPO also separates the training pairs, with no reference model")

policy.save(run_path(f"lab17_dpo_{SIZE}.pt"), TOKENIZER_PATH, extra={"stage": "dpo", "pairs": len(op_train)})
print(f"\n   saved {run_path(f'lab17_dpo_{SIZE}.pt')}")

# ------------------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 4, figsize=(17, 3.8))
ax = axes[0]
ax.plot(rm_hist.step, rm_hist.accuracy, "o-", color="#2563eb", label="batch pair accuracy")
ax.plot(rm_hist.step, rm_hist.margin, "s--", color="#f59e0b", label="mean margin")
ax.set_xlabel("RM step"); ax.set_title("(b) reward model training"); ax.legend(fontsize=8)
ax = axes[1]
Ns = [r[0] for r in bon_rows]
ax.plot(Ns, [r[2] for r in bon_rows], "o-", color="#2563eb", label="gold acc, RM pick")
ax.plot(Ns, [r[4] for r in bon_rows], "s-", color="#dc2626", label="gold acc, rubric pick")
ax.plot(Ns, [r[5] for r in bon_rows], ":", color="#64748b", label="any correct among N")
ax.plot(Ns, [r[3] for r in bon_rows], "^--", color="#f59e0b", label="rubric score of pick (proxy)")
ax.set_xscale("log", base=2); ax.set_xlabel("N samples"); ax.set_title("(c) best-of-N: proxy vs gold"); ax.legend(fontsize=7)
ax = axes[2]
ax.plot(dpo_hist.step, dpo_hist.margin, "o-", color="#2563eb", label="DPO margin")
ax.plot(dpo_hist.step, dpo_hist.accuracy, "s--", color="#16a34a", label="DPO batch acc")
ax.plot(simpo_hist.step, simpo_hist.accuracy, "^--", color="#7c3aed", label="SimPO batch acc")
ax.set_xlabel("step"); ax.set_title("(e,g) DPO / SimPO training"); ax.legend(fontsize=8)
ax = axes[3]
ax.bar([0, 1], [c0, c1], color=["#94a3b8", "#16a34a"], width=0.6)
ax.bar([3, 4], [r0, r1], color=["#94a3b8", "#dc2626"], width=0.6)
ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["chosen\nbefore", "chosen\nafter", "rejected\nbefore", "rejected\nafter"], fontsize=8)
ax.set_ylabel("mean log pi(answer)"); ax.set_title(f"(f) displacement, {DISP_STEPS} steps")
fig.tight_layout()
savefig(fig, "lab17_reward_dpo.png")
done()
