"""Lab 20: distillation — logit KD, rejection-sampling SFT, and on-policy distillation.

Teacher: a `small` TinyLM fine-tuned on addition up to 20 (or runs/grpo_small.pt / sft_small.pt
if another lab produced one that is good at the task). Student: a `nano` TinyLM given a short SFT
warm-start on the same task (or runs/sft_nano.pt if present and usable).

(a) kd_logit_loss on one batch: forward KL between student and teacher next-token distributions.
(b) one on-policy distillation step *by hand*: the student samples, the teacher scores every
    token, and we print the per-token advantage table from figures/20_opd.svg with real numbers.
(c) two recipes at a matched budget: offline_distill (teacher samples -> verifier -> SFT) versus
    opd_train (student samples -> teacher grades -> REINFORCE). Accuracy before/after, the
    reverse-KL curve, wall-clock. --full adds the two-stage recipe (offline first, then OPD).

Warm-starts are trained once and cached under runs/lab20_*.pt (about 10 minutes on a laptop CPU
the first time); after that --quick is under a minute.

Run:  python3 labs/lab20_opd.py            (quick)
      python3 labs/lab20_opd.py --full
"""
from _common import setup, check, banner, section, savefig, done, plt

import copy
import os
import time

import torch

from llm import chat, tasks
from llm.distill import OPDConfig, kd_logit_loss, offline_distill, on_policy_distill_step, opd_train
from llm.evals import eval_tasks
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_tokenizer, run_path
from llm.rl import _special, masked_mean, response_mask, split_completion, token_logprobs
from llm.sft import SFTConfig, build_sft_dataset, sft_train
from llm.generate import sample_group

args = setup("Lab 20: distillation and on-policy distillation")

OPD_STEPS = 8 if args.quick else 60
GROUP, PROMPTS_PER_STEP, MAX_NEW = (4, 4, 16) if args.quick else (8, 8, 16)
OFFLINE_PROMPTS = 8 if args.quick else 60          # teacher samples 4 answers per prompt
OFFLINE_SFT_STEPS = 8 if args.quick else 60
N_EVAL = 30 if args.quick else 100

tok = get_tokenizer()
pad_id, end_id, _ = _special(tok)
student_train = tasks.make_examples(400, seed=20, tasks=["add"], max_value=20)    # student warm-start: ~270 of the 441 problems
teacher_train = tasks.make_examples(1200, seed=20, tasks=["add"], max_value=20)   # teacher: nearly all 441 problems, many times
opd_examples = tasks.make_examples(600, seed=22, tasks=["add"], max_value=20)     # prompts for (c)
held = tasks.make_examples(100, seed=21, tasks=["add"], max_value=20)[:N_EVAL]    # evaluation


def accuracy(model: TinyLM) -> float:
    return eval_tasks(model, tok, held, max_new_tokens=MAX_NEW).accuracy


# ------------------------------------------------------------------ warm-starts
def warm_start(preset_ckpt: str, out_name: str, steps: int, tag: str, examples) -> TinyLM:
    """SFT a base checkpoint on addition <= 20; cache the result under runs/."""
    path = run_path(out_name)
    if os.path.exists(path):
        return TinyLM.load(path)
    print(f"   [one-time] training the {tag}: {preset_ckpt} -> {out_name} ({steps} SFT steps on {len(examples)} examples, lr 1e-3)")
    model = TinyLM.load(run_path(preset_ckpt))
    t0 = time.perf_counter()
    sft_train(model, tok, examples, SFTConfig(steps=steps, batch_size=16, lr=1e-3, log_every=250,
                                              eval_every=500), val_examples=held[:30], verbose=True)
    model.save(path, TOKENIZER_PATH, extra={"stage": "sft", "tasks": ["add"], "max_value": 20, "steps": steps})
    print(f"   saved {path} after {time.perf_counter() - t0:.0f}s")
    return model


def first_usable(candidates: list[str], min_acc: float) -> tuple[TinyLM, str] | tuple[None, None]:
    """Reuse another lab's checkpoint if it exists and is good enough at the task."""
    for name in candidates:
        path = run_path(name)
        if os.path.exists(path):
            m = TinyLM.load(path)
            acc = accuracy(m)
            print(f"   found {name}: accuracy on add<=20 = {acc:.2f} ({'using it' if acc >= min_acc else 'too weak, skipping'})")
            if acc >= min_acc:
                return m, name
    return None, None


section("models: a small teacher and a nano student")
teacher, teacher_name = first_usable(["lab20_teacher_strong.pt", "grpo_small.pt", "sft_small.pt"], min_acc=0.8)
if teacher is None:
    teacher, teacher_name = warm_start("base_small.pt", "lab20_teacher_strong.pt", 1500, "teacher", teacher_train), "lab20_teacher_strong.pt"
student0, student_name = first_usable(["sft_nano.pt"], min_acc=0.1)
if student0 is None:
    student0, student_name = warm_start("base_nano.pt", "lab20_student_sft_nano.pt", 600, "student", student_train), "lab20_student_sft_nano.pt"
teacher.eval(); student0.eval()
t0 = time.perf_counter()
acc_teacher, acc_student0 = accuracy(teacher), accuracy(student0)
print(f"   teacher {teacher_name}: {teacher.num_params():,} params, accuracy {acc_teacher:.2f}")
print(f"   student {student_name}: {student0.num_params():,} params, accuracy {acc_student0:.2f}   "
      f"[{N_EVAL} held-out prompts, greedy, {time.perf_counter() - t0:.0f}s]")
check(acc_teacher > acc_student0 + 0.2, "the teacher is clearly better than the student on the task")

# ------------------------------------------------------------------ (a) logit KD
section("(a) logit distillation: forward KL on one batch of dataset text")
batch = build_sft_dataset(tok, held[:8])
x, y, m = chat.collate(batch, pad_id)                              # x (8, T-1), y (8, T-1), m (8, T-1)
with torch.no_grad():
    t_logits, _ = teacher(x)                                       # (8, T-1, V)
    s_logits, _ = student0(x)                                      # (8, T-1, V)
for temp in (1.0, 2.0, 4.0):
    kd = kd_logit_loss(s_logits, t_logits, m, temperature=temp).item()
    print(f"   tau = {temp:.0f}: KL(teacher || student) on assistant tokens = {kd:.3f} nats/token")
kd_self = kd_logit_loss(t_logits, t_logits, m).item()
kd_all = kd_logit_loss(s_logits, t_logits, torch.ones_like(m)).item()
print(f"   teacher vs itself: {kd_self:.6f}   |   over ALL positions (prompt included): {kd_all:.3f}")
check(kd_self < 1e-5, "KL(teacher || teacher) is zero: the loss measures disagreement only")
check(kd_logit_loss(s_logits, t_logits, m).item() > 0.05, "the student disagrees with the teacher on answer tokens")

# ------------------------------------------------------------------ (b) one OPD step by hand
section("(b) one on-policy step by hand: student samples, teacher grades each token")
ex = held[0]
prompt_ids = tok.encode(chat.render(ex.messages(with_answer=False)))
torch.manual_seed(args.seed)
ids = sample_group(student0, prompt_ids, 4, MAX_NEW, temperature=1.0, stop_ids=[end_id], pad_id=pad_id)  # (4, T)
mask = response_mask(ids, len(prompt_ids), pad_id, end_id)                                              # (4, T-1)
with torch.no_grad():
    logp_s = token_logprobs(student0, ids)                          # (4, T-1)
    logp_t = token_logprobs(teacher, ids)                           # (4, T-1)
adv = logp_t - logp_s                                               # per-token advantage
print(f"   prompt: {ex.prompt!r}   (answer {ex.meta['answer']})")
for g in range(ids.shape[0]):
    comp = tok.decode(split_completion(ids[g].tolist(), len(prompt_ids), pad_id, end_id))
    ok = tasks.verify(ex, comp) >= 1.0
    kl_g = masked_mean(-adv[g:g + 1], mask[g:g + 1]).item()
    print(f"   sample {g}: {comp!r:24s} {'correct' if ok else 'wrong  '} | reverse KL {kl_g:6.3f}")
g_show = next((g for g in range(ids.shape[0]) if tasks.verify(ex, tok.decode(split_completion(ids[g].tolist(), len(prompt_ids), pad_id, end_id))) < 1.0), 0)
print(f"   per-token table for sample {g_show}:")
print(f"   {'token':>12s} {'log pi_s':>9s} {'log pi_t':>9s} {'A_t':>8s}")
for t in range(ids.shape[1] - 1):
    if mask[g_show, t] > 0:
        print(f"   {tok.token_str(ids[g_show, t + 1].item())!r:>12s} {logp_s[g_show, t]:9.3f} {logp_t[g_show, t]:9.3f} {adv[g_show, t]:8.3f}")
worst = adv[g_show][mask[g_show] > 0].min().item()
print(f"   most negative advantage in this sample: {worst:.2f}  (the token the teacher would not have written)")

# ------------------------------------------------------------------ (c) two recipes, matched budget
section(f"(c) offline distillation vs on-policy distillation ({OPD_STEPS} OPD steps)")
results = {"student (before)": acc_student0, "teacher": acc_teacher}

student_off = copy.deepcopy(student0)
t0 = time.perf_counter()
off = offline_distill(student_off, teacher, tok, opd_examples[:OFFLINE_PROMPTS], n_samples=4,
                      sft_steps_n=OFFLINE_SFT_STEPS, max_new_tokens=MAX_NEW, lr=3e-4)
t_off = time.perf_counter() - t0
acc_off = accuracy(student_off)
results["offline (RS-SFT)"] = acc_off
print(f"   offline: teacher keep rate {off['keep_rate']:.2f} ({off['n_kept']}/{off['n_total']} samples), "
      f"{OFFLINE_SFT_STEPS} SFT steps, {t_off:.0f}s -> accuracy {acc_student0:.2f} -> {acc_off:.2f}")

student_opd = copy.deepcopy(student0)
cfg = OPDConfig(steps=OPD_STEPS, group_size=GROUP, prompts_per_step=PROMPTS_PER_STEP, max_new_tokens=MAX_NEW,
                lr=1e-4, seed=args.seed, log_every=max(1, OPD_STEPS // 6))
t0 = time.perf_counter()
hist = opd_train(student_opd, teacher, tok, opd_examples, cfg)
t_opd = time.perf_counter() - t0
acc_opd = accuracy(student_opd)
results["on-policy (OPD)"] = acc_opd
kl_curve = [h["reverse_kl"] for h in hist]
acc_curve = [h["accuracy"] for h in hist]
print(f"   OPD: {OPD_STEPS} steps x {GROUP * PROMPTS_PER_STEP} student samples, {t_opd:.0f}s "
      f"-> accuracy {acc_student0:.2f} -> {acc_opd:.2f}")
print(f"   reverse KL: {kl_curve[0]:.3f} (step 0) -> {min(kl_curve):.3f} (min) -> {kl_curve[-1]:.3f} (last)")
k = max(1, len(kl_curve) // 4)
samp_first, samp_last = sum(acc_curve[:k]) / k, sum(acc_curve[-k:]) / k
n_samp = k * GROUP * PROMPTS_PER_STEP
print(f"   accuracy of the student's own samples (T=1): first {k} steps {samp_first:.2f} -> last {k} steps {samp_last:.2f} "
      f"({n_samp} samples each; far less noisy than {N_EVAL} greedy items)")
# NOTE: on_policy_distill_step reports the mean of the *clipped* advantages, so this curve is the
# reverse KL with each token's contribution capped at OPDConfig.adv_clip nats (5.0), not the raw KL.
early = min(kl_curve[1:max(3, len(kl_curve) // 3) + 1])
check(early < kl_curve[0] - 0.05, f"reverse KL falls early in the run ({kl_curve[0]:.3f} at step 0 -> {early:.3f} within the first third): "
      "the student's own answers move toward the teacher's")
print(f"   (reported, not checked: first {k} steps {sum(kl_curve[:k]) / k:.3f} -> last {k} steps {sum(kl_curve[-k:]) / k:.3f}; "
      "at this scale the early fall is not sustained, see the chapter)")
print(f"   {'OPD raised' if samp_last > samp_first + 0.02 else 'OPD did not raise'} the accuracy of the student's own samples "
      f"({samp_first:.2f} -> {samp_last:.2f}); see the chapter for why a {student0.num_params():,}-parameter student "
      f"needs many more views of each sum than {OPD_STEPS} steps provide")

if args.full:
    student_two = copy.deepcopy(student0)
    t0 = time.perf_counter()
    offline_distill(student_two, teacher, tok, opd_examples[:OFFLINE_PROMPTS // 2], n_samples=4,
                    sft_steps_n=OFFLINE_SFT_STEPS // 2, max_new_tokens=MAX_NEW, lr=3e-4)
    cfg2 = OPDConfig(steps=OPD_STEPS // 2, group_size=GROUP, prompts_per_step=PROMPTS_PER_STEP,
                     max_new_tokens=MAX_NEW, lr=1e-4, seed=args.seed, log_every=10)
    hist2 = opd_train(student_two, teacher, tok, opd_examples, cfg2, verbose=False)
    t_two = time.perf_counter() - t0
    acc_two = accuracy(student_two)
    results["two-stage (offline then OPD)"] = acc_two
    print(f"   two-stage: {OFFLINE_SFT_STEPS // 2} SFT steps then {OPD_STEPS // 2} OPD steps, {t_two:.0f}s "
          f"-> accuracy {acc_two:.2f} | reverse KL at OPD start {hist2[0]['reverse_kl']:.3f}")

print("\n   summary (held-out accuracy, greedy):")
for name, acc in results.items():
    print(f"   {name:32s} {acc:.2f}")
best = max(acc_off, acc_opd)
student_best = student_opd if acc_opd >= acc_off else student_off
if args.full:
    check(best >= acc_student0 - 0.05, "the better recipe did not lose more than 0.05 held-out accuracy (distillation did no harm)")
student_best.save(run_path("lab20_student_opd.pt"), TOKENIZER_PATH,
                  extra={"stage": "distilled", "recipe": "opd" if acc_opd >= acc_off else "offline", "accuracy": best})
print(f"   saved runs/lab20_student_opd.pt (accuracy {best:.2f})")

# ------------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(kl_curve, color="#2563eb", lw=1.5)
ax.set_xlabel("OPD step"); ax.set_ylabel("reverse KL (nats/token)"); ax.set_title("student -> teacher reverse KL")
ax = axes[1]
ax.plot(acc_curve, color="#16a34a", lw=1.5)
ax.set_xlabel("OPD step"); ax.set_ylabel("accuracy of student samples (T=1)"); ax.set_ylim(0, 1); ax.set_title("sample accuracy during OPD")
ax = axes[2]
names = list(results); vals = [results[n] for n in names]
colors = ["#94a3b8", "#7c3aed", "#f59e0b", "#2563eb", "#16a34a"][:len(names)]
ax.bar(range(len(names)), vals, color=colors)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("held-out accuracy"); ax.set_title("before / after each recipe")
fig.tight_layout()
savefig(fig, "lab20_opd.png")
done()
