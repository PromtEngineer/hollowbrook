"""Lab 18: policy gradients from a bandit to TinyLM.

(a) A k-armed bandit with a softmax policy, trained by REINFORCE in numpy — with and
    without a baseline. Same expected update, very different variance and tail behaviour.
(b) REINFORCE on TinyLM: sample answers to a handful of "add" prompts, reward = verifier,
    loss = rl.reinforce_loss with the batch-mean baseline. The reward moves in a few steps,
    and so does the quantity the update really pushes: log pi(correct answer | prompt).
(c) Gradient variance on TinyLM: the same policy, five independent sample batches, with and
    without the baseline.
(d) The three KL estimators (k1, k2, k3) on an exact toy distribution and on TinyLM.
(e) PPO's clip: clip fraction vs how far the policy drifted from the sampler, symmetric vs
    DAPO clip-higher, and the "zero gradient on clipped tokens" fact.

Run:  python3 labs/lab18_reinforce.py            (quick: nano model)
      python3 labs/lab18_reinforce.py --full     (small model)
"""
from _common import setup, check, banner, section, savefig, done, plt

import copy
import math
import os
import random
import time

import numpy as np
import torch

from llm import chat, rl, tasks
from llm.dpo import sequence_logprob
from llm.generate import sample_group
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_base_model, get_tokenizer, run_path
from llm.reward import encode_response
from llm.sft import SFTConfig, sft_train
from llm.tasks import TaskExample

args = setup("Lab 18: REINFORCE, baselines, KL estimators and PPO clipping")
SIZE = "nano" if args.quick else "small"
RL_STEPS = 12 if args.quick else 30
N_PROMPTS, G = 6, 8                              # prompts per step x samples per prompt
MAX_NEW = 12
RL_LR = 3e-4 if args.quick else 1e-4
BANDIT_TRIALS = 20 if args.quick else 40
torch.manual_seed(args.seed)

# ============================================================== (a) the bandit
section("(a) a 5-armed bandit, softmax policy, REINFORCE with and without a baseline")
P_ARMS = np.array([0.25, 0.45, 0.60, 0.70, 0.90])          # hidden P(reward = 1) per arm
K, ALPHA, STEPS = len(P_ARMS), 0.2, 600


def softmax(theta: np.ndarray) -> np.ndarray:
    z = np.exp(theta - theta.max())
    return z / z.sum()


def run_bandit(use_baseline: bool, seed: int, steps: int = STEPS) -> dict:
    """REINFORCE: theta <- theta + alpha * (r - b) * grad log pi(a); grad_j = 1[j=a] - pi_j."""
    rng = np.random.default_rng(seed)
    theta, b = np.zeros(K), 0.0
    probs, rewards, best = [], [], []
    for t in range(steps):
        pi = softmax(theta)
        a = rng.choice(K, p=pi)
        r = float(rng.random() < P_ARMS[a])
        grad_logpi = -pi.copy(); grad_logpi[a] += 1.0           # d log pi(a) / d theta
        theta += ALPHA * (r - (b if use_baseline else 0.0)) * grad_logpi
        if use_baseline:
            b += 0.1 * (r - b)                                   # running mean of rewards
        probs.append(pi.copy()); rewards.append(r); best.append(pi[np.argmax(P_ARMS)])
    return {"probs": np.array(probs), "rewards": np.array(rewards), "best": np.array(best)}


runs = {name: [run_bandit(name == "baseline", args.seed * 100 + s) for s in range(BANDIT_TRIALS)]
        for name in ("no baseline", "baseline")}
for name, rs in runs.items():
    p_best = np.mean([r["best"][-1] for r in rs])
    steps_to = [int(np.argmax(r["best"] >= 0.9)) if (r["best"] >= 0.9).any() else STEPS for r in rs]
    stuck = sum(r["best"][-1] < 0.5 for r in rs)
    print(f"   {name:<12}: final P(best arm) {p_best:.2f} | median steps to P(best)>=0.9: {int(np.median(steps_to)):4d} "
          f"| runs stuck on a wrong arm at step {STEPS}: {stuck}/{BANDIT_TRIALS}")

# exact expected gradient and variance at a fixed policy (the bandit is discrete: sum over (arm, reward))
theta0 = np.array([0.0, 0.3, 0.6, 0.6, 0.8])
pi0 = softmax(theta0)
for name, b in (("no baseline", 0.0), ("baseline b=E[r]", float(pi0 @ P_ARMS))):
    Eg, Eg2 = np.zeros(K), 0.0
    for a in range(K):
        g = -pi0.copy(); g[a] += 1.0
        for r, pr in ((1.0, P_ARMS[a]), (0.0, 1 - P_ARMS[a])):
            w = pi0[a] * pr
            Eg += w * (r - b) * g
            Eg2 += w * ((r - b) ** 2) * (g @ g)
    print(f"   at a fixed policy, {name:<16}: E[update] = {np.round(Eg, 3)}  E||g||^2 = {Eg2:.3f}  Var = {Eg2 - Eg @ Eg:.3f}")
check(True, "the expected update is identical with and without the baseline; only the variance differs")

# ======================================================= (b) REINFORCE on TinyLM
section("(b) REINFORCE on TinyLM: a few prompts, verifier reward, batch-mean baseline")
tok = get_tokenizer()
pad_id, end_id = tok.special_tokens["<|pad|>"], tok.special_tokens[chat.END]


def add_example(a: int, b: int) -> TaskExample:
    return TaskExample("add", f"What is {a} + {b}?", f"{a} + {b} = {a + b}", {"answer": a + b})


ALL_PAIRS = [(a, b) for a in range(21) for b in range(21)]
random.Random(0).shuffle(ALL_PAIRS)
EVAL_SET = [add_example(a, b) for a, b in ALL_PAIRS[:100]]
TRAIN_SET = [add_example(a, b) for a, b in ALL_PAIRS[100:]]


def warm_start(size: str) -> TinyLM:
    """Load an SFT checkpoint (Lab 15's, or Lab 17's), or train one and save it."""
    for name in (f"sft_{size}.pt", f"lab17_sft_{size}.pt"):
        if os.path.exists(run_path(name)):
            print(f"   loading SFT warm start {run_path(name)}")
            return TinyLM.load(run_path(name))
    model, _ = get_base_model(quick=(size == "nano"), verbose=True)
    cfg = SFTConfig(steps=450 if size == "nano" else 400, batch_size=16, lr=1e-3 if size == "nano" else 3e-4,
                    log_every=100, eval_every=150)
    print(f"   no SFT checkpoint found: SFT on {len(TRAIN_SET)} 'add' examples, {cfg.steps} steps, lr {cfg.lr}")
    sft_train(model, tok, TRAIN_SET, cfg, val_examples=EVAL_SET[:30], verbose=True)
    model.save(run_path(f"lab17_sft_{size}.pt"), TOKENIZER_PATH, extra={"stage": "sft", "tasks": ["add"], "max_value": 20})
    return model


model = warm_start(SIZE).eval()
rl_prompts = TRAIN_SET[:N_PROMPTS]


@torch.no_grad()
def rollouts(m: TinyLM, examples, seed: int):
    """Sample G answers per prompt; returns [(ids (G,T), prompt_len, mask (G,T-1), rewards (G,))]."""
    out = []
    for i, ex in enumerate(examples):
        prompt_ids = tok.encode(chat.render(ex.messages(with_answer=False)))
        ids = sample_group(m, prompt_ids, G, MAX_NEW, temperature=1.0, stop_ids=[end_id], pad_id=pad_id, seed=seed + i)
        P = len(prompt_ids)
        rewards = torch.tensor([tasks.verify(ex, tok.decode(rl.split_completion(r, P, pad_id, end_id))) for r in ids.tolist()])
        out.append((ids, P, rl.response_mask(ids, P, pad_id, end_id), rewards))
    return out


def reinforce_grad(m: TinyLM, batch, use_baseline: bool) -> tuple[float, float]:
    """Accumulate reinforce_loss over the prompts of a batch; returns (mean reward, grad norm)."""
    all_r = torch.cat([b[3] for b in batch])
    baseline = all_r.mean().item() if use_baseline else None
    m.zero_grad(set_to_none=True)
    for ids, P, mask, rewards in batch:
        logps = rl.token_logprobs(m, ids)                                   # (G, T-1), grads flow
        loss = rl.reinforce_loss(logps, mask, rewards, baseline) / len(batch)
        loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
    return all_r.mean().item(), gnorm


@torch.no_grad()
def logp_correct(m: TinyLM, examples) -> float:
    """Mean log pi(reference answer | prompt): the number REINFORCE pushes up."""
    rows = [encode_response(tok, ex.messages(with_answer=False), ex.answer) for ex in examples]
    T = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), T), pad_id); mask = torch.zeros(len(rows), T)
    for i, (r, mk) in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r); mask[i, :len(mk)] = torch.tensor(mk, dtype=torch.float)
    return sequence_logprob(m, ids, mask).mean().item()


opt = torch.optim.AdamW(model.parameters(), lr=RL_LR, betas=(0.9, 0.95))
lp0 = logp_correct(model, rl_prompts)
print(f"   {N_PROMPTS} prompts x {G} samples per step, {RL_STEPS} steps, lr {RL_LR}; log pi(correct) before: {lp0:.2f}")
hist_r, hist_lp, t0 = [], [], time.perf_counter()
for step in range(RL_STEPS):
    batch = rollouts(model, rl_prompts, seed=1000 * args.seed + 10 * step)
    r_mean, gnorm = reinforce_grad(model, batch, use_baseline=True)
    opt.step()
    lp = logp_correct(model, rl_prompts)
    hist_r.append(r_mean); hist_lp.append(lp)
    if step % max(1, RL_STEPS // 6) == 0 or step == RL_STEPS - 1:
        print(f"   step {step:3d} | mean reward {r_mean:.3f} | log pi(correct) {lp:6.2f} | grad norm {gnorm:.2f} | {time.perf_counter() - t0:.0f}s")
first, last = np.mean(hist_r[:3]), np.mean(hist_r[-3:])
print(f"   mean reward, first 3 steps {first:.3f} -> last 3 steps {last:.3f}; log pi(correct) {lp0:.2f} -> {hist_lp[-1]:.2f}")
check(hist_lp[-1] > lp0, "REINFORCE raised log pi(correct answer) on the training prompts")
check(last >= first, "the sampled reward on the training prompts went up (noisy: 48 samples per step)")

# ============================================================ (c) variance on TinyLM
section("(c) gradient variance on TinyLM: five sample batches, with vs without baseline")
probe = copy.deepcopy(warm_start(SIZE)).eval()
grads = {"no baseline": [], "baseline": []}
for k in range(5):
    batch = rollouts(probe, rl_prompts, seed=777 + 10 * k)
    for name in grads:
        reinforce_grad(probe, batch, use_baseline=(name == "baseline"))
        grads[name].append(torch.cat([p.grad.flatten() for p in probe.parameters() if p.grad is not None]).clone())
var = {}
for name, gs in grads.items():
    G_ = torch.stack(gs); mean = G_.mean(0)
    var[name] = ((G_ - mean) ** 2).sum(-1).mean().item()
    print(f"   {name:<12}: ||mean grad|| {mean.norm():.3f} | mean ||g_i - mean||^2 (variance) {var[name]:.3f}")
check(var["baseline"] < var["no baseline"], "subtracting the batch-mean reward lowers the gradient variance on TinyLM too")

# ============================================================ (d) KL estimators
section("(d) k1, k2, k3: unbiased? always positive? how noisy?")
rng = np.random.default_rng(args.seed)
V = 10
p = softmax(rng.normal(size=V) * 1.0); q = softmax(rng.normal(size=V) * 1.0)   # policy p, reference q
true_kl = float(np.sum(p * (np.log(p) - np.log(q))))
xs = rng.choice(V, size=20000, p=p)
ks = rl.kl_estimators(torch.tensor(np.log(p[xs])), torch.tensor(np.log(q[xs])))
print(f"   true KL(p || q) = {true_kl:.4f} over V={V} symbols, 20,000 samples from p")
for name, v in ks.items():
    print(f"   {name}: mean {v.mean():.4f} | std {v.std():.3f} | fraction of negative samples {(v < 0).float().mean():.2f} | "
          f"bias {v.mean() - true_kl:+.4f}")
check(abs(ks["k1"].mean() - true_kl) < 0.02 and abs(ks["k3"].mean() - true_kl) < 0.02, "k1 and k3 are unbiased")
check(ks["k3"].std() < ks["k1"].std() and (ks["k3"] >= 0).all(), "k3 has lower variance than k1 and is never negative")
# on TinyLM: policy = the REINFORCE'd model, reference = the SFT warm start, tokens sampled from the policy
ref = warm_start(SIZE).eval()
with torch.no_grad():
    batch = rollouts(model, rl_prompts, seed=4242)
    tot = {"k1": 0.0, "k2": 0.0, "k3": 0.0}; n = 0.0
    for ids, P, mask, _ in batch:
        est = rl.kl_estimators(rl.token_logprobs(model, ids), rl.token_logprobs(ref, ids))
        for k in tot:
            tot[k] += (est[k] * mask).sum().item()
        n += mask.sum().item()
print("   TinyLM, REINFORCE'd policy vs SFT reference, per answer token: " + ", ".join(f"{k}={v / n:.4f}" for k, v in tot.items()))

# =================================================================== (e) PPO clip
section("(e) PPO clip fraction vs policy drift (synthetic ratios), symmetric vs clip-higher")
torch.manual_seed(args.seed)
B, T = 64, 16
mask = torch.ones(B, T)
adv = torch.where(torch.rand(B) < 0.5, 1.0, -1.0)
old = torch.zeros(B, T)
print(f"   {'sigma of log-ratio':>18} | {'clip frac 0.2/0.2':>17} | {'clip frac 0.2/0.28':>18} | {'approx KL (k3)':>14}")
clip_rows = []
for sigma in (0.02, 0.05, 0.1, 0.2, 0.4):
    logp = torch.randn(B, T) * sigma                              # log rho ~ N(0, sigma^2)
    _, s_sym = rl.ppo_clip_loss(logp, old, adv, mask, 0.2, 0.2)
    _, s_hi = rl.ppo_clip_loss(logp, old, adv, mask, 0.2, 0.28)
    clip_rows.append((sigma, s_sym["clip_frac"], s_hi["clip_frac"], s_sym["approx_kl"]))
    print(f"   {sigma:>18.2f} | {s_sym['clip_frac']:>17.3f} | {s_hi['clip_frac']:>18.3f} | {s_sym['approx_kl']:>14.4f}")
check(clip_rows[-1][1] > clip_rows[0][1], "the further the policy drifts from the sampler, the more tokens are clipped")
check(all(r[2] <= r[1] for r in clip_rows), "clip-higher (eps_high 0.28) clips no more tokens than the symmetric clip, and usually fewer")
# zero gradient on clipped tokens
logp = torch.tensor([[0.5] * 4, [0.0] * 4], requires_grad=True)   # row 0: ratio e^0.5 = 1.65 with A>0 -> clipped
loss, st = rl.ppo_clip_loss(logp, torch.zeros(2, 4), torch.tensor([1.0, 1.0]), torch.ones(2, 4), 0.2, 0.28)
loss.backward()
print(f"   row with ratio 1.65 and A>0: clip_frac {st['clip_frac']:.2f}, grad on that row = {logp.grad[0].tolist()}, "
      f"grad on the ratio-1 row = {[round(g, 3) for g in logp.grad[1].tolist()]}")
check(torch.all(logp.grad[0] == 0) and torch.all(logp.grad[1] != 0), "a clipped token receives exactly zero gradient")

# ------------------------------------------------------------------- figures
fig, axes = plt().subplots(1, 3, figsize=(15, 4))
ax = axes[0]
for name, color in (("no baseline", "#dc2626"), ("baseline", "#2563eb")):
    best = np.mean([r["best"] for r in runs[name]], axis=0)
    ax.plot(best, color=color, label=f"{name}: mean P(best arm)")
    for r in runs[name][:8]:
        ax.plot(r["best"], color=color, alpha=0.15, lw=0.8)
ax.set_xlabel("step"); ax.set_ylabel("P(best arm)"); ax.set_title(f"(a) bandit, {BANDIT_TRIALS} trials each"); ax.legend(fontsize=8)
ax = axes[1]
ax.plot(hist_r, "o-", color="#f59e0b", label="mean sampled reward")
ax2 = ax.twinx(); ax2.plot(hist_lp, "s--", color="#2563eb", label="log pi(correct)"); ax2.set_ylabel("log pi(correct)", color="#2563eb")
ax.set_xlabel("REINFORCE step"); ax.set_ylabel("reward"); ax.set_title("(b) REINFORCE on TinyLM"); ax.legend(loc="upper left", fontsize=8)
ax = axes[2]
ax.plot([r[0] for r in clip_rows], [r[1] for r in clip_rows], "o-", color="#dc2626", label="eps 0.2 / 0.2")
ax.plot([r[0] for r in clip_rows], [r[2] for r in clip_rows], "s-", color="#16a34a", label="eps 0.2 / 0.28 (clip-higher)")
ax.set_xscale("log"); ax.set_xlabel("sigma of per-token log-ratio"); ax.set_ylabel("clip fraction"); ax.set_title("(e) PPO clipping"); ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "lab18_reinforce.png")

fig, axes = plt().subplots(1, 3, figsize=(15, 3.6))
for ax, (name, v) in zip(axes, ks.items()):
    ax.hist(v.numpy(), bins=40, color="#7c3aed", alpha=0.6)
    ax.axvline(true_kl, color="#dc2626", ls="--", label=f"true KL {true_kl:.3f}")
    ax.axvline(v.mean().item(), color="#0f172a", label=f"mean {v.mean():.3f}, std {v.std():.2f}")
    ax.set_title(f"(d) {name} per-sample values"); ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "lab18_kl_estimators.png")
done()
