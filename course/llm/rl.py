"""Reinforcement learning for language models (Chapters 18, 19 and 21).

Chapter 18 — the building blocks:  ``token_logprobs``, ``response_mask``,
    ``reinforce_loss``, ``kl_estimators``, ``ppo_clip_loss``.
Chapter 19 — GRPO with verifiable rewards and its 2025–2026 fixes:
    ``grpo_advantages``, ``GRPOConfig``, ``rollout_group``, ``grpo_step``, ``grpo_train``,
    plus ``gspo_ratio`` / ``gspo_loss`` for the sequence-level variant.
Chapter 21 — multi-turn (agentic) RL: ``multi_turn_rollout``, ``multi_turn_grpo_step`` and a
    toy ``CalculatorEnv``.

RL vocabulary, translated to LLMs
---------------------------------
* **state**  = the prompt plus the tokens generated so far
* **action** = the next token
* **policy** π_θ = the model: π_θ(token | prefix) is a row of its softmax
* **reward** R = one number for the whole answer (e.g. 1 if the answer is correct)

Because the reward arrives only at the end and the "environment" (appending a token) is
deterministic, an LLM episode is a *bandit* problem: sample an answer, get a score, make the
scored answer more (R high) or less (R low) likely. Every loss below is a variant of that.

Notation (as everywhere in the course): B batch, T sequence length, G group size,
π_θ the policy being trained, π_old the policy that produced the samples, π_ref the frozen
reference model.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from . import chat, tasks
from .generate import generate_ids, sample_group
from .model import TinyLM
from .tasks import TaskExample
from .tokenizer import BPETokenizer


# =============================================================== Chapter 18
# ------------------------------------------------------------- log-probs
def logprobs_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """log π(target_t | prefix) for every position. logits (B, T, V), targets (B, T) -> (B, T).

    Read: "take the log-softmax over the vocabulary, then pick out the entry of the token
    that was actually there."
    """
    logp = F.log_softmax(logits.float(), dim=-1)                  # (B, T, V)
    return logp.gather(-1, targets[..., None]).squeeze(-1)         # (B, T)


def entropy_from_logits(logits: Tensor) -> Tensor:
    """H = -Σ_v p(v) log p(v) per position. logits (B, T, V) -> (B, T).

    High entropy = the model is unsure what comes next. RL runs watch this number: when
    it collapses to ~0 the policy has stopped exploring (Chapter 19, "entropy collapse").
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def token_logprobs(model: TinyLM, ids: Tensor) -> Tensor:
    """log π_θ(ids[:, t] | ids[:, :t]) for t = 1..T-1.  ids (B, T) -> (B, T-1).

    Position t-1 of the result scores token t of the input (the usual "shift by one" of
    next-token prediction). Gradients flow, so call under ``torch.no_grad()`` for π_old / π_ref.
    """
    logits, _ = model(ids[:, :-1])                                 # (B, T-1, V)
    return logprobs_from_logits(logits, ids[:, 1:])                # (B, T-1)


def response_mask(ids: Tensor, prompt_len: int, pad_id: int, end_id: int) -> Tensor:
    """Which positions of ``token_logprobs`` belong to the *generated* answer. -> (B, T-1) float.

    mask[b, t-1] = 1  iff token t was generated (t >= prompt_len), is not padding, and comes
    no later than the first ``<|end|>`` of that row (the ``<|end|>`` itself counts: the model
    must learn to stop). Everything in the prompt and everything after the answer ended is 0.
    """
    B, T = ids.shape
    gen = ids[:, prompt_len:]                                      # (B, T - prompt_len)
    is_end = gen == end_id
    is_pad = gen == pad_id
    stops = (is_end | is_pad).long()
    # number of stop tokens strictly *before* each position
    stops_before = stops.cumsum(-1) - stops
    keep = (stops_before == 0) & ~is_pad                           # up to and incl. first <|end|>
    mask = torch.zeros(B, T, dtype=torch.float, device=ids.device)
    mask[:, prompt_len:] = keep.float()
    return mask[:, 1:]                                             # align with token_logprobs


def masked_mean(x: Tensor, mask: Tensor, token_level: bool = True) -> Tensor:
    """Average ``x`` (B, T) over the positions where mask = 1.

    token_level=True : one big average over every token in the batch (DAPO's choice: a long
                       answer contributes more tokens, so each *token* has equal weight).
    token_level=False: average within each sequence first, then across sequences (the original
                       GRPO paper: each *sequence* has equal weight, which under-weights the
                       tokens of long answers).
    """
    if token_level:
        return (x * mask).sum() / mask.sum().clamp(min=1.0)
    per_seq = (x * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)    # (B,)
    return per_seq.mean()


# ------------------------------------------------------------- REINFORCE
def reinforce_loss(logps: Tensor, mask: Tensor, rewards: Tensor,
                   baseline: Optional[Tensor | float] = None) -> Tensor:
    """REINFORCE (policy gradient) loss for whole-sequence rewards.

        L = - mean_b [ (R_b - b) · Σ_t mask_bt · log π_θ(token_bt | prefix) ]

    Read: "for each sampled answer, add up the log-probabilities of its tokens; multiply by
    how much better than expected the answer was; push the parameters to make good answers
    more likely and bad ones less likely." The baseline ``b`` does not change the expected
    gradient but shrinks its variance (Chapter 18); GRPO's baseline is the group mean.

    logps (B, T-1), mask (B, T-1), rewards (B,), baseline scalar or (B,).
    """
    seq_logp = (logps * mask).sum(-1)                              # (B,)  log π(answer)
    adv = rewards.float() - (0.0 if baseline is None else baseline)
    return -(adv.detach() * seq_logp).mean()


# ---------------------------------------------------------------- KL
def kl_estimators(logp: Tensor, ref_logp: Tensor) -> dict[str, Tensor]:
    """Per-token Monte-Carlo estimators of KL(π_θ || π_ref) from samples drawn from π_θ.

    With r = π_ref(x) / π_θ(x) = exp(ref_logp - logp) (Schulman, 2020):
        k1 = -log r          = logp - ref_logp        unbiased, high variance, can be negative
        k2 = ½ (log r)²                               biased, low variance, always ≥ 0
        k3 = (r - 1) - log r = exp(ref-logp) - (ref-logp) - 1   unbiased AND always ≥ 0

    Read k3 as: "how much more likely the reference finds this token than we do, minus the
    log of that ratio" — zero when the two models agree, positive otherwise. GRPO uses k3.
    """
    d = ref_logp - logp                                            # log r
    return {"k1": -d, "k2": 0.5 * d * d, "k3": d.exp() - d - 1.0}


# ---------------------------------------------------------------- PPO
def ppo_clip_loss(logp: Tensor, old_logp: Tensor, adv: Tensor, mask: Tensor,
                  eps_low: float = 0.2, eps_high: float = 0.2, token_level: bool = True
                  ) -> tuple[Tensor, dict]:
    """PPO's clipped surrogate objective, per token.

        ρ_t = π_θ(a_t) / π_old(a_t) = exp(logp_t - old_logp_t)
        L_t = - min( ρ_t · A,  clip(ρ_t, 1 - ε_low, 1 + ε_high) · A )

    Read: "move probability toward tokens of good answers (A > 0) and away from tokens of bad
    answers (A < 0), but once a token's probability has already moved by more than ε relative
    to the sampling policy, stop — the min() makes the gradient exactly zero there."
    ε_high > ε_low is DAPO's *clip-higher*: allow rare tokens to grow more than they shrink,
    which fights entropy collapse.

    logp/old_logp/mask (B, T-1); adv (B,) one advantage per sequence, or (B, T-1) per token.
    Returns (loss, stats). stats["clip_frac"] is the fraction of answer tokens whose gradient
    the clip zeroed; stats["approx_kl"] is KL(π_old || π_θ) via k3 (how far we have moved).
    """
    if adv.dim() == 1:
        adv = adv[:, None]                                         # (B, 1) broadcasts over T
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()                                        # ρ_t   (B, T-1)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * adv
    per_token = -torch.min(unclipped, clipped)
    loss = masked_mean(per_token, mask, token_level)
    with torch.no_grad():
        is_clipped = (clipped < unclipped).float()                # the clipped branch won
        stats = {
            "clip_frac": masked_mean(is_clipped, mask).item(),
            "approx_kl": masked_mean(kl_estimators(logp, old_logp)["k3"], mask).item(),
            "ratio_mean": masked_mean(ratio, mask).item(),
        }
    return loss, stats


def gspo_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor:
    """GSPO's *sequence-level* importance ratio (Qwen, 2025). -> (B,)

        s_b = exp( (1/|y_b|) Σ_t mask_bt (logp_bt - old_logp_bt) )

    Read: "the geometric mean, over the answer's tokens, of the per-token ratios." PPO/GRPO
    clip each token's ratio separately, which is noisy (one odd token can be clipped while its
    neighbours are not); GSPO clips one number per answer, matching the fact that the reward
    is also one number per answer. Because s_b is a mean it moves much less than a single ρ_t,
    so GSPO uses tiny ε (≈ 3e-4) instead of 0.2.
    """
    return ((logp - old_logp) * mask).sum(-1).div(mask.sum(-1).clamp(min=1.0)).exp()


def gspo_loss(logp: Tensor, old_logp: Tensor, adv: Tensor, mask: Tensor,
              eps_low: float = 3e-4, eps_high: float = 4e-4) -> tuple[Tensor, dict]:
    """Clipped surrogate with the GSPO sequence ratio: L = - mean_b min(s_b A_b, clip(s_b) A_b)."""
    s = gspo_ratio(logp, old_logp, mask)                           # (B,)
    unclipped = s * adv
    clipped = s.clamp(1.0 - eps_low, 1.0 + eps_high) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        stats = {"clip_frac": (clipped < unclipped).float().mean().item(),
                 "approx_kl": masked_mean(kl_estimators(logp, old_logp)["k3"], mask).item(),
                 "ratio_mean": s.mean().item()}
    return loss, stats


# =============================================================== Chapter 19
def grpo_advantages(rewards: Tensor, normalize_std: bool = True, eps: float = 1e-6) -> Tensor:
    """Group-relative advantages. rewards (G,) for G answers to the *same* prompt -> (G,).

        A_i = (r_i - mean(r)) / (std(r) + ε)          GRPO (DeepSeek, 2024)
        A_i =  r_i - mean(r)                           Dr. GRPO (normalize_std=False)

    Read: "an answer is good if it beat its siblings." The group mean is the baseline (no
    value network needed). Dr. GRPO drops the division by std because it over-weights prompts
    that are almost always right or almost always wrong (tiny std → huge advantages).
    """
    r = rewards.float()
    adv = r - r.mean()
    if normalize_std:
        adv = adv / (r.std(unbiased=False) + eps)
    return adv


@dataclass
class GRPOConfig:
    # Defaults were tuned on the nano model + "add" (max_value=20) task: 19% -> 28% greedy
    # accuracy in 120 steps (~5 min on an idle laptop CPU). A bigger group (16) leaves fewer
    # all-wrong groups to skip; lr above ~2e-4 made this tiny model *worse* on held-out prompts.
    group_size: int = 16            # G answers sampled per prompt
    steps: int = 120                # optimizer steps
    prompts_per_step: int = 4       # prompts per step -> G * prompts_per_step samples per step
    max_new_tokens: int = 24
    temperature: float = 1.0        # sample at 1.0: the loss assumes samples come from π_old
    lr: float = 1e-4                # small model, small data: higher than the 1e-6 of real runs
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28     # DAPO clip-higher (0.28 > 0.2)
    kl_coef: float = 0.0            # DAPO / Dr. GRPO drop the KL term; set > 0 and pass ref
    token_level_loss: bool = True   # DAPO token-level average (False = per-sequence mean)
    normalize_std: bool = True      # False = Dr. GRPO
    dynamic_sampling: bool = True   # DAPO: skip groups whose rewards are all equal
    sequence_ratio: bool = False    # True = GSPO sequence-level ratio instead of per-token
    ppo_epochs: int = 1             # >1 re-uses the samples; then the clip actually bites
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0
    log_every: int = 1
    entropy_log: bool = True        # compute the policy entropy at answer tokens (for logging)


@dataclass
class Rollout:
    """One prompt, G sampled answers, and their rewards."""
    example: TaskExample
    ids: Tensor                     # (G, T) prompt + answer (+ padding)
    prompt_len: int
    rewards: Tensor                 # (G,)
    completions: list[str]          # decoded answers (without <|end|>)
    mask: Tensor                    # (G, T-1) answer positions, aligned with token_logprobs
    advantages: Optional[Tensor] = None   # (G,) filled in by grpo_step


RewardFn = Callable[[TaskExample, str, list[int]], float]


def default_reward(example: TaskExample, completion: str, completion_ids: list[int]) -> float:
    """Verifiable reward: 1 for a correct answer, +0.1 for the expected format (max 1.1)."""
    return tasks.verify(example, completion) + 0.1 * tasks.format_reward(example, completion)


def _special(tok: BPETokenizer) -> tuple[int, int, int]:
    """(pad_id, end_id, assistant_id) from the tokenizer's chat tokens."""
    return tok.special_tokens["<|pad|>"], tok.special_tokens["<|end|>"], tok.special_tokens["<|assistant|>"]


def split_completion(row: Sequence[int], prompt_len: int, pad_id: int, end_id: int) -> list[int]:
    """The generated ids of one row, cut at the first <|end|> (excluded) and without padding."""
    out = []
    for t in list(row)[prompt_len:]:
        if t == end_id or t == pad_id:
            break
        out.append(t)
    return out


@torch.no_grad()
def rollout_group(model: TinyLM, tok: BPETokenizer, example: TaskExample, cfg: GRPOConfig,
                  reward_fn: Optional[RewardFn] = None, seed: Optional[int] = None) -> Rollout:
    """Sample G answers to one prompt and score each with the verifier.

    The prompt is the chat template *without* the answer, ending in ``<|assistant|>``;
    the model continues from there. Generation stops at ``<|end|>``.
    """
    reward_fn = reward_fn or default_reward
    pad_id, end_id, _ = _special(tok)
    prompt_ids = chat.encode_chat(tok, example.messages(with_answer=False))   # safe encoding (Ch. 14)
    ids = sample_group(model, prompt_ids, cfg.group_size, cfg.max_new_tokens, cfg.temperature,
                       stop_ids=[end_id], pad_id=pad_id, seed=seed)          # (G, T)
    P = len(prompt_ids)
    completions, rewards = [], []
    for row in ids.tolist():
        comp_ids = split_completion(row, P, pad_id, end_id)
        text = tok.decode(comp_ids)
        completions.append(text)
        rewards.append(float(reward_fn(example, text, comp_ids)))
    return Rollout(example, ids, P, torch.tensor(rewards), completions,
                   response_mask(ids, P, pad_id, end_id))


def collate_rollouts(rollouts: Sequence[Rollout], pad_id: int) -> tuple[Tensor, Tensor, Tensor]:
    """Stack groups of different lengths into one batch: ids (N, T), mask (N, T-1), adv (N,)."""
    T = max(r.ids.shape[1] for r in rollouts)
    ids = torch.cat([F.pad(r.ids, (0, T - r.ids.shape[1]), value=pad_id) for r in rollouts])
    mask = torch.cat([F.pad(r.mask, (0, T - 1 - r.mask.shape[1]), value=0.0) for r in rollouts])
    adv = torch.cat([r.advantages for r in rollouts])
    return ids, mask, adv


def policy_update(model: TinyLM, ref: Optional[TinyLM], ids: Tensor, mask: Tensor, adv: Tensor,
                  cfg: GRPOConfig, optimizer: torch.optim.Optimizer) -> dict:
    """The GRPO/PPO parameter update on an already-collated batch. Returns loss stats.

    Steps: (1) log-probs of the sampled tokens under π_θ, (2) clipped surrogate against
    π_old, (3) optional k3 KL penalty toward π_ref, (4) backward, clip gradients, step.

    A detail beginners ask about: with ``ppo_epochs = 1`` the policy that *sampled* the
    answers is the policy being updated, so π_old = π_θ, every ratio is exactly 1 and the clip
    never activates. We therefore skip the extra π_old forward pass and use ``logp.detach()``
    (identical numbers). The clip only matters when the same samples are re-used for several
    epochs (``ppo_epochs > 1``) or, in large systems, when the sampler runs a few steps behind
    the trainer ("off-policy lag").
    """
    model.train()
    old_logp = ref_logp = None
    if cfg.ppo_epochs > 1:
        with torch.no_grad():
            old_logp = token_logprobs(model, ids)                  # π_old, frozen numbers
    if ref is not None and cfg.kl_coef > 0:
        with torch.no_grad():
            ref_logp = token_logprobs(ref, ids)                    # π_ref
    stats: dict = {}
    for _ in range(cfg.ppo_epochs):
        logits, _ = model(ids[:, :-1])                             # (N, T-1, V)
        logp = logprobs_from_logits(logits, ids[:, 1:])            # (N, T-1)
        old = old_logp if old_logp is not None else logp.detach()
        if cfg.sequence_ratio:
            loss, stats = gspo_loss(logp, old, adv, mask, cfg.clip_eps_low, cfg.clip_eps_high)
        else:
            loss, stats = ppo_clip_loss(logp, old, adv, mask, cfg.clip_eps_low, cfg.clip_eps_high,
                                        token_level=cfg.token_level_loss)
        if ref_logp is not None:
            kl = masked_mean(kl_estimators(logp, ref_logp)["k3"], mask, cfg.token_level_loss)
            loss = loss + cfg.kl_coef * kl
            stats["kl_ref"] = kl.item()
        if cfg.entropy_log:
            with torch.no_grad():
                stats["entropy"] = masked_mean(entropy_from_logits(logits), mask).item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()
        stats["loss"] = loss.item()
    model.eval()
    return stats


def grpo_step(model: TinyLM, ref: Optional[TinyLM], tok: BPETokenizer, examples: Sequence[TaskExample],
              cfg: GRPOConfig, optimizer: torch.optim.Optimizer,
              reward_fn: Optional[RewardFn] = None) -> dict:
    """One GRPO step: rollout G answers per prompt, group-relative advantages, clipped update.

        L = - (1/Σ tokens) Σ_prompts Σ_i Σ_t min(ρ_it A_i, clip(ρ_it) A_i)  [+ β · KL_k3(π_θ‖π_ref)]

    Read: "every token of an above-average answer gets pushed up, every token of a
    below-average answer gets pushed down, by an amount proportional to how far from the
    group mean the answer was."

    Options from the 2025–2026 papers (all in ``GRPOConfig``):
    * dynamic sampling (DAPO): a group whose rewards are all equal has A = 0 for every answer
      and contributes nothing; we skip it and report the fraction skipped.
    * token-level loss (DAPO) vs sequence-mean (original GRPO): see ``masked_mean``.
    * clip-higher (DAPO): ε_high > ε_low.
    * normalize_std=False (Dr. GRPO), kl_coef=0 (DAPO, Dr. GRPO), sequence_ratio (GSPO).
    """
    pad_id, _, _ = _special(tok)
    t0 = time.perf_counter()
    rollouts = [rollout_group(model, tok, ex, cfg, reward_fn) for ex in examples]
    t_rollout = time.perf_counter() - t0

    all_rewards = torch.cat([r.rewards for r in rollouts])
    all_lens = torch.cat([r.mask.sum(-1) for r in rollouts])
    kept = []
    for r in rollouts:
        r.advantages = grpo_advantages(r.rewards, cfg.normalize_std)
        if cfg.dynamic_sampling and r.rewards.std(unbiased=False) < 1e-8:
            continue                                               # all equal -> zero signal
        kept.append(r)
    stats = {
        "reward": all_rewards.mean().item(),
        "accuracy": (all_rewards >= 1.0).float().mean().item(),
        "resp_len": all_lens.mean().item(),
        "skipped_frac": 1.0 - len(kept) / len(rollouts),
        "n_tokens": int(sum(r.mask.sum().item() for r in kept)),
        "t_rollout": t_rollout,
    }
    if not kept:                                                   # nothing to learn from
        stats.update(loss=0.0, clip_frac=0.0, approx_kl=0.0, entropy=float("nan"), grad_norm=0.0)
        return stats
    ids, mask, adv = collate_rollouts(kept, pad_id)
    stats.update(policy_update(model, ref, ids, mask, adv, cfg, optimizer))
    stats["t_step"] = time.perf_counter() - t0
    stats["rollouts"] = rollouts                                   # for inspection / printing
    return stats


def make_optimizer(model: TinyLM, cfg: GRPOConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                             weight_decay=cfg.weight_decay)


def grpo_train(model: TinyLM, tok: BPETokenizer, examples: Sequence[TaskExample], cfg: GRPOConfig,
               ref: Optional[TinyLM] = None, reward_fn: Optional[RewardFn] = None,
               verbose: bool = True, on_step: Optional[Callable[[int, dict], None]] = None,
               optimizer: Optional[torch.optim.Optimizer] = None) -> list[dict]:
    """Run ``cfg.steps`` GRPO steps, cycling through ``examples``. Returns per-step stats."""
    torch.manual_seed(cfg.seed)
    optimizer = optimizer or make_optimizer(model, cfg)
    history: list[dict] = []
    for step in range(cfg.steps):
        start = (step * cfg.prompts_per_step) % len(examples)
        batch = [examples[(start + i) % len(examples)] for i in range(cfg.prompts_per_step)]
        st = grpo_step(model, ref, tok, batch, cfg, optimizer, reward_fn)
        st["step"] = step
        history.append({k: v for k, v in st.items() if k != "rollouts"})
        if verbose and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"step {step:4d} | reward {st['reward']:.3f} | acc {st['accuracy']:.2f} | "
                  f"len {st['resp_len']:5.1f} | clip {st.get('clip_frac', 0):.2f} | "
                  f"kl {st.get('approx_kl', 0):.4f} | H {st.get('entropy', float('nan')):.2f} | "
                  f"skipped {st['skipped_frac']:.2f} | {st.get('t_step', 0):.1f}s")
        if on_step is not None:
            on_step(step, st)
    return history


# =============================================================== Chapter 21
# Multi-turn (agentic) RL: the episode is a *trajectory* of assistant turns and tool results.
# The environment protocol is deliberately tiny:
#
#     obs = env.reset()                       -> the first user message (str)
#     obs, reward, done = env.step(text)      -> what the assistant's turn caused
#
# ``obs`` is a tool result to feed back (or None when the episode is over), ``reward`` is
# a number for that turn (we sum them into a trajectory reward), ``done`` ends the episode.

@dataclass
class Turn:
    role: str                       # "user" | "assistant" | "tool_result"
    text: str
    ids: list[int]                  # the tokens this turn added to the context


@dataclass
class Trajectory:
    ids: Tensor                     # (T,) the whole conversation as one token sequence
    mask: Tensor                    # (T-1,) 1 only on tokens the *model* generated
    reward: float                   # sum of per-turn rewards
    turns: list[Turn]
    done: bool                      # False if we hit max_turns or the context limit
    n_tool_calls: int = 0


def encode_turn(tok: BPETokenizer, role: str, content: str) -> list[int]:
    """``<|role|>content<|end|>`` as ids (content is untrusted: specials are not parsed)."""
    return ([tok.special_tokens[chat.ROLE_TOKENS[role]]]
            + tok.encode(content, allowed_special=False) + [tok.special_tokens[chat.END]])


@torch.no_grad()
def multi_turn_rollout(model: TinyLM, tok: BPETokenizer, env, cfg: GRPOConfig, max_turns: int = 3,
                       seed: Optional[int] = None) -> Trajectory:
    """Play one episode: the model writes a turn, the environment answers, until done.

    Every turn starts with the harness writing ``<|assistant|>``; the model then generates up
    to ``<|end|>``. A turn that contains a tool call (``chat.parse_tool_call``; the model emits
    ``<|tool_call|>`` as its first token, exactly as in the SFT layout) is sent to the environment,
    whose observation comes back as a ``<|tool_result|>`` turn; the model then writes again.
    The returned mask covers ONLY tokens the model generated (assistant text and its
    ``<|end|>``), so tool results — text the model was *given* — are never trained on.
    """
    pad_id, end_id, assistant_id = _special(tok)
    bos = tok.special_tokens[chat.BOS]
    system = getattr(env, "system_prompt", tasks.SYSTEM_PROMPT)
    user = env.reset()
    ids: list[int] = [bos] + encode_turn(tok, "system", system) + encode_turn(tok, "user", user)
    gen_mask = [0] * len(ids)
    turns = [Turn("user", user, list(ids))]
    reward, done, n_calls = 0.0, False, 0
    device = next(model.parameters()).device
    for _ in range(max_turns):
        ids.append(assistant_id); gen_mask.append(0)               # generation prompt (given)
        budget = min(cfg.max_new_tokens, model.cfg.max_seq_len - len(ids))
        if budget <= 0:
            break                                                  # context is full
        out = generate_ids(model, torch.tensor([ids], device=device), budget, cfg.temperature,
                           stop_ids=[end_id], pad_id=pad_id, seed=seed)
        new = out[0, len(ids):].tolist()
        new = new[:new.index(end_id) + 1] if end_id in new else new   # keep the <|end|>
        ids += new; gen_mask += [1] * len(new)
        text = tok.decode(new)                                     # includes "<|end|>" if any
        turns.append(Turn("assistant", text, new))
        obs, r, done = env.step(text)
        reward += r
        if chat.parse_tool_call(text) is not None:
            n_calls += 1
        if done or obs is None:
            break
        obs_ids = encode_turn(tok, "tool_result", obs)
        ids += obs_ids; gen_mask += [0] * len(obs_ids)
        turns.append(Turn("tool_result", obs, obs_ids))
    # A tool result appended after the last turn may overflow the context; generated tokens
    # always fit (see ``budget``), so truncating only drops *given* tokens (mask = 0).
    L = model.cfg.max_seq_len
    return Trajectory(torch.tensor(ids[:L]), torch.tensor(gen_mask[1:L], dtype=torch.float),
                      reward, turns, done, n_calls)


def multi_turn_grpo_step(model: TinyLM, tok: BPETokenizer, env_factory: Callable[[int], object],
                         cfg: GRPOConfig, optimizer: torch.optim.Optimizer, n_tasks: Optional[int] = None,
                         max_turns: int = 3, ref: Optional[TinyLM] = None, task_offset: int = 0) -> dict:
    """GRPO over trajectories: G episodes per task, advantages from trajectory rewards.

    ``env_factory(i)`` must return an environment for task ``i`` whose ``reset()`` gives the
    same problem every time, so the G trajectories of a group are comparable. Credit
    assignment is the crudest possible — every generated token of a trajectory gets the
    trajectory's advantage (Chapter 21 discusses turn-level alternatives).
    """
    pad_id, _, _ = _special(tok)
    n_tasks = n_tasks or cfg.prompts_per_step
    t0 = time.perf_counter()
    groups: list[list[Trajectory]] = []
    for i in range(n_tasks):
        env = env_factory(task_offset + i)
        groups.append([multi_turn_rollout(model, tok, env, cfg, max_turns) for _ in range(cfg.group_size)])
    rewards = torch.tensor([[t.reward for t in g] for g in groups])            # (n_tasks, G)
    kept_ids, kept_masks, kept_adv = [], [], []
    for g, r in zip(groups, rewards):
        if cfg.dynamic_sampling and r.std(unbiased=False) < 1e-8:
            continue
        adv = grpo_advantages(r, cfg.normalize_std)
        for t, a in zip(g, adv):
            kept_ids.append(t.ids); kept_masks.append(t.mask); kept_adv.append(a)
    stats = {
        "reward": rewards.mean().item(),
        "accuracy": (rewards >= 1.0).float().mean().item(),
        "tool_call_rate": sum(t.n_tool_calls > 0 for g in groups for t in g) / (n_tasks * cfg.group_size),
        "turns": sum(sum(1 for tr in t.turns if tr.role == "assistant") for g in groups for t in g)
                 / (n_tasks * cfg.group_size),
        "skipped_frac": 1.0 - len(kept_ids) / (n_tasks * cfg.group_size),
        "t_rollout": time.perf_counter() - t0,
    }
    if not kept_ids:
        stats.update(loss=0.0, clip_frac=0.0, approx_kl=0.0, grad_norm=0.0)
        return stats
    T = max(len(x) for x in kept_ids)
    ids = torch.stack([F.pad(x, (0, T - len(x)), value=pad_id) for x in kept_ids])       # (N, T)
    mask = torch.stack([F.pad(m, (0, T - 1 - len(m)), value=0.0) for m in kept_masks])  # (N, T-1)
    stats.update(policy_update(model, ref, ids, mask, torch.stack(kept_adv), cfg, optimizer))
    stats["t_step"] = time.perf_counter() - t0
    stats["groups"] = groups
    return stats


# --------------------------------------------------------------- toy env
SAFE_EXPR = set("0123456789+-*/() ")


class CalculatorEnv:
    """A one-question environment with a ``calc`` tool (Chapter 21's toy agent task).

    Task: "What is a + b?". Right after the ``<|assistant|>`` tag the model may emit
        <|tool_call|>{"name": "calc", "arguments": {"expression": "a + b"}}<|end|>
    (the same layout ``chat.render`` uses for a ``tool_call`` message) and gets the evaluated
    result back as a ``<|tool_result|>`` turn, followed by a fresh ``<|assistant|>`` tag. Reward: 1.0 for the correct
    final answer, +0.2 if the tool was called with an expression that evaluates to it.
    (A "tool-use bonus" is a mild shaping reward; Chapter 21 shows how such bonuses can be
    hacked, e.g. by calling the tool and then ignoring it.)
    """
    system_prompt = ("You are TinyLM. You can use the calc tool by writing a tool call with an "
                     "expression. Answer briefly.")

    def __init__(self, a: int, b: int) -> None:
        self.a, self.b = a, b
        self.answer = a + b
        self.tool_ok = False
        self.example = TaskExample("add", f"What is {a} + {b}?", f"{a} + {b} = {a + b}", {"answer": a + b})

    @classmethod
    def from_seed(cls, seed: int, max_value: int = 20) -> "CalculatorEnv":
        import random
        rng = random.Random(seed)
        return cls(rng.randint(0, max_value), rng.randint(0, max_value))

    def reset(self) -> str:
        self.tool_ok = False
        return self.example.prompt

    def calc(self, expression: str) -> str:
        if not expression or set(expression) - SAFE_EXPR:
            return "error: only digits and + - * / ( ) are allowed"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))    # safe: charset restricted
        except Exception as e:                                         # zero division, syntax
            return f"error: {type(e).__name__}"

    def step(self, assistant_text: str) -> tuple[Optional[str], float, bool]:
        call = chat.parse_tool_call(assistant_text)
        if call is not None:                                       # a tool turn: not done yet
            if call.get("name") == "calc":
                result = self.calc(str(call.get("arguments", {}).get("expression", "")))
                self.tool_ok = result == str(self.answer)
                return result, 0.0, False
            return f"error: unknown tool {call.get('name')!r}", 0.0, False
        reward = tasks.verify(self.example, assistant_text.replace(chat.END, ""))
        if reward >= 1.0 and self.tool_ok:
            reward += 0.2
        return None, reward, True


# Names used in the course outline.
reinforce = reinforce_loss
ppo_loss = ppo_clip_loss
multi_turn_grpo = multi_turn_grpo_step
