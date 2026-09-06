"""Reward models and reward functions (Chapter 17).

A **reward** is a single number that says how good a model's answer is. This file
shows the three places that number can come from:

1. A **reward model** (``RewardModel``): a copy of TinyLM with its language head
   replaced by a tiny "score" head. It is trained on **preference pairs** —
   (prompt, chosen answer, rejected answer) — with the Bradley–Terry loss, so that
   the chosen answer gets the higher score. This is the classic RLHF recipe
   (InstructGPT, 2022).
2. A **verifiable reward** (``verifiable_reward`` / ``combined_reward``): a program
   checks the answer. No training needed; this is what RLVR (DeepSeek-R1, 2025) uses.
3. A **rubric reward** (``rubric_reward``): a checklist of yes/no criteria, averaged.
   "Rubrics as rewards" (2025–2026) is the middle ground for tasks with no single
   correct answer but with checkable properties.

Preference pairs can be written by hand, generated synthetically (``make_preference_pairs``)
or, most usefully, sampled from the model itself and graded by a verifier
(``make_preference_pairs_from_model``). Both kinds feed Chapter 17's DPO (``llm/dpo.py``).

Shapes: B batch, T sequence length, d = d_model.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import chat
from . import tasks
from .generate import sample_group
from .model import TinyLM
from .optim import lr_at
from .tokenizer import BPETokenizer
from .train import History


# =============================================================== the reward model
class RewardModel(nn.Module):
    """TinyLM backbone + a scalar head, read at the last real token of each sequence.

    Why the *last* token? In a causal Transformer the last position is the only one
    that has attended to the whole conversation, so its hidden state is a summary of
    "prompt + answer". The head turns that summary into one number.

    ``forward(ids, lengths)`` -> rewards of shape (B,).
    ``lengths[b]`` is the number of real (non-pad) tokens in row b; the sequences are
    right-padded, so the last real token of row b sits at index ``lengths[b] - 1``.
    """

    def __init__(self, backbone: TinyLM) -> None:
        super().__init__()
        self.backbone = backbone
        d = backbone.cfg.d_model
        self.head = nn.Linear(d, 1)
        # Start the head at zero so every sequence begins with reward 0 (a neutral,
        # well-behaved starting point for the Bradley-Terry loss).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids: Tensor, lengths: Tensor) -> Tensor:
        """ids: (B, T) token ids, right-padded. lengths: (B,) real lengths. Returns (B,)."""
        B, T = ids.shape
        _, _, hidden = self.backbone(ids, return_hidden=True)          # hidden: (B, T, d)
        last_index = (lengths - 1).clamp(min=0, max=T - 1)              # (B,)
        summary = hidden[torch.arange(B, device=ids.device), last_index]  # (B, d)
        return self.head(summary).squeeze(-1)                          # (B,)


def bradley_terry_loss(r_chosen: Tensor, r_rejected: Tensor) -> Tensor:
    """The pairwise preference loss:  -log sigmoid(r_chosen - r_rejected), averaged.

    The Bradley–Terry model says the probability that the chosen answer beats the
    rejected one is

        P(chosen ≻ rejected) = sigmoid(r_chosen - r_rejected).

    Read this as: "if the reward model scores the chosen answer 2 points higher, it
    is claiming an 88 % chance a human would prefer it" (sigmoid(2) ≈ 0.88).
    Minimising -log of that probability pushes r_chosen up and r_rejected down,
    but only *relative* to each other — adding a constant to every reward changes
    nothing, which is why reward-model scores have no absolute meaning.

    r_chosen, r_rejected: (B,) tensors. Returns a scalar.
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


# ============================================================== preference pairs
@dataclass
class PreferencePair:
    """One comparison: for this prompt, ``chosen`` is better than ``rejected``."""
    prompt_messages: list[dict]        # chat messages up to (not including) the answer
    chosen: str                        # the preferred assistant reply
    rejected: str                      # the dispreferred assistant reply
    meta: dict = field(default_factory=dict)


# The kinds of "plausible wrong answers" the synthetic generator knows how to make.
WRONG_STYLES = ["wrong", "verbose", "empty", "junk", "off_by_one"]


def _wrong_answer(ex: tasks.TaskExample, rng: random.Random) -> str:
    """A believable but incorrect answer for one task example (same *shape* as the truth)."""
    if ex.task in ("add", "sub"):
        a, b = tasks.NUM_RE.findall(ex.prompt)[:2]
        delta = rng.choice([-10, -2, -1, 1, 2, 10])
        op = "+" if ex.task == "add" else "-"
        return f"{a} {op} {b} = {int(ex.meta['answer']) + delta}"
    if ex.task == "count":
        return str(max(1, int(ex.meta["answer"]) + rng.choice([-1, 1])))
    if ex.task == "reverse":
        word = ex.prompt.split(":")[-1].strip()
        return rng.choice([word, word[::-1][1:] or word, word[::-1].upper()])
    if ex.task == "upper":
        word = ex.prompt.split(":")[-1].strip()
        return rng.choice([word, word.capitalize(), word[::-1].upper()])
    if ex.task == "first":
        words = ex.prompt.split(":")[-1].split()
        return " ".join(w[-1] for w in words)            # *last* letters instead of first
    # story_qa: a different item from the same category
    from .data import COLORS, ANIMALS, NAMES
    answer = str(ex.meta["answer"])
    pool = COLORS if answer in COLORS else ANIMALS if answer in ANIMALS else NAMES
    return rng.choice([w for w in pool if w != answer])


def _apply_style(style: str, correct: str, wrong: str, rng: random.Random) -> str:
    """Wrap a wrong (or right) answer in one of the failure styles a real model shows."""
    if style == "wrong":
        return wrong
    if style == "off_by_one":            # only differs from "wrong" for number tasks
        return wrong
    if style == "verbose":
        return rng.choice([
            f"Well, let me think about this. I believe the answer is {wrong}. I hope that helps!",
            f"That is a great question. After careful thought, {wrong}. Let me know if you need more.",
        ])
    if style == "empty":
        return rng.choice(["", "I don't know.", "Sorry, I cannot answer that."])
    if style == "junk":
        return correct + rng.choice([" and the moon is blue blue blue", " kite kite kite kite",
                                     "\nAlso: the the the the the"])
    raise ValueError(style)


def make_preference_pairs(examples: Sequence[tasks.TaskExample], n_wrong_styles: int = 1,
                          seed: int = 0, styles: Sequence[str] = WRONG_STYLES) -> list[PreferencePair]:
    """Synthetic preferences: chosen = the reference answer, rejected = a plausible failure.

    For each example we draw ``n_wrong_styles`` distinct failure styles from ``styles``
    (see ``WRONG_STYLES``) and emit one pair per style. Note that the "junk" style
    keeps the correct answer and adds rambling, so a *verifier* would still accept it;
    only a reward model that has learned "short and clean is better" ranks it below.
    """
    rng = random.Random(seed)
    pairs = []
    for ex in examples:
        wrong = _wrong_answer(ex, rng)
        for style in rng.sample(list(styles), min(n_wrong_styles, len(styles))):
            rejected = _apply_style(style, ex.answer, wrong, rng)
            pairs.append(PreferencePair(ex.messages(with_answer=False), ex.answer, rejected,
                                        meta={"task": ex.task, "style": style, "example": ex}))
    return pairs


def _decode_completion(tok: BPETokenizer, row: Sequence[int], prompt_len: int,
                       stop_ids: set[int]) -> tuple[str, list[int]]:
    """Cut a sampled row after the prompt and at the first stop/pad token."""
    ids = []
    for t in list(row)[prompt_len:]:
        if t in stop_ids:
            break
        ids.append(t)
    return tok.decode(ids), ids


@torch.no_grad()
def make_preference_pairs_from_model(model: TinyLM, tok: BPETokenizer, examples: Sequence[tasks.TaskExample],
                                     n_samples: int = 4, seed: int = 0, max_new_tokens: int = 24,
                                     temperature: float = 1.0, verbose: bool = False
                                     ) -> tuple[list[PreferencePair], dict]:
    """On-policy pairs: sample ``n_samples`` answers per prompt, grade them with the verifier,
    and pair one correct answer with one incorrect answer.

    This is rejection sampling turned into preference data: prompts where every sample
    is right (or every sample is wrong) teach nothing about *ranking*, so they yield no
    pair. Returns (pairs, stats) where stats counts what happened.
    """
    rng = random.Random(seed)
    end_id = tok.special_tokens[chat.END]
    pad_id = tok.special_tokens["<|pad|>"]
    stop_ids = {end_id, pad_id, tok.special_tokens["<|eos|>"]}
    pairs, n_all_right, n_all_wrong, n_correct_total = [], 0, 0, 0
    for i, ex in enumerate(examples):
        prompt_ids = tok.encode(chat.render(ex.messages(with_answer=False), add_generation_prompt=True))
        budget = model.cfg.max_seq_len - len(prompt_ids)
        if budget <= 1:
            continue                                                   # prompt too long for the model
        rows = sample_group(model, prompt_ids, n_samples, min(max_new_tokens, budget),
                            temperature=temperature, stop_ids=[end_id], pad_id=pad_id, seed=seed + i)
        completions = [_decode_completion(tok, r.tolist(), len(prompt_ids), stop_ids)[0] for r in rows]
        scores = [tasks.verify(ex, c) for c in completions]
        n_correct_total += int(sum(scores))
        right = [c for c, s in zip(completions, scores) if s > 0]
        wrong = [c for c, s in zip(completions, scores) if s == 0]
        if not wrong:
            n_all_right += 1
        elif not right:
            n_all_wrong += 1
        else:
            pairs.append(PreferencePair(ex.messages(with_answer=False), rng.choice(right), rng.choice(wrong),
                                        meta={"task": ex.task, "style": "on_policy", "example": ex}))
    stats = {"n_prompts": len(examples), "n_pairs": len(pairs), "n_all_correct": n_all_right,
             "n_all_wrong": n_all_wrong,
             "sample_accuracy": n_correct_total / max(1, len(examples) * n_samples)}
    if verbose:
        print(f"[pairs] {stats}")
    return pairs, stats


# ------------------------------------------------------------------- encoding
def encode_response(tok: BPETokenizer, prompt_messages: Sequence[dict], response: str,
                    max_len: Optional[int] = None) -> tuple[list[int], list[int]]:
    """Tokenize prompt + response the way the chat template does.

    Returns (ids, response_mask): ``response_mask[i] = 1`` iff ``ids[i]`` belongs to
    the response (its text plus the closing ``<|end|>``); the rendered prompt, including
    the ``<|assistant|>`` tag, is 0. DPO and the reward model both need this split.
    """
    prompt_ids = tok.encode(chat.render(prompt_messages, add_generation_prompt=True))
    response_ids = tok.encode(response, allowed_special=False) + [tok.special_tokens[chat.END]]
    ids = prompt_ids + response_ids
    mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    if max_len is not None:
        ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


def encode_pair(tok: BPETokenizer, pair: PreferencePair, max_len: Optional[int] = None
                ) -> tuple[list[int], list[int]]:
    """(chosen_ids, rejected_ids): the same prompt followed by each response + <|end|>."""
    chosen_ids, _ = encode_response(tok, pair.prompt_messages, pair.chosen, max_len)
    rejected_ids, _ = encode_response(tok, pair.prompt_messages, pair.rejected, max_len)
    return chosen_ids, rejected_ids


def pad_rows(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """Right-pad a list of id lists to a (B, T) tensor; also return the lengths (B,)."""
    T = max(len(r) for r in rows)
    x = torch.full((len(rows), T), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    return x, lengths


def _rm_batch(tok: BPETokenizer, pairs: Sequence[PreferencePair], max_len: Optional[int]
              ) -> tuple[Tensor, Tensor]:
    """Chosen rows first, then rejected rows: ids (2B, T), lengths (2B,)."""
    pad_id = tok.special_tokens["<|pad|>"]
    encoded = [encode_pair(tok, p, max_len) for p in pairs]
    rows = [c for c, _ in encoded] + [r for _, r in encoded]
    return pad_rows(rows, pad_id)


# ------------------------------------------------------------------- training
@dataclass
class PrefHistory(History):
    """``train.History`` plus the two numbers preference training cares about."""
    accuracy: list[float] = field(default_factory=list)   # fraction of pairs ranked correctly
    margin: list[float] = field(default_factory=list)     # mean (chosen - rejected) score


@dataclass
class RMConfig:
    steps: int = 200
    batch_size: int = 8          # pairs per step (so 2 * batch_size sequences)
    lr: float = 1e-4
    warmup: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 20
    seed: int = 0
    max_len: Optional[int] = None   # truncate sequences (default: the backbone's max_seq_len)


def _pair_batches(pairs: Sequence[PreferencePair], batch_size: int, rng: random.Random):
    """Endless generator of shuffled mini-batches of pairs (reshuffles every epoch)."""
    order = list(range(len(pairs)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            yield [pairs[j] for j in order[i:i + batch_size]]


def train_reward_model(rm: RewardModel, tok: BPETokenizer, pairs: Sequence[PreferencePair],
                       cfg: RMConfig = RMConfig(), verbose: bool = True) -> PrefHistory:
    """Fit the reward model on preference pairs with the Bradley–Terry loss."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    max_len = cfg.max_len or rm.backbone.cfg.max_seq_len
    device = next(rm.parameters()).device
    opt = torch.optim.AdamW(rm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    batches = _pair_batches(pairs, cfg.batch_size, rng)
    history = PrefHistory()
    rm.train()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at(step, cfg.steps, cfg.lr, cfg.warmup, "cosine")
        for g in opt.param_groups:
            g["lr"] = lr
        batch = next(batches)
        ids, lengths = _rm_batch(tok, batch, max_len)
        rewards = rm(ids.to(device), lengths.to(device))               # (2B,)
        r_chosen, r_rejected = rewards[:len(batch)], rewards[len(batch):]
        loss = bradley_terry_loss(r_chosen, r_rejected)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            margin = (r_chosen - r_rejected).detach()
            acc = (margin > 0).float().mean().item()
            history.step.append(step)
            history.train_loss.append(loss.item())
            history.lr.append(lr)
            history.accuracy.append(acc)
            history.margin.append(margin.mean().item())
            history.tokens_per_sec.append(0.0)
            if verbose:
                print(f"[rm] step {step:4d} | loss {loss.item():.4f} | pair acc {acc:.2f} | "
                      f"margin {margin.mean().item():+.3f} | {time.perf_counter() - t0:.1f}s")
    rm.eval()
    return history


@torch.no_grad()
def reward_accuracy(rm: RewardModel, tok: BPETokenizer, pairs: Sequence[PreferencePair],
                    batch_size: int = 16, max_len: Optional[int] = None) -> float:
    """Fraction of pairs where the reward model scores chosen above rejected."""
    rm.eval()
    max_len = max_len or rm.backbone.cfg.max_seq_len
    device = next(rm.parameters()).device
    correct = 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        ids, lengths = _rm_batch(tok, batch, max_len)
        r = rm(ids.to(device), lengths.to(device))
        correct += int((r[:len(batch)] > r[len(batch):]).sum())
    return correct / max(1, len(pairs))


@torch.no_grad()
def score_completions(rm: RewardModel, tok: BPETokenizer, prompt_messages: Sequence[dict],
                      completions: Sequence[str], max_len: Optional[int] = None) -> Tensor:
    """Reward-model scores (n,) for several candidate answers to one prompt (best-of-n)."""
    rm.eval()
    max_len = max_len or rm.backbone.cfg.max_seq_len
    rows = [encode_response(tok, prompt_messages, c, max_len)[0] for c in completions]
    ids, lengths = pad_rows(rows, tok.special_tokens["<|pad|>"])
    device = next(rm.parameters()).device
    return rm(ids.to(device), lengths.to(device)).cpu()


# ============================================================ verifiable rewards
def verifiable_reward(example: tasks.TaskExample, completion: str) -> float:
    """1.0 if the completion answers the task correctly, else 0.0 (``tasks.verify``)."""
    return tasks.verify(example, completion)


def combined_reward(example: tasks.TaskExample, completion: str, completion_ids: Sequence[int],
                    max_len: int, format_weight: float = 0.1) -> float:
    """The shaped reward used by Chapter 19's GRPO:

        reward = verify(answer)             1.0 or 0.0    — the thing we actually want
               + 0.1 * format_reward        0.1 or 0.0    — a nudge toward the expected layout
               + length_penalty             -0.5 or 0.0   — punish running into the token limit

    The small format term breaks ties between wrong answers (a well-formed wrong answer
    beats gibberish), and the length penalty stops the model from rambling forever.
    Keep shaping terms small: if they outweigh correctness, the model optimises them instead.
    """
    return (tasks.verify(example, completion)
            + format_weight * tasks.format_reward(example, completion)
            + tasks.length_penalty(completion_ids, max_len))


# ================================================================ rubric rewards
Criterion = Callable[[str], bool | float]


def rubric_reward(completion: str, rubric: Sequence[tuple[str, Criterion]]) -> tuple[float, dict[str, float]]:
    """Score a completion against a checklist. Returns (mean score, {criterion: score}).

    Each rubric entry is ``(name, check)`` where ``check(completion)`` returns True/False
    (or a number in [0, 1]). The reward is the average, so every criterion weighs the
    same; scale a check's return value if you want it to count more. Rubrics are the
    2025–2026 answer to "how do we reward answers with no single right string?": an
    LLM or a human writes the checklist once, then any grader (even a regex) applies it.
    """
    per_criterion = {name: float(check(completion)) for name, check in rubric}
    score = sum(per_criterion.values()) / max(1, len(per_criterion))
    return score, per_criterion


# A demo rubric for arithmetic answers such as "23 + 45 = 68".
# WARNING (deliberate): no criterion checks *correctness*, so ``rubric_reward("0 + 0 = 0")``
# scores 1.0 for any question. Lab 17 uses this to show a policy exploiting a rubric; a real
# rubric pairs style criteria with a verifiable-correctness criterion (``tasks.strict_verify``).
ARITHMETIC_RUBRIC: list[tuple[str, Criterion]] = [
    ("has_equation", lambda c: re.search(r"\d+ [+-] \d+ = -?\d+", c) is not None),
    ("single_line", lambda c: len(c.strip().split("\n")) == 1),
    ("is_short", lambda c: 0 < len(c.strip()) <= 40),
    ("no_hedging", lambda c: not any(w in c.lower() for w in ("i think", "maybe", "not sure", "i believe"))),
    ("ends_cleanly", lambda c: c.strip() != "" and (c.strip()[-1].isdigit() or c.strip().endswith("."))),
]
