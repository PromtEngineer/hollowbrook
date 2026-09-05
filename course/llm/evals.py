"""Evaluation (Chapter 23): is the model actually better, and can we trust the number?

Four kinds of measurement, from cheapest to most involved:

* ``perplexity``      — how surprised the model is by held-out text (a training signal,
                        not a capability measure);
* ``eval_tasks``      — exact-match accuracy on tasks with a verifiable answer;
* ``bootstrap_ci``    — a confidence interval, so "0.62 vs 0.58" is not over-read;
* ``judge_pairwise``  — LLM-as-judge scaffolding, with a position-bias check.

Plus two hygiene tools: ``contamination_check`` (did the eval leak into training?) and
``compare_checkpoints`` (one markdown table for several saved models).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .chat import render
from .data import ngrams
from .model import TinyLM
from .sft import respond
from .tasks import SYSTEM_PROMPT, TaskExample, verify
from .tokenizer import BPETokenizer
from .train import estimate_loss


# --------------------------------------------------------------- perplexity
def perplexity(model: TinyLM, tokens: Tensor, batch_size: int = 16, seq_len: int = 128,
               n_batches: int = 20) -> float:
    """exp(mean next-token loss) over random windows of ``tokens`` (a 1-D id stream).

    Read it as "the model is as unsure as if choosing uniformly among this many tokens".
    Lower is better; 1.0 would be a perfect predictor.
    """
    seq_len = min(seq_len, model.cfg.max_seq_len)
    device = next(model.parameters()).device.type
    return math.exp(estimate_loss(model, tokens, batch_size, seq_len, n_batches, device=device))


# ----------------------------------------------------------- task accuracy
@dataclass
class EvalResult:
    accuracy: float                                  # overall exact-match accuracy
    per_task: dict[str, float]                       # task name -> accuracy
    n_per_task: dict[str, int]                       # task name -> number of examples
    correct: list[float]                             # 1.0 / 0.0 per example (for bootstrap_ci)
    samples: list[tuple[str, str, bool]] = field(default_factory=list)   # (prompt, completion, correct)
    tasks: list[str] = field(default_factory=list)   # task name per example, aligned with ``correct``

    def table(self, ci: bool = True) -> str:
        """A markdown table: one row per task, then the overall row (with a 95% CI)."""
        lines = ["| task | n | accuracy | 95% CI |", "|---|---:|---:|---|"]
        by_task: dict[str, list[float]] = {}
        for task, score in zip(self.tasks, self.correct):
            by_task.setdefault(task, []).append(score)
        rows = [(t, by_task[t]) for t in sorted(self.per_task)] + [("**all**", self.correct)]
        for name, scores in rows:
            acc = sum(scores) / max(1, len(scores))
            lo, hi = bootstrap_ci(scores) if ci else (0.0, 0.0)
            lines.append(f"| {name} | {len(scores)} | {acc:.2f} | {f'[{lo:.2f}, {hi:.2f}]' if ci else '-'} |")
        return "\n".join(lines)


def eval_tasks(model: TinyLM, tok: BPETokenizer, examples: Sequence[TaskExample], max_new_tokens: int = 32,
               temperature: float = 0.0, system: Optional[str] = SYSTEM_PROMPT) -> EvalResult:
    """Ask the model every example's question and grade the answer with ``tasks.verify``.

    Greedy decoding (temperature 0) makes the score reproducible. Each example is
    generated on its own, so the cost is ``len(examples) * max_new_tokens`` forward steps.
    """
    was_training = model.training
    correct: list[float] = []
    samples: list[tuple[str, str, bool]] = []
    tasks: list[str] = []
    for ex in examples:
        completion = respond(model, tok, ex.prompt, system=system,
                             max_new_tokens=max_new_tokens, temperature=temperature)
        score = verify(ex, completion)                     # 1.0 or 0.0 — the verifiable reward
        correct.append(score)
        samples.append((ex.prompt, completion, bool(score)))
        tasks.append(ex.task)
    if was_training:
        model.train()

    n_per_task: dict[str, int] = {}
    hits: dict[str, float] = {}
    for task, score in zip(tasks, correct):
        n_per_task[task] = n_per_task.get(task, 0) + 1
        hits[task] = hits.get(task, 0.0) + score
    per_task = {t: hits[t] / n_per_task[t] for t in n_per_task}
    accuracy = sum(correct) / len(correct) if correct else 0.0
    return EvalResult(accuracy, per_task, n_per_task, correct, samples, tasks)


# ---------------------------------------------------------- significance
def bootstrap_ci(correct: Sequence[float], n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """A bootstrap confidence interval for the mean of ``correct`` (0/1 scores).

    Resample the scores with replacement ``n_boot`` times, take the mean each time,
    and report the central ``level`` of those means. With 100 examples the 95% CI is
    roughly ±0.10 — so two models scoring 0.62 and 0.58 are *not* distinguishable.
    """
    arr = np.asarray(list(correct), dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))            # (n_boot, n) resampled indices
    means = arr[idx].mean(axis=1)                          # (n_boot,)
    tail = (1.0 - level) / 2
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


# ---------------------------------------------------------- LLM-as-judge
Prompt = Union[str, TaskExample]
JudgeFn = Callable[[Prompt, str, str], str]      # (prompt, answer_a, answer_b) -> "A" | "B" | "tie"


def rule_based_judge(prompt: Prompt, a: str, b: str) -> str:
    """A stand-in for an LLM judge, so the scaffolding can be exercised without one.

    Rules, in order:
    1. If ``prompt`` is a ``TaskExample``, the answer that ``tasks.verify`` marks correct wins.
    2. If exactly one answer is empty, the other wins.
    3. If both contain a digit, the shorter wins (concise arithmetic beats rambling).
    4. Otherwise it is a tie.
    A real judge would be ``lambda p, a, b: ask_llm(JUDGE_TEMPLATE.format(...))``.
    """
    if isinstance(prompt, TaskExample):
        sa, sb = verify(prompt, a), verify(prompt, b)
        if sa != sb:
            return "A" if sa > sb else "B"
    if bool(a.strip()) != bool(b.strip()):
        return "A" if a.strip() else "B"
    if any(c.isdigit() for c in a) and any(c.isdigit() for c in b) and len(a) != len(b):
        return "A" if len(a) < len(b) else "B"
    return "tie"


def judge_pairwise(prompt: Prompt, a: str, b: str, judge_fn: JudgeFn = rule_based_judge) -> str:
    """Ask ``judge_fn`` which of two answers to ``prompt`` is better: "A", "B" or "tie"."""
    verdict = judge_fn(prompt, a, b)
    assert verdict in ("A", "B", "tie"), f"judge returned {verdict!r}"
    return verdict


def position_bias_check(judge_fn: JudgeFn, prompt: Prompt, a: str, b: str) -> dict:
    """Judge (a, b), then judge (b, a) and translate back. A consistent judge gives
    the same winner both times; LLM judges often prefer whichever answer is shown
    first, and this check exposes that."""
    forward = judge_pairwise(prompt, a, b, judge_fn)
    swapped_raw = judge_pairwise(prompt, b, a, judge_fn)
    swapped = {"A": "B", "B": "A", "tie": "tie"}[swapped_raw]     # back in terms of a/b
    return {"forward": forward, "swapped": swapped, "consistent": forward == swapped,
            "position_bias": forward != swapped}


# ---------------------------------------------------------- contamination
def contamination_check(train_docs: Sequence[dict], eval_examples: Sequence[Prompt], n: int = 8) -> float:
    """Fraction of eval prompts sharing at least one ``n``-word n-gram with the training data.

    This is the GPT-3 / Llama style overlap test (``data.decontaminate`` is the
    training-side version). Prompts shorter than ``n`` words cannot match — pick
    ``n`` with the prompt length in mind (8 here; 13 in production reports).
    """
    if not eval_examples:
        return 0.0
    train_ngrams: set[tuple[str, ...]] = set()
    for d in train_docs:
        train_ngrams |= ngrams(d["text"], n)
    hits = 0
    for ex in eval_examples:
        text = ex.prompt if isinstance(ex, TaskExample) else ex
        if ngrams(text, n) & train_ngrams:
            hits += 1
    return hits / len(eval_examples)


# ------------------------------------------------------- checkpoint table
def compare_checkpoints(paths: Sequence[str], tok: BPETokenizer, examples: Sequence[TaskExample],
                        tokens: Optional[Tensor] = None, max_new_tokens: int = 32, n_batches: int = 10) -> str:
    """Markdown table of task accuracy and perplexity for several saved models.

    Perplexity is measured on ``tokens`` if given, otherwise on the chat-formatted
    ``examples`` themselves (so it tracks how well each checkpoint models the
    conversation format, not just Storyland prose).
    """
    if tokens is None:
        ids: list[int] = []
        for ex in examples:
            ids.extend(tok.encode(render(ex.messages(), add_generation_prompt=False)))
        tokens = torch.tensor(ids, dtype=torch.long)
    lines = ["| checkpoint | accuracy | perplexity |", "|---|---:|---:|"]
    for path in paths:
        model = TinyLM.load(path)
        res = eval_tasks(model, tok, examples, max_new_tokens=max_new_tokens)
        seq_len = min(128, model.cfg.max_seq_len, len(tokens) - 2)
        ppl = perplexity(model, tokens, batch_size=8, seq_len=seq_len, n_batches=n_batches)
        lines.append(f"| {os.path.basename(path)} | {res.accuracy:.2f} | {ppl:.1f} |")
    return "\n".join(lines)
