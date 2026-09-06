"""Post-training tasks for TinyLM: instruction examples with *verifiable* answers.

Used by SFT (Chapter 15), reward models / DPO (17), GRPO (19), distillation (20),
agentic RL (21) and evals (23). Every task has a deterministic checker, so we can
compute a reward without a human — the "verifiable rewards" idea behind RLVR.

Task types
----------
add        "What is 23 + 45?"                      -> "23 + 45 = 68"
sub        "What is 45 - 23?"                      -> "45 - 23 = 22"
reverse    "Reverse the word: kite"                -> "etik"
upper      "Write in capitals: kite"               -> "KITE"
count      "How many words: the red kite flew"     -> "4"
first      "First letter of each word: red kite"   -> "r k"
story_qa   "<story>\nWhat color is Mia's kite?"    -> "red"
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .data import make_story, OBJECTS, COLORS, NAMES, PLACES, ANIMALS

TASK_TYPES = ["add", "sub", "reverse", "upper", "count", "first", "story_qa"]
SYSTEM_PROMPT = "You are TinyLM, a helpful assistant. Answer briefly."


@dataclass
class TaskExample:
    task: str
    prompt: str            # the user message
    answer: str            # the reference answer (what a perfect assistant says)
    meta: dict = field(default_factory=dict)

    def messages(self, with_answer: bool = True, system: Optional[str] = SYSTEM_PROMPT) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": self.prompt})
        if with_answer:
            msgs.append({"role": "assistant", "content": self.answer})
        return msgs


def make_example(task: str, rng: random.Random, max_value: int = 99) -> TaskExample:
    if task == "add":
        a, b = rng.randint(0, max_value), rng.randint(0, max_value)
        return TaskExample(task, f"What is {a} + {b}?", f"{a} + {b} = {a + b}", {"answer": a + b})
    if task == "sub":
        a, b = rng.randint(0, max_value), rng.randint(0, max_value)
        a, b = max(a, b), min(a, b)
        return TaskExample(task, f"What is {a} - {b}?", f"{a} - {b} = {a - b}", {"answer": a - b})
    word = rng.choice(OBJECTS + COLORS + PLACES + ANIMALS)
    if task == "reverse":
        return TaskExample(task, f"Reverse the word: {word}", word[::-1], {"answer": word[::-1]})
    if task == "upper":
        return TaskExample(task, f"Write in capitals: {word}", word.upper(), {"answer": word.upper()})
    words = rng.sample(COLORS + OBJECTS + ANIMALS + ["the", "a", "big", "small"], rng.randint(2, 6))
    if task == "count":
        return TaskExample(task, "How many words: " + " ".join(words), str(len(words)), {"answer": str(len(words))})
    if task == "first":
        ans = " ".join(w[0] for w in words)
        return TaskExample(task, "First letter of each word: " + " ".join(words), ans, {"answer": ans})
    if task == "story_qa":
        story = make_story(rng, 3)
        s = story["slots"]
        text = story["text"].split("\nQuestion:")[0]
        q, a = rng.choice([
            (f"What color is {s['A']}'s {s['o']}?", s["c"]),
            (f"Which animal was at the {s['p']}?", s["an"]),
            (f"Who did {s['A']} meet?", s["B"]),
        ])
        return TaskExample(task, text + "\n" + q, a, {"answer": a})
    raise ValueError(task)


def make_examples(n: int, seed: int = 0, tasks: Sequence[str] = TASK_TYPES,
                  weights: Optional[Sequence[float]] = None, max_value: int = 99) -> list[TaskExample]:
    rng = random.Random(seed)
    return [make_example(rng.choices(list(tasks), weights)[0], rng, max_value) for _ in range(n)]


# ------------------------------------------------------------- verification
NUM_RE = re.compile(r"-?\d+")


def extract_answer(task: str, completion: str) -> Optional[str]:
    """Pull the model's final answer out of free text (lenient, like a real grader)."""
    text = completion.strip()
    if task in ("add", "sub", "count"):
        nums = NUM_RE.findall(text.split("=")[-1] if "=" in text else text)
        return nums[-1] if nums else None
    first_line = text.split("\n")[0].strip().rstrip(".")
    return first_line


def verify(example: TaskExample, completion: str) -> float:
    """1.0 if the completion answers correctly, else 0.0. This *is* the verifiable reward."""
    got = extract_answer(example.task, completion)
    if got is None:
        return 0.0
    want = str(example.meta["answer"])
    if example.task in ("add", "sub", "count", "upper"):   # "upper" must be case-sensitive!
        return float(got.strip() == want)
    return float(got.strip().lower() == want.lower())


EQUATION_RE = re.compile(r"(-?\d+)\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)")


def strict_verify(example: TaskExample, completion: str) -> float:
    """Like ``verify`` but, for arithmetic, also requires the restated operands to match.

    ``verify`` only grades the number after the last "=", so a policy under RL can learn
    to write ``8 + 8 = 10`` for "What is 2 + 8?" and still be rewarded (Labs 18–19 show
    this happening). A stricter grader closes that hole — the general lesson is that
    every reward has an exploit until you look for it (Chapter 17).
    """
    if example.task not in ("add", "sub"):
        return verify(example, completion)
    m = EQUATION_RE.search(completion)
    if m is None:
        return 0.0
    a, op, b, ans = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    ea, eb = example.meta.get("a"), example.meta.get("b")
    if ea is None:                                   # recover operands from the prompt text
        nums = NUM_RE.findall(example.prompt)
        ea, eb = int(nums[0]), int(nums[1])
    want_op = "+" if example.task == "add" else "-"
    return float(a == ea and b == eb and op == want_op and ans == example.meta["answer"])


def format_reward(example: TaskExample, completion: str) -> float:
    """A small shaping reward for answering in the expected format (Chapter 19)."""
    if example.task in ("add", "sub"):
        return 1.0 if re.search(r"\d+ [+-] \d+ = -?\d+", completion) else 0.0
    return 1.0 if len(completion.strip().split("\n")) == 1 else 0.0


def length_penalty(completion_ids: Sequence[int], max_len: int) -> float:
    """Penalise hitting the length limit without finishing (no <|end|>)."""
    return -0.5 if len(completion_ids) >= max_len else 0.0
