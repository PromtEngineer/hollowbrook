"""Data: a synthetic corpus generator and a miniature pretraining-data pipeline.

Chapter 8. Real pipelines (FineWeb, DCLM, Nemotron-CC, 2024–2026) run these exact
stages over petabytes of Common Crawl; here they run over a few thousand documents so
you can watch each stage work.

    raw docs -> normalise -> language filter -> heuristic quality filter -> PII scrub
             -> exact dedup -> fuzzy (MinHash) dedup -> model-based quality classifier
             -> decontaminate against evals -> mix sources -> tokenize & pack

The corpus is "Storyland": simple English stories about a fixed cast of characters,
plus arithmetic and question/answer lines. A tiny model can learn it on a CPU in
minutes, and the arithmetic gives later chapters *verifiable* tasks for RL.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch

# ============================================================== Storyland corpus
NAMES = ["Mia", "Leo", "Ava", "Sam", "Zoe", "Max", "Lily", "Ben", "Ivy", "Tom", "Ruby", "Jack",
         "Nora", "Finn", "Ella", "Owen"]
OBJECTS = ["kite", "ball", "book", "hat", "cake", "boat", "drum", "lamp", "coin", "cup", "bell",
           "map", "rope", "shell", "flag", "pear", "apple", "box", "key", "ring"]
COLORS = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "brown", "white", "black"]
PLACES = ["park", "beach", "garden", "forest", "river", "hill", "farm", "market", "school", "lake",
          "cave", "bridge", "castle", "village", "meadow"]
ANIMALS = ["dog", "cat", "bird", "frog", "duck", "fox", "owl", "rabbit", "horse", "goat"]
FEELINGS = ["happy", "sad", "proud", "sleepy", "brave", "calm", "excited", "shy"]
WEATHER = ["sunny", "rainy", "windy", "cold", "warm", "foggy", "snowy"]

STORY_TEMPLATES = [
    "{A} had a {c} {o}. One {w} day {A} took the {o} to the {p}.",
    "At the {p}, {A} met {B}. {B} was {f} because {B} lost a {o2}.",
    "{A} and {B} looked for the {o2} near the {p}. A {an} was sitting on it!",
    "The {an} gave the {o2} back. {B} was very {f2} and said thank you.",
    "Then {A} and {B} played with the {c} {o} until the sun went down.",
    "{A} went home and put the {o} in a {c2} box. The end.",
    "{B} liked the {c} {o} more than the {c2} {o2}.",
    "It was {w} at the {p}. {A} wore a {c2} hat and felt {f}.",
    "A {an} ran across the {p}. {A} laughed and {B} clapped.",
    "{A} counted the {o}s: one, two, three. There were three {o}s.",
]

QA_TEMPLATES = [
    ("What color is {A}'s {o}?", "{A}'s {o} is {c}."),
    ("Where did {A} take the {o}?", "{A} took the {o} to the {p}."),
    ("Who did {A} meet at the {p}?", "{A} met {B} at the {p}."),
    ("How did {B} feel?", "{B} felt {f}."),
    ("What animal was at the {p}?", "A {an} was at the {p}."),
]


def make_story(rng: random.Random, n_sentences: int = 5) -> dict:
    """One Storyland document: a short story, sometimes followed by Q&A about it."""
    A, B = rng.sample(NAMES, 2)
    o, o2 = rng.sample(OBJECTS, 2)
    c, c2 = rng.sample(COLORS, 2)
    p = rng.choice(PLACES)
    an = rng.choice(ANIMALS)
    f, f2 = rng.sample(FEELINGS, 2)
    w = rng.choice(WEATHER)
    slots = dict(A=A, B=B, o=o, o2=o2, c=c, c2=c2, p=p, an=an, f=f, f2=f2, w=w)
    sents = [t.format(**slots) for t in rng.sample(STORY_TEMPLATES, n_sentences)]
    text = " ".join(sents)
    if rng.random() < 0.5:
        q, a = rng.choice(QA_TEMPLATES)
        text += "\nQuestion: " + q.format(**slots) + "\nAnswer: " + a.format(**slots)
    return {"text": text, "source": "stories", "slots": slots}


def make_arithmetic(rng: random.Random, max_value: int = 99) -> dict:
    a, b = rng.randint(0, max_value), rng.randint(0, max_value)
    op = rng.choice(["+", "+", "+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    style = rng.random()
    if style < 0.5:
        text = f"What is {a} {op} {b}?\nAnswer: {a} {op} {b} = {ans}."
    else:
        text = f"{a} {op} {b} = {ans}"
    return {"text": text, "source": "math", "a": a, "b": b, "op": op, "answer": ans}


def make_corpus(n_docs: int = 4000, seed: int = 0, math_frac: float = 0.25) -> list[dict]:
    """A clean Storyland corpus of ``n_docs`` documents."""
    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        d = make_arithmetic(rng) if rng.random() < math_frac else make_story(rng, rng.randint(3, 6))
        d["id"] = f"doc{i:06d}"
        docs.append(d)
    return docs


# ------------------------------------------------------------- planted noise
SPAM_LINES = ["Click here to subscribe!!! BUY NOW >>> free free free", "LOGIN | REGISTER | CART (0) | HOME | HOME | HOME",
              "Cookie policy: we use cookies. Accept. Accept. Accept.", "$$$ WIN BIG $$$ !!!! ##### @@@@@ %%%%"]
NON_ENGLISH = ["Der Hund läuft schnell durch den Park und bellt laut.", "El gato duerme en la cama toda la tarde.",
               "Le chat mange du poisson près de la rivière.", "Il cane corre veloce nel parco verde."]


def add_noise(docs: list[dict], seed: int = 0, dup_frac: float = 0.10, near_dup_frac: float = 0.05,
              spam_frac: float = 0.05, foreign_frac: float = 0.03, pii_frac: float = 0.03,
              short_frac: float = 0.03, eval_questions: Sequence[str] = ()) -> list[dict]:
    """Return a *dirty* copy of the corpus with realistic problems planted in it.

    Every planted problem is tagged in ``doc["planted"]`` so the lab can check
    which stage caught it.
    """
    rng = random.Random(seed + 1)
    out = [dict(d, planted=None, orig_id=d["id"]) for d in docs]   # orig_id: which clean doc this came from
    n = len(docs)
    for _ in range(int(n * dup_frac)):                         # exact duplicates
        src = rng.choice(docs)
        out.append(dict(src, planted="exact_dup", orig_id=src["id"]))
    for _ in range(int(n * near_dup_frac)):                    # near duplicates: swap one word
        src = rng.choice(docs)
        d = dict(src, orig_id=src["id"])
        words = d["text"].split()
        i = rng.randrange(len(words))
        words[i] = rng.choice(COLORS + OBJECTS)
        d["text"] = " ".join(words)
        d["planted"] = "near_dup"
        out.append(d)
    for _ in range(int(n * spam_frac)):
        out.append({"text": " ".join(rng.choices(SPAM_LINES, k=3)), "source": "web", "planted": "spam"})
    for _ in range(int(n * foreign_frac)):
        out.append({"text": " ".join(rng.choices(NON_ENGLISH, k=3)), "source": "web", "planted": "non_english"})
    for _ in range(int(n * pii_frac)):
        d = dict(rng.choice(docs))
        d["text"] += f" Contact {d.get('slots', {}).get('A', 'Mia').lower()}@example.com or call 555-0{rng.randint(100,199)}-{rng.randint(1000,9999)}."
        d["planted"] = "pii"
        out.append(d)
    for _ in range(int(n * short_frac)):
        out.append({"text": rng.choice(["The end.", "ok", "Mia.", "hello hello"]), "source": "web", "planted": "too_short"})
    for q in eval_questions:                                   # contamination: eval items leak in
        out.append({"text": f"Here is a popular quiz. {q} Everyone knows this one.", "source": "web",
                    "planted": "contaminated"})
    rng.shuffle(out)
    for i, d in enumerate(out):
        d["id"] = f"raw{i:06d}"
    return out


# ============================================================ curation stages
@dataclass
class StageReport:
    name: str
    kept: int
    dropped: int
    dropped_by_reason: dict = field(default_factory=dict)
    caught_planted: dict = field(default_factory=dict)


@dataclass
class CurationReport:
    stages: list[StageReport] = field(default_factory=list)

    def table(self) -> str:
        lines = [f"{'stage':<28}{'kept':>8}{'dropped':>9}  planted problems caught"]
        for s in self.stages:
            caught = ", ".join(f"{k}:{v}" for k, v in sorted(s.caught_planted.items())) or "-"
            lines.append(f"{s.name:<28}{s.kept:>8}{s.dropped:>9}  {caught}")
        return "\n".join(lines)


def _run_stage(name: str, docs: list[dict], keep_fn: Callable[[dict], Optional[str]],
               report: Optional[CurationReport]) -> list[dict]:
    """Apply ``keep_fn`` (returns None to keep, or a reason string to drop)."""
    kept, reasons, caught = [], Counter(), Counter()
    for d in docs:
        reason = keep_fn(d)
        if reason is None:
            kept.append(d)
        else:
            reasons[reason] += 1
            if d.get("planted"):
                caught[d["planted"]] += 1
    if report is not None:
        report.stages.append(StageReport(name, len(kept), len(docs) - len(kept), dict(reasons), dict(caught)))
    return kept


# 1. normalisation ---------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(docs: list[dict]) -> list[dict]:
    return [dict(d, text=normalize_text(d["text"])) for d in docs]


# 2. language identification (a stopword heuristic; real pipelines use fastText) ----
ENGLISH_STOPWORDS = set("the a an and or but of to in on at for with was were is are it he she they "
                        "his her their this that had has have not very then went one two three what "
                        "where who how did do".split())


def english_score(text: str) -> float:
    """Fraction of alphabetic words that are common English stopwords (0..1)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    return sum(w in ENGLISH_STOPWORDS for w in words) / len(words)


def language_filter(docs: list[dict], min_score: float = 0.08, min_alpha_words: int = 3,
                    report: Optional[CurationReport] = None) -> list[dict]:
    """Keep documents that look English. Documents with fewer than ``min_alpha_words``
    alphabetic words (e.g. a bare equation ``75 + 80 = 155``) carry no language signal
    and are passed through — a real fastText-based filter makes the same exception."""
    def keep(d: dict) -> Optional[str]:
        n_alpha = len(re.findall(r"[a-zA-Z']+", d["text"]))
        if n_alpha < min_alpha_words:
            return None
        return None if english_score(d["text"]) >= min_score else "not_english"
    return _run_stage("language_filter", docs, keep, report)


# 3. heuristic quality rules (Gopher / C4 style) ----------------------------
def gopher_reason(text: str, min_words: int = 5, max_words: int = 2000,
                  max_symbol_ratio: float = 0.1, max_dup_line_frac: float = 0.3) -> Optional[str]:
    words = text.split()
    if len(words) < min_words:
        return "too_short"
    if len(words) > max_words:
        return "too_long"
    mean_len = sum(len(w) for w in words) / len(words)
    if not (2 <= mean_len <= 12):
        return "odd_word_length"
    symbols = sum(text.count(s) for s in "#$%@>|{}[]")
    if symbols / max(1, len(words)) > max_symbol_ratio:
        return "symbol_heavy"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and (len(lines) - len(set(lines))) / len(lines) > max_dup_line_frac:
        return "repeated_lines"
    # repeated word n-grams (spam repeats itself)
    if len(words) >= 8:
        trigrams = list(zip(words, words[1:], words[2:]))
        if (len(trigrams) - len(set(trigrams))) / len(trigrams) > 0.3:
            return "repetitive"
    lowered = text.lower()
    if any(b in lowered for b in ("click here", "subscribe", "cookie policy", "buy now")):
        return "boilerplate"
    return None


def heuristic_filter(docs: list[dict], report: Optional[CurationReport] = None,
                     skip_sources: Sequence[str] = ("math", "code"), **kw) -> list[dict]:
    """Apply the prose heuristics. Documents whose ``source`` is in ``skip_sources`` are
    passed through untouched: rules written for prose (mean word length 2–12, symbol
    ratio) reject bare equations and code, which is exactly why real pipelines curate
    math and code with their own dedicated filters."""
    def keep(d: dict) -> Optional[str]:
        if d.get("source") in skip_sources:
            return None
        return gopher_reason(d["text"], **kw)
    return _run_stage("heuristic_filter", docs, keep, report)


# 4. PII scrubbing ----------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b\d{3}-\d{4}-\d{4}\b|\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\) ?\d{3}-\d{4}")


def scrub_pii(text: str) -> tuple[str, int]:
    text, n1 = EMAIL_RE.subn("<EMAIL>", text)
    text, n2 = PHONE_RE.subn("<PHONE>", text)
    return text, n1 + n2


def pii_scrub(docs: list[dict], report: Optional[CurationReport] = None) -> list[dict]:
    out, n_hits, caught = [], 0, Counter()
    for d in docs:
        text, hits = scrub_pii(d["text"])
        if hits and d.get("planted"):
            caught[d["planted"]] += 1
        n_hits += hits
        out.append(dict(d, text=text))
    if report is not None:
        report.stages.append(StageReport("pii_scrub (rewrites)", len(out), 0, {"pii_replaced": n_hits}, dict(caught)))
    return out


# 5. exact deduplication ----------------------------------------------------
def exact_dedup(docs: list[dict], report: Optional[CurationReport] = None) -> list[dict]:
    seen: set[str] = set()

    def keep(d: dict) -> Optional[str]:
        h = hashlib.sha1(d["text"].strip().lower().encode()).hexdigest()
        if h in seen:
            return "exact_duplicate"
        seen.add(h)
        return None
    return _run_stage("exact_dedup", docs, keep, report)


# 6. fuzzy deduplication with MinHash + LSH ----------------------------------
def shingles(text: str, n: int = 3) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def minhash_signature(shingle_set: set[str], n_hashes: int = 64, seed: int = 0) -> np.ndarray:
    """For each of n_hashes hash functions, the minimum hash over the shingles.
    Two documents' signatures agree in ≈ Jaccard(A, B) fraction of positions."""
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2**31 - 1, size=n_hashes, dtype=np.int64)
    b = rng.randint(0, 2**31 - 1, size=n_hashes, dtype=np.int64)
    prime = 2**31 - 1
    if not shingle_set:
        return np.full(n_hashes, prime, dtype=np.int64)
    hashes = np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % prime for s in shingle_set], dtype=np.int64)
    # (n_shingles, n_hashes) -> min over shingles
    return ((hashes[:, None] * a[None, :] + b[None, :]) % prime).min(axis=0)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def minhash_dedup(docs: list[dict], threshold: float = 0.8, n_hashes: int = 64, bands: int = 16,
                  report: Optional[CurationReport] = None) -> list[dict]:
    """Drop documents whose estimated Jaccard similarity to an earlier one is >= threshold.

    LSH: split the signature into ``bands``; documents sharing any identical band are
    candidates, and only candidates are compared exactly — that is what makes this
    scale to billions of documents.
    """
    rows = n_hashes // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    sigs, shingle_sets = [], []
    for d in docs:
        s = shingles(d["text"])
        shingle_sets.append(s)
        sigs.append(minhash_signature(s, n_hashes))
    drop: dict[int, str] = {}
    for i, sig in enumerate(sigs):
        if i in drop:
            continue
        candidates: set[int] = set()
        keys = []
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            keys.append(key)
            candidates.update(buckets[key])
        is_dup = any(j not in drop and jaccard(shingle_sets[i], shingle_sets[j]) >= threshold for j in candidates)
        if is_dup:
            drop[i] = "near_duplicate"
        else:
            for key in keys:
                buckets[key].append(i)
    return _run_stage("minhash_dedup", docs, _IndexReason(docs, drop), report)


class _IndexReason:
    """Helper so _run_stage can look up drop reasons by position."""

    def __init__(self, docs: list[dict], drop: dict[int, str]) -> None:
        self.pos = {id(d): i for i, d in enumerate(docs)}
        self.drop = drop

    def __call__(self, d: dict) -> Optional[str]:
        return self.drop.get(self.pos[id(d)])


# 7. model-based quality classifier ------------------------------------------
class QualityClassifier:
    """A tiny logistic-regression classifier on hashed bag-of-words features.

    This is the miniature of FineWeb-Edu's "educational value" classifier: label a few
    hundred documents (here: clean Storyland = good, planted junk = bad), train, and
    score everything. Balance the two classes when you build the label set: an
    unbalanced set (few "good", many "bad") shifts the bias and rejects clean documents. Real pipelines use an LLM to label ~500k documents, then train a
    small model to imitate it (that is *distillation of a labeling function*).
    """

    def __init__(self, n_features: int = 4096, seed: int = 0) -> None:
        self.n = n_features
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.seed = seed

    def featurize(self, text: str) -> np.ndarray:
        x = np.zeros(self.n)
        for w in re.findall(r"\w+|[^\w\s]", text.lower()):
            x[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.n] += 1
        return x / max(1.0, np.sqrt((x ** 2).sum()))

    def fit(self, texts: Sequence[str], labels: Sequence[int], epochs: int = 30, lr: float = 1.0) -> "QualityClassifier":
        X = np.stack([self.featurize(t) for t in texts])
        y = np.asarray(labels, dtype=float)
        for _ in range(epochs):                      # plain gradient descent on log-loss
            p = 1 / (1 + np.exp(-(X @ self.w + self.b)))
            g = p - y
            self.w -= lr * (X.T @ g) / len(y)
            self.b -= lr * g.mean()
        return self

    def score(self, text: str) -> float:
        return float(1 / (1 + np.exp(-(self.featurize(text) @ self.w + self.b))))


def quality_filter(docs: list[dict], clf: QualityClassifier, min_score: float = 0.5,
                   report: Optional[CurationReport] = None) -> list[dict]:
    return _run_stage("quality_classifier", docs,
                      lambda d: None if clf.score(d["text"]) >= min_score else "low_quality", report)


# 8. decontamination ---------------------------------------------------------
def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def decontaminate(docs: list[dict], eval_texts: Sequence[str], n: int = 8,
                  report: Optional[CurationReport] = None) -> list[dict]:
    """Drop any document sharing an n-gram with an evaluation item (n-gram overlap,
    the method used by GPT-3/Llama reports; 8 words here, 13 in production)."""
    banned: set[tuple[str, ...]] = set()
    for t in eval_texts:
        banned |= ngrams(t, n)
    return _run_stage("decontaminate", docs,
                      lambda d: "eval_overlap" if ngrams(d["text"], n) & banned else None, report)


# 9. mixing -----------------------------------------------------------------
def mix_sources(docs: list[dict], weights: dict[str, float], n_out: int, seed: int = 0) -> list[dict]:
    """Sample ``n_out`` documents with per-source probabilities (``weights`` need not sum to 1).
    Sources with weight 0 are excluded. Sampling is with replacement, so a small
    high-weight source is *up-sampled* (repeated), as real mixes do for e.g. Wikipedia."""
    rng = random.Random(seed)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_src[d.get("source", "unknown")].append(d)
    srcs = [s for s in by_src if weights.get(s, 0) > 0]
    probs = [weights[s] for s in srcs]
    return [rng.choice(by_src[rng.choices(srcs, probs)[0]]) for _ in range(n_out)]


# 10. full pipeline ------------------------------------------------------------
def curate(raw_docs: list[dict], eval_texts: Sequence[str] = (), clf: Optional[QualityClassifier] = None,
           report: Optional[CurationReport] = None) -> list[dict]:
    docs = normalize(raw_docs)
    docs = language_filter(docs, report=report)
    docs = heuristic_filter(docs, report=report)
    docs = pii_scrub(docs, report=report)
    docs = exact_dedup(docs, report=report)
    docs = minhash_dedup(docs, report=report)
    if clf is not None:
        docs = quality_filter(docs, clf, report=report)
    if eval_texts:
        docs = decontaminate(docs, eval_texts, report=report)
    return docs


# ========================================================= tokenize and pack
def tokenize_and_pack(docs: Iterable[dict], tok, eos_token: str = "<|eos|>") -> torch.Tensor:
    """Concatenate all documents into one long 1-D tensor of ids, separated by EOS.

    Training takes random windows of ``seq_len + 1`` from this stream. Packing wastes
    nothing on padding; the EOS token tells the model where documents end.
    """
    eos = tok.special_tokens[eos_token]
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(d["text"], allowed_special=False))
        ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


def split_train_val(tokens: torch.Tensor, val_frac: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = int(len(tokens) * val_frac)
    return tokens[:-n_val], tokens[-n_val:]


def corpus_text(docs: Iterable[dict]) -> str:
    return "\n\n".join(d["text"] for d in docs)


# ============================================================== real text
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tinyshakespeare(cache_dir: str = "runs/data") -> Optional[str]:
    """Download Karpathy's tiny Shakespeare (1.1 MB) for an optional 'real data' run.
    Returns the file path, or None if the network is unavailable."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tinyshakespeare.txt")
    if os.path.exists(path):
        return path
    try:
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        return path
    except Exception as e:  # noqa: BLE001
        print(f"[data] download failed ({e}); continuing with Storyland only")
        return None
