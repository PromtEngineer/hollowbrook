"""Lab 1: language models are next-token predictors — build one by counting.

We build unigram, bigram and trigram models with plain Python dicts on the Storyland
corpus, print probability tables, generate text, measure perplexity on held-out
documents, and watch the sparsity problem appear.

    python3 labs/lab01_ngram.py            # --quick (default): Storyland only, < 60 s
    python3 labs/lab01_ngram.py --full     # + 4/5-gram, a training-size sweep, and
                                           #   tiny Shakespeare if it can be downloaded
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import os
import random
import re
from collections import Counter, defaultdict

from llm.pipeline import get_corpus
from llm.data import download_tinyshakespeare

args = setup("Lab 01: next-token prediction by counting")
rng = random.Random(args.seed)

# ------------------------------------------------------------------ tokens
BOS, EOS = "<s>", "</s>"                       # start / end of a document


def words(text: str) -> list[str]:
    """A word-level tokenizer for this chapter: words and punctuation marks.
    (Chapter 2 replaces this with byte-level BPE.)"""
    return re.findall(r"[A-Za-z']+|\d+|[^\w\s]", text)


def doc_tokens(text: str, n: int) -> list[str]:
    """Pad each document with n-1 start tokens so the first real word has a context."""
    return [BOS] * (n - 1) + words(text) + [EOS]


# --------------------------------------------------------------- the model
class NGramModel:
    """A count-based n-gram language model with add-k smoothing.

    counts[context][next] = how often ``next`` followed ``context`` (a tuple of n-1 tokens)
    """

    def __init__(self, n: int, k: float = 0.01) -> None:
        self.n, self.k = n, k
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, texts: list[str]) -> "NGramModel":
        for t in texts:
            toks = doc_tokens(t, self.n)
            self.vocab.update(toks)
            for i in range(self.n - 1, len(toks)):
                ctx = tuple(toks[i - self.n + 1:i])
                self.counts[ctx][toks[i]] += 1
        return self

    def prob(self, ctx: tuple, nxt: str) -> float:
        """P(next | context) = (count(ctx, next) + k) / (count(ctx) + k * V)"""
        c = self.counts.get(ctx, Counter())
        V = len(self.vocab)
        return (c[nxt] + self.k) / (sum(c.values()) + self.k * V)

    def top(self, ctx: tuple, m: int = 5) -> list[tuple[str, float]]:
        c = self.counts.get(ctx, Counter())
        return [(w, self.prob(ctx, w)) for w, _ in c.most_common(m)]

    def log2_prob_doc(self, text: str) -> tuple[float, int]:
        """Chain rule: log2 P(doc) = sum_i log2 P(w_i | w_{i-n+1..i-1}). Returns (bits, n_tokens)."""
        toks = doc_tokens(text, self.n)
        bits, n = 0.0, 0
        for i in range(self.n - 1, len(toks)):
            ctx = tuple(toks[i - self.n + 1:i])
            bits += math.log2(self.prob(ctx, toks[i]))
            n += 1
        return bits, n

    def perplexity(self, texts: list[str]) -> float:
        """2 ** (average bits per token): the 'effective branching factor'."""
        total_bits, total_n = 0.0, 0
        for t in texts:
            b, n = self.log2_prob_doc(t)
            total_bits += b
            total_n += n
        return 2 ** (-total_bits / total_n)

    def generate(self, max_tokens: int = 40, seed: int = 0, temperature: float = 1.0) -> str:
        r = random.Random(seed)
        toks = [BOS] * (self.n - 1)
        for _ in range(max_tokens):
            ctx = tuple(toks[len(toks) - self.n + 1:]) if self.n > 1 else ()
            c = self.counts.get(ctx)
            if not c:
                break
            cand, wts = zip(*c.items())
            if temperature != 1.0:
                wts = [w ** (1.0 / temperature) for w in wts]
            nxt = r.choices(cand, weights=wts)[0]
            if nxt == EOS:
                break
            toks.append(nxt)
        return " ".join(t for t in toks if t != BOS)

    def unseen_fraction(self, texts: list[str]) -> float:
        """Fraction of held-out (context, next) pairs never seen in training."""
        unseen, total = 0, 0
        for t in texts:
            toks = doc_tokens(t, self.n)
            for i in range(self.n - 1, len(toks)):
                ctx = tuple(toks[i - self.n + 1:i])
                total += 1
                if self.counts.get(ctx, Counter())[toks[i]] == 0:
                    unseen += 1
        return unseen / total


# ---------------------------------------------------------------- corpus
section("corpus")
docs = get_corpus()                                    # 6000 Storyland documents
texts = [d["text"] for d in docs]
n_train = int(0.9 * len(texts))
train_texts, test_texts = texts[:n_train], texts[n_train:]
n_tok_train = sum(len(words(t)) for t in train_texts)
print(f"{len(train_texts)} training docs, {len(test_texts)} held-out docs, {n_tok_train:,} training word-tokens")
print("first document:\n  " + train_texts[0][:200].replace("\n", "\n  "))

# ------------------------------------------------------------ fit models
section("fit unigram / bigram / trigram by counting")
models = {n: NGramModel(n).fit(train_texts) for n in (1, 2, 3)}
V = len(models[1].vocab)
print(f"vocabulary: {V} distinct word-tokens (incl. <s> and </s>)")
for n, m in models.items():
    print(f"{n}-gram: {len(m.counts):>7,} distinct contexts, "
          f"{sum(len(c) for c in m.counts.values()):>8,} distinct (context, next) pairs")

# ---------------------------------------------------- probability tables
section("probability tables: P(next | context)")
uni = models[1].top((), 8)
print("unigram  P(w)              : " + ", ".join(f"{w}={p:.3f}" for w, p in uni))
for ctx in [("the",), ("Mia",), ("was",)]:
    row = models[2].top(ctx, 6)
    print(f"bigram   P(w | {ctx[0]!r:8})  : " + ", ".join(f"{w}={p:.3f}" for w, p in row))
for ctx in [("Mia", "had"), ("the", "sun"), ("was", "very")]:
    row = models[3].top(ctx, 6)
    print(f"trigram  P(w | {ctx[0]!r},{ctx[1]!r}): " + ", ".join(f"{w}={p:.3f}" for w, p in row))
check(abs(sum(p for _, p in uni) - 1.0) < 1.0, "probabilities are in [0, 1]")
check(models[3].prob(("the", "sun"), "went") > models[2].prob(("sun",), "went") * 0.99,
      "trigram context sharpens the prediction of 'went' after 'the sun'")

# --------------------------------------------- the chain rule on one doc
section("chain rule on one held-out sentence")
sent = "Mia had a red kite."
toks = doc_tokens(sent, 3)
print(f"{'token':<8}{'context':<22}{'P(bigram)':>10}{'P(trigram)':>12}{'bits(tri)':>10}")
total_bits = 0.0
for i in range(2, len(toks)):
    ctx2, ctx3 = (toks[i - 1],), (toks[i - 2], toks[i - 1])
    p2, p3 = models[2].prob(ctx2, toks[i]), models[3].prob(ctx3, toks[i])
    total_bits += -math.log2(p3)
    print(f"{toks[i]:<8}{' '.join(ctx3):<22}{p2:>10.4f}{p3:>12.4f}{-math.log2(p3):>10.2f}")
print(f"log2 P(sentence) under trigram = {-total_bits:.2f} bits  ->  P = {2 ** -total_bits:.2e}")
print(f"per-token: {total_bits / (len(toks) - 2):.2f} bits  ->  perplexity {2 ** (total_bits / (len(toks) - 2)):.2f}")

# --------------------------------------------------------- perplexity
section("perplexity on held-out documents (lower is better)")
ppl = {n: m.perplexity(test_texts) for n, m in models.items()}
uniform_ppl = NGramModel(1, k=1e9).fit(train_texts).perplexity(test_texts)   # huge k -> uniform
print(f"uniform guess : {uniform_ppl:8.2f}   (= V = {V}: every word equally likely)")
for n in (1, 2, 3):
    print(f"{n}-gram        : {ppl[n]:8.2f}")
check(abs(uniform_ppl - V) < 1.0, "a uniform model has perplexity = vocabulary size")
check(ppl[2] < ppl[1], f"bigram ({ppl[2]:.1f}) beats unigram ({ppl[1]:.1f})")
check(ppl[3] < ppl[2], f"trigram ({ppl[3]:.1f}) beats bigram ({ppl[2]:.1f})")

# --------------------------------------------------------- generation
section("generate text by sampling from the counts")
for n in (1, 2, 3):
    for s in range(2):
        print(f"{n}-gram #{s}: {models[n].generate(30, seed=args.seed + s)}")

# ---------------------------------------------------------- sparsity
section("the sparsity problem: held-out n-grams never seen in training")
unseen = {n: m.unseen_fraction(test_texts) for n, m in models.items()}
for n in (1, 2, 3):
    print(f"{n}-gram: {100 * unseen[n]:5.2f}% of held-out (context, next) pairs were never seen")
check(unseen[3] >= unseen[2] >= unseen[1], "longer contexts -> more unseen combinations")

# training-size sweep: how much data does each order need?
sizes = [100, 300, 1000, 3000, n_train] if args.quick else [30, 100, 300, 1000, 3000, n_train]
orders = (1, 2, 3) if args.quick else (1, 2, 3, 4, 5)
section(f"perplexity vs training size (orders {orders})")
print(f"{'train docs':>10}" + "".join(f"{n}-gram ppl".rjust(12) for n in orders)
      + "".join(f"unseen{n}".rjust(9) for n in orders))
sweep: dict[int, list[float]] = defaultdict(list)
for sz in sizes:
    ms = {n: NGramModel(n).fit(train_texts[:sz]) for n in orders}
    row = f"{sz:>10}"
    for n in orders:
        p = ms[n].perplexity(test_texts)
        sweep[n].append(p)
        row += f"{p:>12.1f}"
    for n in orders:
        row += f"{100 * ms[n].unseen_fraction(test_texts):>8.1f}%"
    print(row)
if not args.quick:
    check(sweep[3][0] > sweep[2][0], "with only 30 docs the trigram is WORSE than the bigram (too sparse)")
check(sweep[3][-1] < sweep[2][-1], "with the full corpus the trigram wins")

# ------------------------------------------------ real text (full mode)
shakespeare_ppl = None
if not args.quick:
    section("real text: tiny Shakespeare")
    path = download_tinyshakespeare()
    if path is None:
        print("(no network: skipping Shakespeare)")
    else:
        lines = [l for l in open(path, encoding="utf-8").read().split("\n\n") if l.strip()]
        n_tr = int(0.9 * len(lines))
        sh_train, sh_test = lines[:n_tr], lines[n_tr:]
        print(f"{len(sh_train)} training paragraphs, {len(sh_test)} held-out, "
              f"{sum(len(words(t)) for t in sh_train):,} word-tokens, "
              f"{len(set(w for t in sh_train for w in words(t))):,} distinct words")
        shakespeare_ppl = {}
        for n in (1, 2, 3, 4):
            m = NGramModel(n).fit(sh_train)
            shakespeare_ppl[n] = m.perplexity(sh_test)
            print(f"{n}-gram: perplexity {shakespeare_ppl[n]:9.1f}   unseen {100 * m.unseen_fraction(sh_test):5.1f}%"
                  f"   sample: {m.generate(14, seed=args.seed)!r}")
        check(shakespeare_ppl[3] > ppl[3] * 5, "Shakespeare is far harder to predict than Storyland")

# ----------------------------------------------------------- figures
p = plt()
fig, axes = p.subplots(1, 3, figsize=(13, 3.6))
ctx = ("had", "a")
top = models[3].top(ctx, 10)
axes[0].barh([w for w, _ in top][::-1], [q for _, q in top][::-1], color="#2563eb")
axes[0].set_title("trigram P(next | 'had a')")
axes[0].set_xlabel("probability")
axes[1].bar([f"{n}-gram" for n in (1, 2, 3)], [ppl[n] for n in (1, 2, 3)], color=["#64748b", "#f59e0b", "#16a34a"])
axes[1].axhline(uniform_ppl, ls="--", color="#dc2626", label=f"uniform = V = {V}")
axes[1].set_yscale("log")
axes[1].set_title("held-out perplexity (log scale)")
axes[1].legend()
for n in orders:
    axes[2].plot(sizes, sweep[n], marker="o", label=f"{n}-gram")
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("training documents")
axes[2].set_ylabel("held-out perplexity")
axes[2].set_title("more context needs more data")
axes[2].legend()
fig.tight_layout()
savefig(fig, "lab01_ngram.png")
done()
