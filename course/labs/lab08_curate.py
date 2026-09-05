"""Lab 08: curate a planted-noise corpus.

Runs the miniature pretraining-data pipeline from ``llm/data.py`` over a Storyland
corpus into which we have planted the problems a real Common Crawl dump has:
exact and near duplicates, spam, non-English text, personal data, fragments, and
leaked evaluation questions. Every stage reports what it removed and which planted
problems it caught. Then we look inside MinHash, mix sources, and pack tokens.

    python3 labs/lab08_curate.py            # --quick (default): 1,500 clean documents
    python3 labs/lab08_curate.py --full     # 6,000 clean documents
"""
from _common import setup, check, banner, section, savefig, done, plt

import random
import time
from collections import Counter, defaultdict

import numpy as np

from llm.data import (make_corpus, add_noise, curate, CurationReport, QualityClassifier,
                      english_score, gopher_reason, scrub_pii, shingles, minhash_signature,
                      jaccard, decontaminate, mix_sources, tokenize_and_pack, OBJECTS, COLORS,
                      NAMES, PLACES)
from llm.pipeline import get_tokenizer

args = setup("Lab 08: curate a planted-noise corpus")
n_docs = 1500 if args.quick else 6000
rng = random.Random(args.seed)

# Three evaluation questions we pretend belong to a benchmark. They must never be
# trained on; ``add_noise`` leaks them into the corpus the way benchmark text leaks
# onto the web. Each has >= 8 words so the 8-gram decontaminator can see them.
EVAL_QUESTIONS = [
    "What color was the kite that Mia carried up the tallest hill in Storyland?",
    "Which animal returned the lost silver bell to Leo at the old stone bridge?",
    "How many pears did Zoe count in the purple box beside the frozen lake?",
]
PLANTED = ["exact_dup", "near_dup", "spam", "non_english", "pii", "too_short", "contaminated"]

# ---------------------------------------------------------------- 1. the corpus
section("1. a clean corpus, then a dirty one")
clean = make_corpus(n_docs, seed=args.seed)
dirty = add_noise(clean, seed=args.seed, eval_questions=EVAL_QUESTIONS)
planted_counts = Counter(d.get("planted") for d in dirty if d.get("planted"))
print(f"clean docs: {len(clean):,}   dirty docs: {len(dirty):,}   (+{len(dirty) - len(clean):,} planted)")
for k in PLANTED:
    print(f"  planted {k:<13} {planted_counts[k]:>5}")
print("\nexample clean doc:\n  " + clean[0]["text"].replace("\n", "\n  "))
spam_example = next(d for d in dirty if d.get("planted") == "spam")
print("example planted spam:\n  " + spam_example["text"][:110] + "...")

# ------------------------------------------------- 2. the stages, one at a time
section("2. what each stage looks at")
en = next(d for d in dirty if d.get("planted") is None)["text"]
de = next(d for d in dirty if d.get("planted") == "non_english")["text"]
print(f"english_score(English story) = {english_score(en):.2f}   english_score(German/Spanish) = {english_score(de):.2f}")
print(f"gopher_reason(spam)          = {gopher_reason(spam_example['text'])!r}")
print(f"gopher_reason('ok')          = {gopher_reason('ok')!r}")
print(f"gopher_reason(clean story)   = {gopher_reason(en)!r}")
pii_doc = next(d for d in dirty if d.get("planted") == "pii")["text"]
scrubbed, n_hits = scrub_pii(pii_doc)
print(f"scrub_pii: replaced {n_hits} items -> ...{scrubbed[-60:]!r}")
# Filters have false positives. Measure them on data you trust: the CLEAN corpus.
bare_math = next(d for d in clean if d["source"] == "math" and "\n" not in d["text"])
print(f"english_score({bare_math['text']!r}) = {english_score(bare_math['text']):.2f}  <- no stopwords at all")
fp_report = CurationReport()
survivors_clean = curate(clean, report=fp_report)
print("\nthe pipeline run on the CLEAN corpus (every drop here is a false positive):")
print(fp_report.table())
by_src_lost = Counter(d["source"] for d in clean) - Counter(d["source"] for d in survivors_clean)
print(f"clean docs lost by source: {dict(by_src_lost)}   (bare 'a + b = c' lines have no English stopwords)")

# ------------------------------------------- 3. a model-based quality classifier
section("3. train a quality classifier on a small labelled set")
t0 = time.perf_counter()
junk = [d["text"] for d in dirty if d.get("planted") in ("spam", "non_english", "too_short")]
n_lab = min(300, int(len(junk) * 0.6))          # same number of good and bad: an unbalanced set
good, bad = [d["text"] for d in clean[:n_lab]], junk[:n_lab]   # learns "most docs are X" instead
clf = QualityClassifier().fit(good + bad, [1] * len(good) + [0] * len(bad))
held_good, held_bad = [d["text"] for d in clean[n_lab:n_lab + 300]], junk[n_lab:]
acc_good = np.mean([clf.score(t) >= 0.5 for t in held_good])
acc_bad = np.mean([clf.score(t) < 0.5 for t in held_bad])
print(f"labelled {len(good)} good + {len(bad)} bad docs; trained in {time.perf_counter() - t0:.2f}s")
print(f"held-out: {acc_good:.0%} of clean docs scored >= 0.5, {acc_bad:.0%} of junk scored < 0.5")
print(f"score(clean story) = {clf.score(en):.3f}   score(spam) = {clf.score(spam_example['text']):.3f}")
for thr in (0.3, 0.5, 0.7):
    kept_frac = np.mean([clf.score(t) >= thr for t in held_good])
    rej_frac = np.mean([clf.score(t) < thr for t in held_bad])
    print(f"  threshold {thr}: keeps {kept_frac:.0%} of clean, rejects {rej_frac:.0%} of junk")
check(acc_good > 0.95, "quality classifier keeps > 95% of held-out clean docs")
check(acc_bad > 0.9, "quality classifier rejects > 90% of held-out junk")

# ------------------------------------------------------- 4. the whole pipeline
section("4. run the pipeline with a CurationReport")
t0 = time.perf_counter()
report = CurationReport()
curated = curate(dirty, eval_texts=EVAL_QUESTIONS, clf=clf, report=report)
dt_curate = time.perf_counter() - t0
print(report.table())
print(f"\n{len(dirty):,} raw -> {len(curated):,} curated documents in {dt_curate:.1f}s")

caught_total: Counter = Counter()
for s in report.stages:
    caught_total.update(s.caught_planted)
survivors = Counter(d.get("planted") for d in curated if d.get("planted"))
print("\n'caught' counts documents carrying the tag that a stage removed (or, for pii, rewrote);")
print("for duplicates the kept twin may carry the tag, so 'left' > 0 is expected there.")
print(f"\n{'planted problem':<14}{'planted':>9}{'caught':>8}{'left':>10}  caught by")
for k in PLANTED:
    by = [s.name.split(" ")[0] for s in report.stages if s.caught_planted.get(k)]
    print(f"{k:<14}{planted_counts[k]:>9}{caught_total[k]:>8}{survivors[k]:>10}  {', '.join(by)}")
    check(caught_total[k] > 0, f"planted {k} was caught by at least one stage")
for k in ("spam", "non_english", "too_short", "contaminated"):
    check(survivors[k] == 0, f"no planted {k} survives curation")
# For duplicates, "survived" is the wrong question: the corpus was shuffled, so the copy may
# come first and be the one kept. What matters is that each text appears once.
by_origin: dict[str, list[dict]] = defaultdict(list)
for d in curated:
    if d.get("orig_id"):
        by_origin[d["orig_id"]].append(d)
leftover_pairs = [(a, b) for v in by_origin.values() for a in v for b in v if a is not b and id(a) < id(b)]
leftover_j = [jaccard(shingles(a["text"]), shingles(b["text"])) for a, b in leftover_pairs]
print(f"\ndistinct texts: {len({d['text'] for d in curated}):,} of {len(curated):,};  "
      f"origins with >1 surviving copy: {sum(len(v) > 1 for v in by_origin.values())}")
if leftover_j:
    ex_a, ex_b = leftover_pairs[int(np.argmax(leftover_j))]
    print(f"  the closest surviving pair has Jaccard {max(leftover_j):.2f} (< 0.8 threshold), "
          f"{len(ex_a['text'].split())} words: swapping one word in a short doc changes a large share of its shingles")
check(len({d["text"] for d in curated}) == len(curated), "every curated text is unique (exact dedup)")
check(not leftover_j or max(leftover_j) < 0.8, "no surviving pair has Jaccard >= 0.8 (MinHash-LSH missed nothing above threshold)")
check(all("@" not in d["text"] for d in curated), "no e-mail address survives PII scrubbing")
n_clean_left = sum(1 for d in curated if d.get("planted") is None)
print(f"\nclean documents that survived: {n_clean_left:,} / {len(clean):,} "
      f"({len(clean) - n_clean_left:,} clean docs were dropped — mostly short math lines, see stage 2)")

# Decontamination on its own: the eval questions leaked into 3 docs.
kept = decontaminate(dirty, EVAL_QUESTIONS, n=8)
print(f"decontaminate alone drops {len(dirty) - len(kept)} of {len(dirty):,} raw docs (the 3 leaked eval items)")
check(len(dirty) - len(kept) == 3, "8-gram decontamination removes exactly the 3 leaked eval questions")

# funnel figure
fig, ax = plt().subplots(figsize=(8, 3.6))
names = [s.name.replace(" (rewrites)", "") for s in report.stages]
dropped = [s.dropped for s in report.stages]
ax.bar(names, dropped, color="#2563eb")
for i, v in enumerate(dropped):
    ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("documents removed")
ax.set_title(f"Lab 08: documents removed per stage ({len(dirty):,} raw -> {len(curated):,} kept)")
ax.tick_params(axis="x", rotation=25)
savefig(fig, "lab08_funnel.png")

# --------------------------------------------------------- 5. inside MinHash
section("5. MinHash: estimated vs true Jaccard on 200 pairs")
stories = [d["text"] for d in clean if d["source"] == "stories"]
vocab = OBJECTS + COLORS + NAMES + PLACES
pairs = []
for i in range(200):
    a = rng.choice(stories)
    if i % 4 == 0:                                     # 1 in 4: two unrelated documents
        b = rng.choice(stories)
    else:                                              # 3 in 4: a copy with k words replaced
        words = a.split()
        k = rng.randint(0, len(words) // 2)
        for j in rng.sample(range(len(words)), k):
            words[j] = rng.choice(vocab)
        b = " ".join(words)
    pairs.append((a, b))
true_j, est_j = [], []
for a, b in pairs:
    sa, sb = shingles(a), shingles(b)
    true_j.append(jaccard(sa, sb))
    est_j.append(float((minhash_signature(sa, 64) == minhash_signature(sb, 64)).mean()))
true_j, est_j = np.array(true_j), np.array(est_j)
mae = np.abs(true_j - est_j).mean()
print(f"64-hash signatures: mean |estimate - true| = {mae:.3f}, max = {np.abs(true_j - est_j).max():.3f}")
print(f"theory: std of the estimate ≈ sqrt(J(1-J)/64) = {np.sqrt(0.5 * 0.5 / 64):.3f} at J = 0.5")
check(mae < 0.08, "MinHash estimate is within 0.08 of true Jaccard on average")

print("\nLSH with 64 hashes = 16 bands × 4 rows: P(candidate) = 1 - (1 - J^4)^16")
for J in (0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
    print(f"  J = {J:.1f}  ->  P(candidate) = {1 - (1 - J ** 4) ** 16:.4f}")

fig, ax = plt().subplots(figsize=(4.8, 4.4))
ax.scatter(true_j, est_j, s=14, alpha=0.7, color="#2563eb")
ax.plot([0, 1], [0, 1], color="#64748b", lw=1, ls="--")
ax.axvline(0.8, color="#dc2626", lw=1, ls=":", label="dedup threshold 0.8")
ax.set_xlabel("true Jaccard (3-word shingles)")
ax.set_ylabel("MinHash estimate (64 hashes)")
ax.set_title("Lab 08: MinHash estimates Jaccard")
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "lab08_minhash.png")

# ------------------------------------------------------------ 6. mixing sources
section("6. mix sources")
by_src = Counter(d.get("source") for d in curated)
print(f"curated docs by source: {dict(by_src)}")
n_mix = 2000
mixed = mix_sources(curated, {"stories": 1.0, "math": 3.0}, n_out=n_mix, seed=args.seed)
mix_counts = Counter(d["source"] for d in mixed)
uniq_math = len({d["id"] for d in mixed if d["source"] == "math"})
print(f"mix_sources(weights stories:1, math:3, n_out={n_mix}): {dict(mix_counts)}")
print(f"  -> math is {mix_counts['math'] / n_mix:.0%} of the mix; the {uniq_math} distinct math docs "
      f"were up-sampled ~{mix_counts['math'] / max(1, uniq_math):.1f}× each")
check(0.65 < mix_counts["math"] / n_mix < 0.85, "math is ~75% of the mix (weights 1:3)")

# ---------------------------------------------------------- 7. tokenize & pack
section("7. tokenize and pack")
tok = get_tokenizer()
t0 = time.perf_counter()
tokens = tokenize_and_pack(curated, tok)
dt_tok = time.perf_counter() - t0
per_source: dict[str, int] = defaultdict(int)
for d in curated:
    per_source[d.get("source", "?")] += len(tok.encode(d["text"], allowed_special=False)) + 1
print(f"packed {len(tokens):,} tokens from {len(curated):,} docs in {dt_tok:.1f}s "
      f"({len(tokens) / len(curated):.1f} tokens/doc incl. <|eos|>)")
for s, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
    print(f"  {s:<10} {n:>9,} tokens  ({n / len(tokens):.0%})")
eos = tok.special_tokens["<|eos|>"]
check(int((tokens == eos).sum()) == len(curated), "exactly one <|eos|> per document in the packed stream")
check(sum(per_source.values()) == len(tokens), "per-source token counts add up to the packed length")

done()
