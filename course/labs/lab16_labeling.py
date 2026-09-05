"""Lab 16: labeling and curation for post-training — build a preference set and measure its quality.

    python3 labs/lab16_labeling.py            # --quick: 30 prompts, < 60 s
    python3 labs/lab16_labeling.py --full     # 150 prompts, a flip-rate sweep, on-policy pairs

What you will see:
  1. a synthetic preference set (chosen = reference answer, rejected = a plausible failure);
  2. two simulated annotators — one perfect, one that flips labels at a given rate — and
     Cohen's kappa between them, implemented here in a dozen lines and checked against the
     interactive's worked example;
  3. a position-bias check on three judges (evals.position_bias_check): the rule-based judge
     passes, a "first answer wins" judge fails, a "longer answer wins" judge has verbosity bias;
  4. exact + MinHash deduplication of instruction prompts (llm.data), with planted duplicates;
  5. re-mixing the surviving prompts to a target task mix (data.mix_sources);
  6. the labels written to runs/preferences.jsonl in the format interactive/16_labeling_tool.html
     exports, read back, and turned into PreferencePair objects again;
  7. (if runs/sft_*.pt exists) on-policy pairs by rejection sampling from the SFT model.
"""
from _common import setup, check, banner, section, savefig, done, plt

import json
import os
import random
import time
from collections import Counter

from llm.data import exact_dedup, minhash_dedup, mix_sources, shingles, jaccard, CurationReport
from llm.evals import rule_based_judge, position_bias_check
from llm.model import TinyLM
from llm.pipeline import get_tokenizer, get_base_model, run_path
from llm.reward import PreferencePair, make_preference_pairs, make_preference_pairs_from_model, WRONG_STYLES
from llm.tasks import make_examples

args = setup("Lab 16: labeling and curation — annotators, agreement, judges, dedup")
rng = random.Random(args.seed)
TASKS = ["upper", "reverse", "add", "count"]
N_PROMPTS = 30 if args.quick else 150
FLIP = 0.2                                    # the noisy annotator's flip rate

# ------------------------------------------------------------ 1. a preference set
section(f"1. synthetic preference pairs from {N_PROMPTS} prompts (reward.make_preference_pairs)")
examples = make_examples(N_PROMPTS, seed=args.seed, tasks=TASKS)
pairs = make_preference_pairs(examples, n_wrong_styles=2, seed=args.seed)
print(f"{len(examples)} prompts -> {len(pairs)} pairs (2 failure styles per prompt)")
print("failure styles:", dict(Counter(p.meta["style"] for p in pairs)))
for p in pairs[:5]:
    print(f"  [{p.meta['task']:<7}|{p.meta['style']:<10}] {p.prompt_messages[-1]['content']!r}")
    print(f"      chosen  : {p.chosen!r}\n      rejected: {p.rejected!r}")
check(len(pairs) == 2 * len(examples), "one pair per (prompt, failure style)")
check(all(p.chosen != p.rejected for p in pairs), "chosen and rejected always differ")


# ------------------------------------------------------------ 2. two annotators + kappa
def cohens_kappa(labels1: list[str], labels2: list[str]) -> tuple[float, float, float]:
    """kappa = (p_o - p_e) / (1 - p_e): agreement beyond what label frequencies predict."""
    assert len(labels1) == len(labels2) and labels1
    n = len(labels1)
    p_o = sum(a == b for a, b in zip(labels1, labels2)) / n            # observed agreement
    c1, c2 = Counter(labels1), Counter(labels2)
    p_e = sum((c1[c] / n) * (c2[c] / n) for c in set(c1) | set(c2))      # chance agreement
    kappa = 1.0 if p_e == 1.0 else (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e


def present(pair: PreferencePair, rng: random.Random) -> dict:
    """What an annotator sees: the two answers in a random order, plus the hidden gold label."""
    chosen_first = rng.random() < 0.5
    a, b = (pair.chosen, pair.rejected) if chosen_first else (pair.rejected, pair.chosen)
    return {"prompt": pair.prompt_messages[-1]["content"], "response_a": a, "response_b": b,
            "gold": "A" if chosen_first else "B", "pair": pair}


def noisy_annotator(gold: str, flip: float, rng: random.Random) -> str:
    """Return the gold label with probability 1 - flip; otherwise the other answer, or (1 in 4 slips) a tie."""
    if rng.random() >= flip:
        return gold
    return "tie" if rng.random() < 0.25 else {"A": "B", "B": "A"}[gold]


section(f"2. two annotators: one perfect, one that flips {FLIP:.0%} of labels -> Cohen's kappa")
items = [present(p, rng) for p in pairs]
for k, it in enumerate(items):
    it["id"] = k + 1
labels_perfect = [it["gold"] for it in items]
labels_noisy = [noisy_annotator(it["gold"], FLIP, rng) for it in items]
kappa, p_o, p_e = cohens_kappa(labels_perfect, labels_noisy)
print(f"n = {len(items)} items | raw agreement p_o = {p_o:.3f} | chance agreement p_e = {p_e:.3f} | kappa = {kappa:.3f}")
conf = Counter(zip(labels_perfect, labels_noisy))
cats = ["A", "B", "tie"]
print("confusion (rows: perfect, cols: noisy)  " + "  ".join(f"{c:>4}" for c in cats))
for r in cats:
    print(f"{r:>38}  " + "  ".join(f"{conf[(r, c)]:>4}" for c in cats))
check(kappa < p_o, "kappa is below raw agreement: chance agreement is subtracted")
check(0.3 < kappa < 0.95, f"a {FLIP:.0%} flip rate gives kappa ~ {kappa:.2f} (Landis & Koch: 'moderate' to 'almost perfect')")

# the interactive's built-in example, by hand
ref = dict(zip(range(1, 13), "A B A A A B B A tie A A B".split()))
noisy = {1: "B", 2: "B", 3: "A", 4: "A", 5: "A", 6: "B", 7: "A", 8: "A", 9: "A", 10: "A", 11: "B", 12: "B"}
k12, po12, pe12 = cohens_kappa([ref[i] for i in range(1, 13)], [noisy[i] for i in range(1, 13)])
print(f"the interactive's 12-item example: p_o = {po12:.3f} (8/12), p_e = {pe12:.3f} (69/144), kappa = {k12:.3f}")
check(abs(po12 - 8 / 12) < 1e-9 and abs(pe12 - 69 / 144) < 1e-9, "matches the by-hand computation in the chapter")
check(abs(k12 - 0.36) < 0.01, "kappa = 0.36: 'fair' agreement despite 67% raw agreement")

flip_rates = [0.0, 0.1, 0.2, 0.3, 0.5] if args.quick else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
kappas = []
for f in flip_rates:
    r2 = random.Random(1000 + int(f * 100))
    kappas.append(cohens_kappa(labels_perfect, [noisy_annotator(g, f, r2) for g in labels_perfect])[0])
print("flip rate -> kappa: " + ", ".join(f"{f:.2f}->{k:.2f}" for f, k in zip(flip_rates, kappas)))
check(kappas[0] > 0.999, "a perfect copy has kappa = 1")
check(abs(kappas[-1]) < 0.35, f"flipping {flip_rates[-1]:.0%} of labels drives kappa toward 0 (or below)")

# ------------------------------------------------------------ 3. judges and their biases
section("3. LLM-as-labeler stand-ins: position and verbosity bias (evals.position_bias_check)")
first_wins = lambda prompt, a, b: "A"                                          # pure position bias
longer_wins = lambda prompt, a, b: "A" if len(a) > len(b) else ("B" if len(b) > len(a) else "tie")
judges = {"rule_based_judge (verifier)": rule_based_judge, "first_wins": first_wins, "longer_wins": longer_wins}
bias_rate, agree_rate, agree_verbose = {}, {}, {}
for name, fn in judges.items():
    res = [position_bias_check(fn, p.meta["example"], p.chosen, p.rejected) for p in pairs]
    bias_rate[name] = sum(r["position_bias"] for r in res) / len(res)
    # accuracy when the two answers are shown in a RANDOM order (as a labeling tool would show them)
    verdicts = [fn(it["pair"].meta["example"], it["response_a"], it["response_b"]) for it in items]
    agree_rate[name] = sum(v == it["gold"] for v, it in zip(verdicts, items)) / len(items)
    vb = [(v, it) for v, it in zip(verdicts, items) if it["pair"].meta["style"] == "verbose"]
    agree_verbose[name] = sum(v == it["gold"] for v, it in vb) / max(1, len(vb))
    print(f"  {name:<28} position-bias rate {bias_rate[name]:.2f} | accuracy vs gold (random order) {agree_rate[name]:.2f} "
          f"| ...on 'verbose' pairs {agree_verbose[name]:.2f}")
check(bias_rate["rule_based_judge (verifier)"] == 0.0, "a verifier-backed judge gives the same verdict in both orders")
check(bias_rate["first_wins"] == 1.0, "a judge that always picks the first answer fails the swap test on every pair")
check(agree_verbose["longer_wins"] == 0.0, "the length-loving judge prefers every verbose WRONG answer (verbosity bias)")
ex = position_bias_check(first_wins, pairs[0].meta["example"], pairs[0].chosen, pairs[0].rejected)
print(f"  example for first_wins: {ex}")

# ------------------------------------------------------------ 4. dedup the prompts
section("4. deduplicating instruction prompts: exact, then MinHash (llm.data)")
pool = make_examples(N_PROMPTS * 4, seed=args.seed + 7, tasks=TASKS + ["story_qa"])
docs = [{"text": ex.prompt, "source": ex.task, "planted": None} for ex in pool]
rng.shuffle(docs)                                     # shuffle the originals; planted copies go AFTER them
n_clean = len(docs)
# plant near-duplicates: the same story with one word of the question changed, and short prompts with a suffix
stories = [d for d in docs if d["source"] == "story_qa"][:8 if args.quick else 30]
for d in stories:
    docs.append({"text": d["text"].replace("What", "Which", 1).replace("?", " today?"), "source": "story_qa", "planted": "near_dup"})
for d in [d for d in docs if d["source"] == "upper"][:5]:
    docs.append({"text": d["text"] + " please", "source": "upper", "planted": "near_dup_short"})
report = CurationReport()
kept = exact_dedup(docs, report)
kept = minhash_dedup(kept, threshold=0.8, report=report)
print(f"{n_clean} generated prompts + {len(docs) - n_clean} planted near-duplicates = {len(docs)} -> {len(kept)} kept")
print(report.table())
n_exact = report.stages[0].dropped
print(f"exact duplicates among the GENERATED prompts: {n_exact} (a 55-word pool makes 'Write in capitals: kite' recur)")
s1, s2 = shingles("Write in capitals: kite"), shingles("Write in capitals: kite please")
print(f"why the short near-dups survive: Jaccard({s1}, {s2}) = {jaccard(s1, s2):.2f} < 0.8")
caught = report.stages[1].caught_planted
check(caught.get("near_dup", 0) == len(stories), f"MinHash caught all {len(stories)} planted story near-duplicates")
check(caught.get("near_dup_short", 0) == 0, "...and none of the 5 short ones: 3-word shingles on a 4-word prompt are too coarse")
check(n_exact > 0, "exact dedup removed repeated prompts the generator produced by chance")

# ------------------------------------------------------------ 5. mixing
section("5. re-mix the surviving prompts to a target task mix (data.mix_sources)")
before = Counter(d["source"] for d in kept)
target = {"add": 3, "count": 1, "reverse": 2, "upper": 2, "story_qa": 2}
mixed = mix_sources(kept, target, n_out=len(kept), seed=args.seed)
after = Counter(d["source"] for d in mixed)
print(f"{'task':<10}{'before':>8}{'target w':>10}{'after':>8}")
for t in sorted(before):
    print(f"{t:<10}{before[t]:>8}{target[t]:>10}{after[t]:>8}")
check(after["add"] > before["add"], "up-weighting 'add' 3:1 gives it a larger share (sampled with replacement)")

# ------------------------------------------------------------ 6. JSONL export / import
section("6. write runs/preferences.jsonl in the interactive's export format, then read it back")
RUBRIC_FOR_STYLE = {"wrong": [1, 3, 4], "off_by_one": [1, 3, 4], "empty": [1, 2, 4], "verbose": [1, 2, 3, 4], "junk": [1, 2, 3]}


def export_rows(annotator: str, labels: list[str]) -> list[dict]:
    rows = []
    for it, label in zip(items, labels):
        style = it["pair"].meta["style"]
        rows.append({"id": it["id"], "prompt": it["prompt"], "response_a": it["response_a"], "response_b": it["response_b"],
                     "label": label, "rubric": RUBRIC_FOR_STYLE[style] if label == it["gold"] else [],
                     "note": f"rejected style: {style}" if label == it["gold"] else "",
                     "seconds": round(rng.uniform(4, 20), 1), "annotator": annotator})
    return rows


path = run_path("preferences.jsonl")
path2 = run_path("preferences_annotator2.jsonl")
with open(path, "w") as f:
    for row in export_rows("sim_perfect", labels_perfect):
        f.write(json.dumps(row) + "\n")
with open(path2, "w") as f:
    for row in export_rows("sim_noisy", labels_noisy):
        f.write(json.dumps(row) + "\n")
print(f"wrote {os.path.relpath(path)} ({len(items)} lines) and {os.path.relpath(path2)}")
with open(path) as f:
    first_line = f.readline().strip()
print("first line:", first_line[:160] + ("..." if len(first_line) > 160 else ""))
FIELDS = ["id", "prompt", "response_a", "response_b", "label", "rubric", "note", "seconds", "annotator"]
check(list(json.loads(first_line).keys()) == FIELDS, "fields match interactive/16_labeling_tool.html's export exactly")


def read_preferences(path: str) -> tuple[list[PreferencePair], list[dict]]:
    """JSONL -> PreferencePair list (ties are skipped: they carry no ranking)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    out = []
    for r in rows:
        if r["label"] not in ("A", "B"):
            continue
        chosen, rejected = (r["response_a"], r["response_b"]) if r["label"] == "A" else (r["response_b"], r["response_a"])
        out.append(PreferencePair([{"role": "user", "content": r["prompt"]}], chosen, rejected,
                                  meta={"id": r["id"], "annotator": r["annotator"], "rubric": r["rubric"]}))
    return out, rows


back, rows1 = read_preferences(path)
back2, rows2 = read_preferences(path2)
same = sum(b.chosen == it["pair"].chosen and b.rejected == it["pair"].rejected for b, it in zip(back, items))
print(f"read back {len(back)} pairs from the perfect annotator ({same} identical to the originals); "
      f"{len(back2)} from the noisy one ({len(rows2) - len(back2)} ties skipped)")
check(same == len(pairs), "the perfect annotator's file round-trips to the original PreferencePairs")
by_id = {r["id"]: r["label"] for r in rows2}
k_file, _, _ = cohens_kappa([r["label"] for r in rows1], [by_id[r["id"]] for r in rows1])
check(abs(k_file - kappa) < 1e-9, f"kappa recomputed from the two files = {k_file:.3f} (paste them into the interactive to confirm)")
mean_s = sum(r["seconds"] for r in rows1) / len(rows1)
print(f"mean seconds per item (simulated): {mean_s:.1f}  — in a real run, items far below the mean are the ones to audit")

# ------------------------------------------------------------ 7. on-policy pairs
section("7. on-policy pairs by rejection sampling (reward.make_preference_pairs_from_model)")
tok = get_tokenizer()
sft_path = run_path("sft_nano.pt" if args.quick else "sft_small.pt")
if os.path.exists(sft_path):
    model, label = TinyLM.load(sft_path), os.path.relpath(sft_path)
else:
    model, _ = get_base_model(quick=args.quick, verbose=False)
    label = "the BASE model (run labs/lab15_sft.py first to use the SFT checkpoint)"
n_rs = 8 if args.quick else 40
t0 = time.perf_counter()
op_pairs, stats = make_preference_pairs_from_model(model, tok, make_examples(n_rs, seed=99, tasks=TASKS),
                                                   n_samples=4, max_new_tokens=12, temperature=1.0, seed=args.seed)
print(f"sampling 4 answers x {n_rs} prompts from {label}: {time.perf_counter() - t0:.1f}s")
print(f"stats: {stats}")
for p in op_pairs[:3]:
    print(f"  {p.prompt_messages[-1]['content']!r}: chosen {p.chosen!r} | rejected {p.rejected!r}")
if os.path.exists(sft_path):
    check(stats["n_pairs"] > 0, "an SFT model that is sometimes right yields (right, wrong) pairs from its own samples")
else:
    check(stats["n_pairs"] == 0 or stats["sample_accuracy"] < 0.5,
          "a base model is almost never right, so rejection sampling yields (almost) no pairs: on-policy data needs a policy that sometimes succeeds")
print(f"prompts with all-correct samples: {stats['n_all_correct']}, all-wrong: {stats['n_all_wrong']} — neither kind produces a pair")

# ------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(flip_rates, kappas, marker="o", color="#7c3aed")
axes[0].axhline(0.6, ls=":", color="#64748b"); axes[0].text(flip_rates[-1], 0.62, "0.6 'substantial'", ha="right", fontsize=8, color="#64748b")
axes[0].set_xlabel("noisy annotator's flip rate"); axes[0].set_ylabel("Cohen's kappa vs perfect annotator")
axes[0].set_title("agreement falls faster than the flip rate rises")
names = list(judges)
xs = range(3)
axes[1].bar([x - 0.2 for x in xs], [bias_rate[n] for n in names], 0.4, color="#dc2626", label="fails the swap test (position bias)")
axes[1].bar([x + 0.2 for x in xs], [1 - agree_verbose[n] for n in names], 0.4, color="#f59e0b", label="wrong on long-but-wrong pairs (verbosity bias)")
axes[1].set_xticks(list(xs)); axes[1].set_xticklabels(["verifier judge", "first wins", "longer wins"], fontsize=9)
axes[1].set_ylim(0, 1.15); axes[1].set_ylabel("fraction of pairs"); axes[1].set_title("two bias checks per judge"); axes[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
savefig(fig, "lab16_labeling.png")
done()
