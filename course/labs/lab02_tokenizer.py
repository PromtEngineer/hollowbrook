"""Lab 2: tokenization — train byte-level BPE on Storyland and look inside it.

    python3 labs/lab02_tokenizer.py            # --quick: vocab 512 and 1024 on Storyland, < 60 s
    python3 labs/lab02_tokenizer.py --full     # + vocab sweep to 4096, tiny Shakespeare comparison
"""
from _common import setup, check, banner, section, savefig, done, plt

import os
import time

from llm.tokenizer import BPETokenizer, pretokenize, CHAT_SPECIAL_TOKENS
from llm.pipeline import get_corpus, get_tokenizer
from llm.data import corpus_text, download_tinyshakespeare

args = setup("Lab 02: byte-level BPE tokenizer")

# ------------------------------------------------------------------ corpus
section("corpus")
docs = get_corpus()
text = corpus_text(docs)
n_bytes = len(text.encode("utf-8"))
print(f"{len(docs)} Storyland docs, {len(text):,} characters, {n_bytes:,} UTF-8 bytes")
print(f"distinct pre-tokens (regex chunks): {len(set(pretokenize(text))):,}")

# ------------------------------------------------------------ pre-tokenize
section("step 1: pre-tokenization (the regex decides where merges may NOT cross)")
demo = "Mia's kite flew in 2024! It cost $12.50 ... wow"
print(repr(demo))
print("->", pretokenize(demo))
check(pretokenize("kites kite") == ["kites", " kite"], "a leading space belongs to the word that follows it")

# ------------------------------------------------------------- train BPE
section("step 2: train BPE with vocab 512 and 1024 (merge = most frequent adjacent pair)")
toks: dict[int, BPETokenizer] = {}
train_time: dict[int, float] = {}
for vs in (512, 1024):
    t0 = time.perf_counter()
    toks[vs] = BPETokenizer().train(text, vs)
    train_time[vs] = time.perf_counter() - t0
    print(f"vocab {vs:>5}: learned {len(toks[vs].merges):>4} merges in {train_time[vs]:.2f}s "
          f"-> actual vocab {toks[vs].vocab_size}")
check(len(toks[512].merges) == 256, "vocab 512 = 256 bytes + 256 merges")
check(len(toks[1024].merges) < 768,
      f"vocab 1024 stopped early at {len(toks[1024].merges)} merges: Storyland ran out of pairs to merge")

section("the first 15 merges (byte pair -> new token)")
for i, ((a, b), new) in enumerate(list(toks[512].merges.items())[:15]):
    v = toks[512].vocab
    print(f"merge {i:2d}: {v[a]!r:>8} + {v[b]!r:<8} -> id {new} = {v[new]!r}")
first = list(toks[512].merges.items())[0]
check(toks[512].vocab[first[1]] == b" t", "the very first merge is ' ' + 't' -> ' t'")

# ------------------------------------------------------- compression ratio
section("step 3: compression ratio = bytes per token (higher = each token covers more text)")
sample = text[:200_000]
byte_ratio = 1.0
ratios = {vs: t.compression_ratio(sample) for vs, t in toks.items()}
print(f"raw bytes  : {byte_ratio:.2f} bytes/token  ({len(sample.encode()):,} tokens for the sample)")
for vs, r in ratios.items():
    print(f"vocab {vs:>5}: {r:.2f} bytes/token  ({len(toks[vs].encode(sample)):,} tokens)")
print(f"word-level : {len(sample.encode()) / max(1, len(sample.split())):.2f} bytes/token  ({len(sample.split()):,} whitespace words)")
check(ratios[1024] > ratios[512] > 1.0, "bigger vocab -> fewer tokens per byte")

# ------------------------------------------------------- round trip & viewing
section("step 4: encode / decode round trip on Storyland text")
s = docs[7]["text"]
ids = toks[1024].encode(s)
print(f"{s[:80]!r}...")
print(f"-> {len(s)} chars, {len(s.encode())} bytes, {len(ids)} tokens")
print("   " + "|".join(toks[1024].token_str(i) for i in ids[:20]) + "|...")
check(toks[1024].decode(ids) == s, "decode(encode(text)) == text")

# --------------------------------------------------------- tricky strings
section("step 5: tricky strings (tokens shown between |bars|)")
tricky = [
    ("a common word", "The kite was red."),
    ("a number", "In 2024 there were 12345 kites."),
    ("strawberry", "strawberry"),
    ("whitespace runs", "Mia    had\t\ta kite"),
    ("an emoji", "I love 🍓"),
    ("German", "Der Hund läuft schnell durch den Park."),
    ("code", "for i in range(10): print(i)"),
]
for name, s in tricky:
    ids = toks[1024].encode(s, allowed_special=False)
    parts = [toks[1024].token_str(i) for i in ids]
    print(f"{name:<16} {len(s.encode()):>3} bytes -> {len(ids):>2} tokens  |" + "|".join(parts) + "|")
    check(toks[1024].decode(ids) == s, f"round trip ok: {name}")
emoji_ids = toks[1024].encode("🍓", allowed_special=False)
check(len(emoji_ids) == 4 and all(i < 256 for i in emoji_ids), "🍓 is 4 raw bytes (never seen in Storyland)")
num_ids = toks[1024].encode("2024", allowed_special=False)
print(f"'2024' pre-tokenizes into {pretokenize('2024')} -> tokens {[toks[1024].token_str(i) for i in num_ids]}")
check(pretokenize("2024") == ["202", "4"], "digits are chunked in groups of at most 3 (GPT-4 rule)")
r_count = sum(toks[1024].token_str(i).count("r") for i in toks[1024].encode("strawberry"))
print(f"'strawberry' -> {[toks[1024].token_str(i) for i in toks[1024].encode('strawberry')]}: "
      f"a model sees these pieces, not letters, yet must count {r_count} r's")

# --------------------------------------------------------- special tokens
section("step 6: special tokens are protected (and untrusted text can be locked out)")
chat = BPETokenizer().train(text, 1024, CHAT_SPECIAL_TOKENS)
print(f"special tokens: {chat.special_tokens}")
msg = "<|user|>What is 2 + 2?<|assistant|>"
trusted = chat.encode(msg, allowed_special=True)
untrusted = chat.encode(msg, allowed_special=False)
print(f"allowed_special=True : {len(trusted):>2} tokens |" + "|".join(chat.token_str(i) for i in trusted) + "|")
print(f"allowed_special=False: {len(untrusted):>2} tokens |" + "|".join(chat.token_str(i) for i in untrusted) + "|")
check(chat.special_tokens["<|user|>"] in trusted, "with allowed_special=True '<|user|>' is ONE token")
check(chat.special_tokens["<|user|>"] not in untrusted, "with allowed_special=False it is spelled out as text")
check(chat.decode(trusted) == msg and chat.decode(untrusted) == msg, "both decode back to the same string")
course_tok = get_tokenizer()
check(course_tok.vocab_size == chat.vocab_size and course_tok.merges == chat.merges,
      f"the course tokenizer in runs/tokenizer.json is exactly this one (vocab {course_tok.vocab_size})")

# ------------------------------------------------------- vocab sweep (full)
sizes = [256, 512, 1024] if args.quick else [256, 300, 400, 512, 768, 1024, 2048, 4096]
d_model = 192                                         # the 'small' TinyLM width (Chapter 6)
section(f"vocabulary size trade-off (sweep {sizes})")
print(f"{'requested':>10}{'actual V':>10}{'merges':>8}{'bytes/tok':>11}{'tokens':>9}{'embed params (V×d)':>21}{'train s':>9}")
sweep_v, sweep_ratio = [], []
for vs in sizes:
    t0 = time.perf_counter()
    t = toks.get(vs) or BPETokenizer().train(text, vs)
    dt = train_time.get(vs, time.perf_counter() - t0)
    r = t.compression_ratio(sample)
    sweep_v.append(t.vocab_size)
    sweep_ratio.append(r)
    print(f"{vs:>10}{t.vocab_size:>10}{len(t.merges):>8}{r:>11.2f}{len(t.encode(sample)):>9,}"
          f"{t.vocab_size * d_model:>21,}{dt:>9.2f}")
check(sweep_ratio == sorted(sweep_ratio), "compression never gets worse as the vocabulary grows")

# ---------------------------------------------------- real text (full)
shakes = None
if not args.quick:
    section("real text: tiny Shakespeare vs Storyland")
    path = download_tinyshakespeare()
    if path is None:
        print("(no network and no cached file: skipping Shakespeare)")
    else:
        sh = open(path, encoding="utf-8").read()
        print(f"{len(sh):,} chars, {len(set(pretokenize(sh))):,} distinct pre-tokens (Storyland has {len(set(pretokenize(text))):,})")
        t0 = time.perf_counter()
        shakes = BPETokenizer().train(sh, 1024)
        print(f"trained vocab 1024 on Shakespeare in {time.perf_counter() - t0:.1f}s ({len(shakes.merges)} merges)")
        print("first 10 Shakespeare merges: " + ", ".join(repr(shakes.vocab[n]) for n in list(shakes.merges.values())[:10]))
        sh_sample = sh[:200_000]
        table = [
            ("Storyland tok  on Storyland", toks[1024].compression_ratio(sample)),
            ("Storyland tok  on Shakespeare", toks[1024].compression_ratio(sh_sample)),
            ("Shakespeare tok on Shakespeare", shakes.compression_ratio(sh_sample)),
            ("Shakespeare tok on Storyland", shakes.compression_ratio(sample)),
        ]
        for name, r in table:
            print(f"{name:<32}{r:6.2f} bytes/token")
        check(table[0][1] > table[1][1], "a tokenizer compresses the text it was trained on best")
        check(table[2][1] > table[1][1], "Shakespeare needs its own tokenizer to compress well")
        s = "To be, or not to be, that is the question:"
        for name, t in (("Storyland tok", toks[1024]), ("Shakespeare tok", shakes)):
            ids = t.encode(s)
            print(f"{name:<16}{len(ids):>3} tokens |" + "|".join(t.token_str(i) for i in ids) + "|")

# ------------------------------------------------------------- figure
p = plt()
fig, axes = p.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(sweep_v, sweep_ratio, marker="o", color="#2563eb")
axes[0].set_xscale("log")
axes[0].set_xlabel("vocabulary size V (actual)")
axes[0].set_ylabel("bytes per token on Storyland")
axes[0].set_title("bigger vocab = fewer tokens (diminishing returns)")
axes[1].plot(sweep_v, [v * d_model / 1e3 for v in sweep_v], marker="s", color="#f59e0b")
axes[1].set_xscale("log")
axes[1].set_xlabel("vocabulary size V")
axes[1].set_ylabel("embedding parameters (thousands), d=192")
axes[1].set_title("...but a bigger embedding matrix")
fig.tight_layout()
savefig(fig, "lab02_tokenizer.png")
done()
