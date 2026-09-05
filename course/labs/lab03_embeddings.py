"""Lab 3: embeddings — words as vectors.

What you will see:
  1. nn.Embedding is a lookup table, and a lookup is a one-hot matrix multiply.
  2. Dot product and cosine similarity on hand-made vectors.
  3. The embedding matrix of a *trained* TinyLM: nearest neighbours by cosine.
  4. Group structure: colours cluster with colours, names with names.
  5. A 2-D picture of the 96/192-dimensional space (PCA via numpy SVD).
  6. Tied embeddings: the output layer reuses the same matrix.

Run:  python3 labs/lab03_embeddings.py            (--quick: nano base model, ~1 min)
      python3 labs/lab03_embeddings.py --full     (small base model, ~5-10 min if it must train)
"""
from _common import setup, check, banner, section, savefig, done, plt

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from llm.config import preset
from llm.data import COLORS, NAMES, OBJECTS, ANIMALS
from llm.model import TinyLM
from llm.pipeline import get_base_model, get_tokenizer, BASE_FULL, BASE_QUICK, COURSE_DIR

args = setup("Lab 3: embeddings — words as vectors")
plt = plt()
torch.manual_seed(args.seed)


def wait_for_checkpoint(path: str, every: int = 30, max_wait: int = 600) -> bool:
    """Another lab (or a teammate) may be pretraining the base model right now.
    Poll for the file before training our own copy."""
    if os.path.exists(path):
        return True
    rel = os.path.relpath(path, COURSE_DIR)
    print(f"[lab03] {rel} not found; polling every {every}s for up to {max_wait // 60} min "
          f"before training one here")
    waited = 0
    while waited < max_wait and not os.path.exists(path):
        time.sleep(every)
        waited += every
        print(f"   ... waited {waited}s; runs/ contains {sorted(os.listdir(os.path.dirname(path)))}")
    if os.path.exists(path):
        time.sleep(3)          # let the writer finish torch.save
        return True
    return False


# ----------------------------------------------------------------------------
section("1. nn.Embedding is a lookup table — and a lookup is a one-hot matmul")
tok = get_tokenizer()
V, d = tok.vocab_size, 192
emb = torch.nn.Embedding(V, d)                       # weight E: (V, d), random for now
ids = torch.tensor(tok.encode(" red kite"))          # (2,) two token ids
print("tokens:", [tok.token_str(i) for i in ids.tolist()], "ids:", ids.tolist())

onehot = F.one_hot(ids, V).float()                   # (2, V): a single 1 per row
via_matmul = onehot @ emb.weight                     # (2, V) @ (V, d) -> (2, d)
via_lookup = emb(ids)                                # (2, d): PyTorch's fast path
print(f"one-hot shape {tuple(onehot.shape)}, ones per row {onehot.sum(1).tolist()}")
print(f"max |onehot @ E - E[ids]| = {(via_matmul - via_lookup).abs().max().item():.1e}")
check(torch.equal(via_matmul, via_lookup), "one-hot @ E == E[ids]: an embedding is a row lookup")
print(f"a one-hot vector stores {V} numbers per token (all but one zero); "
      f"the embedding stores {d}, all of them informative")

# ----------------------------------------------------------------------------
section("2. Dot product and cosine similarity on 2-D vectors")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()))


a = torch.tensor([3.0, 4.0])      # length 5
b = torch.tensor([6.0, 8.0])      # same direction, twice as long
c = torch.tensor([4.0, -3.0])     # at right angles to a
for name, v in [("b (same direction, 2x longer)", b), ("c (perpendicular)", c), ("-a (opposite)", -a)]:
    print(f"a·{name:<32} dot = {float(a @ v):6.1f}   cosine = {cosine(a, v):5.2f}")
check(abs(cosine(a, b) - 1.0) < 1e-6 and abs(cosine(a, c)) < 1e-6 and abs(cosine(a, -a) + 1) < 1e-6,
      "cosine is +1 for same direction, 0 for perpendicular, -1 for opposite (length ignored)")

# ----------------------------------------------------------------------------
section("3. The embedding matrix of a trained TinyLM")
path = BASE_QUICK if args.quick else BASE_FULL
wait_for_checkpoint(path)
model, tok = get_base_model(quick=args.quick)
E = model.embed.weight.detach()                      # (V, d) — these are learned parameters
V, d = E.shape
print(f"model: {model.cfg.n_layers} layers, d_model={d}, vocab={V}; "
      f"embedding matrix {tuple(E.shape)} = {E.numel():,} parameters "
      f"({E.numel() / model.num_params(False):.0%} of the model)")
print(f"E[' red'][:6] = {[round(x, 3) for x in E[tok.encode(' red')[0]][:6].tolist()]} ...")
print(f"row norms: mean {E.norm(dim=1).mean():.3f}, min {E.norm(dim=1).min():.3f}, max {E.norm(dim=1).max():.3f}")


def tid(s: str) -> int:
    ids = tok.encode(s)
    assert len(ids) == 1, f"{s!r} is {len(ids)} tokens: {[tok.token_str(i) for i in ids]}"
    return ids[0]


# Only compare against tokens that look like text (skip control bytes and special tokens).
n_special = len(tok.special_tokens)
candidates = torch.tensor([i for i in range(V - n_special)
                           if tok.vocab[i].decode("utf-8", errors="replace").isprintable()])
En = F.normalize(E, dim=1)                           # unit-length rows -> dot = cosine


def neighbours(word: str, k: int = 5, matrix: torch.Tensor = En) -> list[tuple[str, float]]:
    i = tid(word)
    sims = matrix[candidates] @ matrix[i]            # (n_candidates,) cosine to `word`
    sims[candidates == i] = -2.0                     # exclude the word itself
    top = sims.topk(k)
    return [(tok.token_str(int(candidates[j])), float(s)) for s, j in zip(top.values, top.indices)]


print("\nnearest neighbours (cosine) in the TRAINED embedding matrix:")
probe = [" red", " kite", " Mia", "7", " three"]
hits = {}
for w in probe:
    nb = neighbours(w)
    print(f"  {w!r:10} -> " + ", ".join(f"{t!r} {s:.2f}" for t, s in nb))
    hits[w] = nb
print("(note: ' 7' is two tokens in this tokenizer — ' ' and '7' — so we probe the digit '7')")

colour_hits = sum(t.strip() in COLORS for t, _ in hits[" red"])
name_hits = sum(t.strip() in NAMES for t, _ in hits[" Mia"])
digit_hits = sum(t.strip().isdigit() for t, _ in hits["7"])
print(f"' red' neighbours that are colours: {colour_hits}/5 | ' Mia' -> names: {name_hits}/5 | '7' -> digits: {digit_hits}/5")
need = 3 if args.full else 1
check(colour_hits >= need and name_hits >= need, f"at least {need}/5 neighbours share the word's category")

untrained = TinyLM(model.cfg)                        # same shapes, random weights
Eu = F.normalize(untrained.embed.weight.detach(), dim=1)
print("\nfor contrast, neighbours of ' red' in an UNTRAINED model:")
print("  " + ", ".join(f"{t!r} {s:.2f}" for t, s in neighbours(" red", matrix=Eu)))

# ----------------------------------------------------------------------------
section("4. Group structure: within-group vs across-group cosine")


def single_tokens(words: list[str]) -> list[str]:
    return [w for w in words if len(tok.encode(w)) == 1]


groups = {
    "colour": single_tokens([" " + c for c in COLORS]),
    "name": single_tokens([" " + n for n in NAMES]),
    "object": single_tokens([" " + o for o in OBJECTS]),
    "animal": single_tokens([" " + a for a in ANIMALS]),
    "number": single_tokens([" one", " two", " three"] + list("0123456789")),
}
for g, ws in groups.items():
    print(f"  {g:<7}: {len(ws)} single-token words")


def mean_cos(ws1: list[str], ws2: list[str]) -> float:
    A = En[[tid(w) for w in ws1]]
    B = En[[tid(w) for w in ws2]]
    S = A @ B.T
    if ws1 is ws2:                                   # ignore the diagonal (word with itself)
        S = S[~torch.eye(len(ws1), dtype=torch.bool)]
    return float(S.mean())


names = list(groups)
table = np.array([[mean_cos(groups[g1], groups[g2]) for g2 in names] for g1 in names])
print("\nmean cosine between groups (rows/cols: " + ", ".join(names) + "):")
for g, row in zip(names, table):
    print(f"  {g:<7}" + "".join(f"{v:7.2f}" for v in row))
within = np.diag(table).mean()
across = table[~np.eye(len(names), dtype=bool)].mean()
print(f"mean within-group cosine {within:.3f} vs across-group {across:.3f}")
check(within > across, "tokens are closer to their own category than to others")

fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(table, cmap="viridis")
ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
ax.set_title("mean cosine between token groups")
fig.colorbar(im, ax=ax, fraction=0.046)
savefig(fig, "lab03_group_similarity.png")

# ----------------------------------------------------------------------------
section("5. A 2-D picture: PCA with numpy SVD (no sklearn)")
plot_groups = ["colour", "name", "number", "animal"]
words = [w for g in plot_groups for w in groups[g]]
labels = [g for g in plot_groups for _ in groups[g]]
X = E[[tid(w) for w in words]].numpy()               # (n_words, d)
Xc = X - X.mean(axis=0)                              # centre: PCA looks at spread around the mean
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)    # Xc = U S Vt ; rows of Vt are the principal axes
P = Xc @ Vt[:2].T                                    # (n_words, 2) coordinates on the top-2 axes
explained = S ** 2 / (S ** 2).sum()
print(f"top-2 principal axes explain {explained[0]:.0%} + {explained[1]:.0%} = {explained[:2].sum():.0%} of the variance"
      f" in {d} dimensions")
check(abs(float(explained.sum()) - 1.0) < 1e-4, "explained-variance fractions sum to 1")

palette = {"colour": "#dc2626", "name": "#2563eb", "number": "#16a34a", "animal": "#f59e0b"}
fig, ax = plt.subplots(figsize=(7, 5.5))
for g in plot_groups:
    pts = P[[i for i, l in enumerate(labels) if l == g]]
    ax.scatter(pts[:, 0], pts[:, 1], s=28, color=palette[g], label=g)
for (x, y), w in zip(P, words):
    ax.annotate(w.strip(), (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)")
ax.set_title(f"TinyLM {'nano' if args.quick else 'small'} token embeddings, projected to 2-D")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "lab03_pca.png")

# ----------------------------------------------------------------------------
section("6. Tied embeddings: the output layer is the same matrix")
check(model.lm_head.weight is model.embed.weight, "model.lm_head.weight IS model.embed.weight (one tensor, two jobs)")
ids = torch.tensor([tok.encode("Mia had a")])                      # (1, T)
with torch.no_grad():
    logits, _, h = model(ids, return_hidden=True)                # logits (1, T, V); h (1, T, d) after final norm
manual = h[0, -1] @ E.T                                          # (d,) @ (d, V) -> (V,)
print(f"logits shape {tuple(logits.shape)}; max |logits - h·Eᵀ| = {(logits[0, -1] - manual).abs().max().item():.1e}")
check(torch.allclose(logits[0, -1], manual, atol=1e-4), "next-token logits = hidden state · embedding rows")
probs = F.softmax(logits[0, -1], dim=-1)
top = probs.topk(5)
print("after 'Mia had a', top next tokens:", ", ".join(f"{tok.token_str(i)!r} {p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))
untied = TinyLM(preset("nano" if args.quick else "small", vocab_size=V, tie_embeddings=False))
print(f"parameters: tied {model.num_params(False):,} vs untied {untied.num_params(False):,} "
      f"-> tying saves {untied.num_params(False) - model.num_params(False):,} (= V x d = {V} x {d})")
check(untied.num_params(False) - model.num_params(False) == V * d, "tying saves exactly V x d parameters")

done()
