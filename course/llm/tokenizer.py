"""Byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch.

Chapter 2 walks through this file line by line. The design follows GPT-2 / tiktoken:

1. Text is split into *pre-tokens* with a regex (so merges never cross a space or
   punctuation boundary in unhelpful ways).
2. Each pre-token becomes a list of bytes (0..255). The base vocabulary is therefore
   256 entries and *every* string is representable — there is no "unknown" token.
3. Training repeatedly finds the most frequent adjacent pair of symbols and merges it
   into a new symbol, until the vocabulary reaches ``vocab_size``.
4. Encoding applies the learned merges in the order they were learned.

Special tokens (e.g. ``<|user|>``) are matched *before* the regex split so they are
never broken into pieces.

The implementation favours clarity over speed; training a 4k vocabulary on ~1 MB of
text takes about a minute on a laptop.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Iterable, Optional

try:  # ``regex`` supports \p{L} classes; fall back to a simpler pattern with ``re``
    import regex as _re
    # Simplified GPT-4-style pattern: contractions, words with optional leading space,
    # numbers (up to 3 digits, so "2024" -> "202","4" like GPT-4), punctuation, whitespace.
    PRETOKEN_PATTERN = _re.compile(
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )
except ImportError:  # pragma: no cover
    import re as _re
    PRETOKEN_PATTERN = _re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+")


def pretokenize(text: str) -> list[str]:
    """Split text into chunks that BPE merges will never cross."""
    return PRETOKEN_PATTERN.findall(text)


class BPETokenizer:
    """A byte-level BPE tokenizer.

    Attributes
    ----------
    merges : dict[tuple[int, int], int]
        Learned merges, pair -> new token id, in learning order (dict preserves order).
    vocab : dict[int, bytes]
        Token id -> the bytes it stands for.
    special_tokens : dict[str, int]
        Special token string -> id.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._special_pattern = None
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ props
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def inverse_special(self) -> dict[int, str]:
        return {v: k for k, v in self.special_tokens.items()}

    # --------------------------------------------------------------- training
    @staticmethod
    def _count_pairs(word_counts: dict[tuple[int, ...], int]) -> Counter:
        pairs: Counter = Counter()
        for word, cnt in word_counts.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += cnt
        return pairs

    @staticmethod
    def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def train(self, text: str, vocab_size: int, special_tokens: Iterable[str] = (),
              verbose: bool = False) -> "BPETokenizer":
        """Learn merges from ``text`` until the vocabulary has ``vocab_size`` entries
        (special tokens included)."""
        specials = list(special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        assert n_merges >= 0, "vocab_size must be at least 256 + number of special tokens"

        # 1. pre-tokenize and count unique chunks (huge speed-up: merges act on the
        #    *types*, weighted by frequency, rather than on every token occurrence)
        chunks = Counter(pretokenize(text))
        word_counts: dict[tuple[int, ...], int] = {
            tuple(chunk.encode("utf-8")): cnt for chunk, cnt in chunks.items()
        }

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i in range(n_merges):
            pairs = self._count_pairs(word_counts)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            word_counts = {self._merge_word(w, best, new_id): c for w, c in word_counts.items()}
            if verbose and (i < 10 or i % 200 == 0):
                print(f"merge {i:5d}: {best} -> {new_id} {self.vocab[new_id]!r} (count {pairs[best]})")

        self.special_tokens = {}
        for s in specials:
            self.add_special_token(s)
        self._cache.clear()
        return self

    def add_special_token(self, s: str) -> int:
        if s in self.special_tokens:
            return self.special_tokens[s]
        tid = 256 + len(self.merges) + len(self.special_tokens)
        self.special_tokens[s] = tid
        self._special_pattern = None
        return tid

    # --------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        # Apply merges by priority: repeatedly merge the earliest-learned pair present.
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = list(self._merge_word(tuple(ids), best, self.merges[best]))
        self._cache[chunk] = ids
        return ids

    def _split_specials(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        if self._special_pattern is None:
            import re
            alts = "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            self._special_pattern = re.compile(f"({alts})")
        return [p for p in self._special_pattern.split(text) if p]

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Text -> list of token ids. Special tokens in ``text`` are recognised when
        ``allowed_special`` is True (turn it off for untrusted user text)."""
        ids: list[int] = []
        parts = self._split_specials(text) if allowed_special else [text]
        for part in parts:
            if allowed_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                for chunk in pretokenize(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        inv = self.inverse_special
        buf = bytearray()
        out: list[str] = []
        for i in ids:
            if i in inv:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                out.append(inv[i])
            else:
                buf += self.vocab[i]
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def token_str(self, i: int) -> str:
        """Human-readable form of one token (for printing)."""
        inv = self.inverse_special
        if i in inv:
            return inv[i]
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        data = {
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        for a, b, new in data["merges"]:
            tok.merges[(a, b)] = new
            tok.vocab[new] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = dict(data["special_tokens"])
        return tok

    # ------------------------------------------------------------- utilities
    def compression_ratio(self, text: str) -> float:
        """Bytes per token: higher means each token covers more text."""
        n = len(self.encode(text, allowed_special=False))
        return len(text.encode("utf-8")) / max(n, 1)


# The special tokens TinyLM uses for chat (Chapter 14). Declared here so the
# tokenizer and the chat template agree.
CHAT_SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|system|>", "<|user|>",
                       "<|assistant|>", "<|end|>", "<|tool_call|>", "<|tool_result|>"]
