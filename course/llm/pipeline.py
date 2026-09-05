"""Shared helpers so every lab can get "the" corpus, tokenizer and base model.

Artifacts live in ``course/runs/`` (git-ignored). A lab that needs a pretrained base
model calls ``get_base_model()``; if no checkpoint exists it trains one (quick mode:
~1 min, full mode: ~5 min on a laptop CPU).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .config import preset
from .data import (make_corpus, corpus_text, tokenize_and_pack, split_train_val,
                   download_tinyshakespeare)
from .model import TinyLM
from .tokenizer import BPETokenizer, CHAT_SPECIAL_TOKENS
from .train import TrainConfig, train

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(COURSE_DIR, "runs")
FIG_DIR = os.path.join(COURSE_DIR, "figures", "generated")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TOKENIZER_PATH = os.path.join(RUNS_DIR, "tokenizer.json")
BASE_QUICK = os.path.join(RUNS_DIR, "base_nano.pt")
BASE_FULL = os.path.join(RUNS_DIR, "base_small.pt")
VOCAB_SIZE = 1024


def run_path(name: str) -> str:
    return os.path.join(RUNS_DIR, name)


def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def get_corpus(n_docs: int = 6000, seed: int = 0) -> list[dict]:
    return make_corpus(n_docs, seed)


def get_tokenizer(docs: Optional[list[dict]] = None, force: bool = False) -> BPETokenizer:
    """The course tokenizer: byte-level BPE, 1024 vocab, trained on Storyland."""
    if os.path.exists(TOKENIZER_PATH) and not force:
        return BPETokenizer.load(TOKENIZER_PATH)
    docs = docs or get_corpus()
    tok = BPETokenizer().train(corpus_text(docs), VOCAB_SIZE, CHAT_SPECIAL_TOKENS)
    tok.save(TOKENIZER_PATH)
    return tok


def get_tokens(tok: Optional[BPETokenizer] = None, docs: Optional[list[dict]] = None):
    docs = docs or get_corpus()
    tok = tok or get_tokenizer(docs)
    return split_train_val(tokenize_and_pack(docs, tok))


def get_base_model(quick: bool = False, force: bool = False, verbose: bool = True) -> tuple[TinyLM, BPETokenizer]:
    """Return (base model, tokenizer), pretraining one if needed."""
    path = BASE_QUICK if quick else BASE_FULL
    tok = get_tokenizer()
    if os.path.exists(path) and not force:
        return TinyLM.load(path), tok
    docs = get_corpus()
    train_tokens, val_tokens = get_tokens(tok, docs)
    cfg = preset("nano" if quick else "small", vocab_size=tok.vocab_size)
    model = TinyLM(cfg)
    tc = TrainConfig(steps=150 if quick else 700, batch_size=32, seq_len=128, optimizer="muon",
                     lr=1e-3, warmup_steps=20 if quick else 50, schedule="wsd", eval_every=50 if quick else 100,
                     log_every=25)
    if verbose:
        print(f"[pipeline] pretraining {'nano' if quick else 'small'} base model "
              f"({model.num_params():,} params, {tc.steps} steps) -> {path}")
    train(model, train_tokens, val_tokens, tc, verbose=verbose)
    model.save(path, TOKENIZER_PATH, extra={"stage": "base"})
    return model, tok


def device_summary() -> str:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return f"torch {torch.__version__} | device {dev} | threads {torch.get_num_threads()}"
