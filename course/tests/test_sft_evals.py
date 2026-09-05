"""Tests for llm/sft.py (Chapter 15) and llm/evals.py (Chapter 23). Whole file < 60 s on CPU."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.chat import END
from llm.config import preset
from llm.data import corpus_text, make_corpus
from llm.evals import (bootstrap_ci, contamination_check, eval_tasks, judge_pairwise,
                       position_bias_check, rule_based_judge)
from llm.model import TinyLM
from llm.sft import (LoRALinear, SFTConfig, apply_lora, build_sft_dataset, describe_mask,
                     make_sft_examples, merge_lora, respond, sft_train, trainable_params)
from llm.tasks import TaskExample
from llm.tokenizer import BPETokenizer, CHAT_SPECIAL_TOKENS


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer().train(corpus_text(make_corpus(300)), 512, CHAT_SPECIAL_TOKENS)


@pytest.fixture
def model(tok) -> TinyLM:
    torch.manual_seed(0)
    return TinyLM(preset("nano", vocab_size=tok.vocab_size))


@pytest.fixture(scope="module")
def examples():
    return make_sft_examples(64, seed=1, tasks=["upper", "add"])


# --------------------------------------------------------------- loss mask
def test_loss_mask_covers_only_assistant_tokens(tok):
    ex = TaskExample("upper", "Write in capitals: kite", "KITE", {"answer": "KITE"})
    (ids, mask), = build_sft_dataset(tok, [ex])
    assert len(ids) == len(mask)
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    untrained = tok.decode([i for i, m in zip(ids, mask) if not m])
    assert trained == "KITE" + END                    # answer + its closing tag
    assert "kite" in untrained and "<|user|>" in untrained and "<|assistant|>" in untrained
    assert END not in trained[:-len(END)]              # only the assistant's <|end|> is trained
    shown = describe_mask(tok, ids, mask)                  # trained tokens in [brackets]
    assert shown.endswith("[K][I][T][E][<|end|>]") and "[" not in shown.split("<|assistant|>")[0]


def test_max_len_truncates(tok, examples):
    for ids, mask in build_sft_dataset(tok, examples, max_len=10):
        assert len(ids) <= 10 and len(mask) == len(ids)


# --------------------------------------------------------------- training
def test_sft_reduces_loss(model, tok, examples):
    cfg = SFTConfig(steps=20, batch_size=8, lr=1e-3, warmup_steps=2, log_every=1)
    hist = sft_train(model, tok, examples, cfg, verbose=False)
    assert len(hist.train_loss) == 20
    assert hist.train_loss[-1] < hist.train_loss[0] * 0.9


def test_sft_with_val_examples_records_accuracy(model, tok, examples):
    cfg = SFTConfig(steps=4, batch_size=8, log_every=2, eval_every=2)
    hist = sft_train(model, tok, examples, cfg, val_examples=examples[:4], verbose=False)
    assert hist.val_step and len(hist.val_loss) == len(hist.val_acc)
    assert all(0.0 <= a <= 1.0 for a in hist.val_acc)


# -------------------------------------------------------------------- LoRA
def test_lora_apply_and_merge(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 12))
    with torch.no_grad():
        before, _ = model(x)
    n_full = trainable_params(model)
    names_before = set(model.state_dict())

    apply_lora(model, rank=4)
    loras = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(loras) == model.cfg.n_layers * 6           # q, v, o, gate, up, down per block
    assert 0 < trainable_params(model) < 0.1 * n_full
    assert all(n.endswith(("lora_A", "lora_B")) for n, p in model.named_parameters() if p.requires_grad)
    with torch.no_grad():
        after_apply, _ = model(x)
    assert torch.allclose(before, after_apply, atol=1e-6)  # B = 0 -> identical function

    with torch.no_grad():                                  # pretend we trained the adapters
        for m in loras:
            m.lora_B.normal_(std=0.05)
        with_adapters, _ = model(x)
    assert not torch.allclose(before, with_adapters, atol=1e-3)

    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert set(model.state_dict()) == names_before
    assert trainable_params(model) == n_full
    with torch.no_grad():
        merged, _ = model(x)
    assert torch.allclose(with_adapters, merged, atol=1e-5)


def test_sft_train_with_lora(model, tok, examples):
    base = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = SFTConfig(steps=5, batch_size=8, lr=1e-3, warmup_steps=1, lora_rank=2, log_every=1)
    hist = sft_train(model, tok, examples, cfg, verbose=False)
    assert len(hist.train_loss) == 5
    assert not any(isinstance(m, LoRALinear) for m in model.modules())   # merged back
    assert torch.equal(base["embed.weight"], model.state_dict()["embed.weight"])  # base frozen
    assert not torch.equal(base["blocks.0.attn.q_proj.weight"], model.state_dict()["blocks.0.attn.q_proj.weight"])


# ------------------------------------------------------------------- evals
def test_respond_returns_text(model, tok):
    out = respond(model, tok, "Write in capitals: kite", max_new_tokens=4)
    assert isinstance(out, str) and END not in out


def test_eval_tasks_and_table(model, tok, examples):
    res = eval_tasks(model, tok, examples[:6], max_new_tokens=4)
    assert 0.0 <= res.accuracy <= 1.0
    assert set(res.per_task) <= {"upper", "add"}
    assert len(res.samples) == 6 and len(res.correct) == 6
    table = res.table()
    assert "| task |" in table and "all" in table


def test_bootstrap_ci_bounds():
    scores = [1.0] * 30 + [0.0] * 20
    lo, hi = bootstrap_ci(scores, n_boot=500)
    assert 0.0 <= lo <= 0.6 <= hi <= 1.0
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_contamination_check_detects_overlap():
    leaked = TaskExample("story_qa", "Mia had a red kite. One sunny day Mia took the kite to the park. What color?",
                         "red", {"answer": "red"})
    clean = TaskExample("upper", "Write in capitals: kite", "KITE", {"answer": "KITE"})
    train_docs = [{"text": "Mia had a red kite. One sunny day Mia took the kite to the park."},
                  {"text": "Leo met Ava at the beach."}]
    assert contamination_check(train_docs, [leaked, clean]) == 0.5
    assert contamination_check(train_docs, [clean]) == 0.0
    assert contamination_check(train_docs, ["one sunny day mia took the kite to the park"]) == 1.0


def test_judge_and_position_bias():
    ex = TaskExample("add", "What is 2 + 3?", "2 + 3 = 5", {"answer": 5})
    assert judge_pairwise(ex, "2 + 3 = 5", "2 + 3 = 6") == "A"
    assert judge_pairwise(ex, "2 + 3 = 6", "2 + 3 = 5") == "B"
    assert judge_pairwise("free-form question", "", "an answer") == "B"
    assert judge_pairwise("q", "42", "the answer is 42") == "A"    # both numeric -> shorter
    assert judge_pairwise("q", "cat", "dog") == "tie"

    fair = position_bias_check(rule_based_judge, ex, "2 + 3 = 5", "2 + 3 = 6")
    assert fair["consistent"] and fair["forward"] == "A"
    always_first = lambda p, a, b: "A"
    biased = position_bias_check(always_first, ex, "2 + 3 = 5", "2 + 3 = 6")
    assert biased["position_bias"] and not biased["consistent"]
