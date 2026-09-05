"""Tests for Chapter 17: reward models (llm/reward.py) and DPO (llm/dpo.py).

Everything runs on CPU with the ``nano`` preset and a tiny fresh tokenizer, well
under a minute in total.
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import chat, tasks                                   # noqa: E402
from llm.config import preset                                 # noqa: E402
from llm.data import make_corpus, corpus_text                 # noqa: E402
from llm.model import TinyLM                                  # noqa: E402
from llm.tokenizer import BPETokenizer, CHAT_SPECIAL_TOKENS   # noqa: E402
from llm import reward as R                                   # noqa: E402
from llm import dpo as D                                      # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(corpus_text(make_corpus(300, seed=1)), 512, CHAT_SPECIAL_TOKENS)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    return TinyLM(preset("nano", vocab_size=tok.vocab_size))


@pytest.fixture(scope="module")
def pairs():
    examples = tasks.make_examples(64, seed=3, tasks=["add", "sub", "reverse", "upper", "count", "first"])
    return R.make_preference_pairs(examples, n_wrong_styles=1, seed=0)


# ------------------------------------------------------------ reward model
def test_bradley_terry_known_values():
    zero = torch.zeros(4)
    # equal scores -> P(chosen wins) = 0.5 -> loss = log 2
    assert math.isclose(R.bradley_terry_loss(zero, zero).item(), math.log(2), rel_tol=1e-6)
    # a huge margin -> loss ~ 0; a huge negative margin -> loss ~ margin
    assert R.bradley_terry_loss(torch.full((2,), 30.0), torch.zeros(2)).item() < 1e-6
    assert math.isclose(R.bradley_terry_loss(torch.zeros(2), torch.full((2,), 30.0)).item(), 30.0, rel_tol=1e-3)
    # hand computation for one pair: -log sigmoid(1.5 - 0.5)
    got = R.bradley_terry_loss(torch.tensor([1.5]), torch.tensor([0.5])).item()
    assert math.isclose(got, -math.log(1 / (1 + math.exp(-1.0))), rel_tol=1e-6)


def test_reward_model_forward_shape(model):
    rm = R.RewardModel(model)
    ids = torch.randint(0, model.cfg.vocab_size, (3, 12))
    lengths = torch.tensor([12, 5, 1])
    out = rm(ids, lengths)
    assert out.shape == (3,)
    # padding after the last real token must not change the reward (causal attention)
    ids2 = ids.clone()
    ids2[1, 5:] = 0
    assert torch.allclose(rm(ids2, lengths)[1], out[1], atol=1e-5)


def test_train_reward_model_learns(tok, model, pairs):
    rm = R.RewardModel(model)
    before = R.reward_accuracy(rm, tok, pairs)          # zero head -> every margin is 0
    assert before == 0.0
    cfg = R.RMConfig(steps=30, batch_size=8, lr=1e-3, warmup=3, log_every=10)
    hist = R.train_reward_model(rm, tok, pairs, cfg, verbose=False)
    assert isinstance(hist, R.PrefHistory) and len(hist.accuracy) == len(hist.step)
    after = R.reward_accuracy(rm, tok, pairs)
    assert after > 0.5, after


def test_make_preference_pairs_shapes(pairs):
    assert len(pairs) == 64
    for p in pairs:
        assert p.chosen != p.rejected
        assert p.prompt_messages[-1]["role"] == "user"
        assert p.meta["style"] in R.WRONG_STYLES


def test_encode_pair_and_response(tok, pairs):
    p = pairs[0]
    ids, mask = R.encode_response(tok, p.prompt_messages, p.chosen)
    prompt = tok.encode(chat.render(p.prompt_messages, add_generation_prompt=True))
    assert ids[:len(prompt)] == prompt
    assert mask[:len(prompt)] == [0] * len(prompt) and all(m == 1 for m in mask[len(prompt):])
    assert ids[-1] == tok.special_tokens["<|end|>"]
    c, r = R.encode_pair(tok, p)
    assert c[:len(prompt)] == r[:len(prompt)]


def test_verifiable_and_combined_reward():
    ex = tasks.make_examples(1, seed=0, tasks=["add"])[0]
    assert R.verifiable_reward(ex, ex.answer) == 1.0
    assert R.verifiable_reward(ex, "nope") == 0.0
    full = R.combined_reward(ex, ex.answer, [1, 2, 3], max_len=10)
    assert math.isclose(full, 1.1)
    truncated = R.combined_reward(ex, ex.answer, [1] * 10, max_len=10)
    assert math.isclose(truncated, 0.6)


def test_rubric_reward_returns_per_criterion():
    score, per = R.rubric_reward("23 + 45 = 68", R.ARITHMETIC_RUBRIC)
    assert set(per) == {name for name, _ in R.ARITHMETIC_RUBRIC}
    assert score == 1.0
    score2, per2 = R.rubric_reward("I think maybe\nit is 68", R.ARITHMETIC_RUBRIC)
    assert score2 < score and per2["has_equation"] == 0.0 and per2["single_line"] == 0.0


# -------------------------------------------------------------------- DPO
def test_sequence_logprob_matches_manual(model):
    ids = torch.tensor([[5, 7, 9, 11, 13]])
    mask = torch.tensor([[0., 0., 1., 1., 1.]])            # response = ids[2:]
    got = D.sequence_logprob(model, ids, mask)
    logits, _ = model(ids)
    logp = F.log_softmax(logits[0].float(), -1)
    # position t predicts ids[t+1]; response tokens are ids[2], ids[3], ids[4]
    manual = logp[1, 9] + logp[2, 11] + logp[3, 13]
    assert torch.allclose(got[0], manual, atol=1e-5)
    avg = D.sequence_logprob(model, ids, mask, average=True)
    assert torch.allclose(avg[0], manual / 3, atol=1e-5)


def test_dpo_loss_at_init():
    lp = torch.randn(6) * 5
    loss, stats = D.dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)     # policy == reference
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert stats["accuracy"] == 0.5 and stats["margin"] == 0.0
    # hand computation: beta * [(pc - rc) - (pr - rr)] = 0.1 * (2 - (-1)) = 0.3
    loss2, stats2 = D.dpo_loss(torch.tensor([2.0]), torch.tensor([-1.0]), torch.zeros(1), torch.zeros(1))
    assert math.isclose(loss2.item(), -math.log(torch.sigmoid(torch.tensor(0.3)).item()), rel_tol=1e-5)
    assert stats2["accuracy"] == 1.0 and math.isclose(stats2["margin"], 0.3, rel_tol=1e-5)


def test_variants_run():
    pc, pr, rc, rr = (torch.randn(5) - 3 for _ in range(4))
    for fn in (D.ipo_loss,):
        loss, stats = fn(pc, pr, rc, rr)
        assert torch.isfinite(loss) and set(stats) >= {"margin", "accuracy"}
    avg_c, avg_r = -torch.rand(5), -torch.rand(5)
    for loss, stats in (D.simpo_loss(avg_c, avg_r), D.orpo_loss(avg_c, avg_r)):
        assert torch.isfinite(loss) and "margin" in stats
    # a clearly preferred chosen answer gives a small loss for every variant
    assert D.simpo_loss(torch.zeros(1), torch.full((1,), -3.0))[0].item() < 0.1


def test_build_pair_batch(tok, pairs):
    c, cm, r, rm = D.build_pair_batch(tok, pairs[:4], tok.special_tokens["<|pad|>"])
    assert c.shape == r.shape == cm.shape == rm.shape and c.shape[0] == 4
    assert cm.sum() > 0 and rm.sum() > 0
    assert ((c == tok.special_tokens["<|pad|>"]).float() * cm).sum() == 0   # pads never masked in


def test_dpo_train_increases_margin(tok, model, pairs):
    cfg = D.DPOConfig(steps=20, batch_size=8, lr=1e-4, warmup=2, log_every=5)
    ref = D.make_reference(model)
    before = D.dpo_eval(model, ref, tok, pairs, cfg)
    assert math.isclose(before["margin"], 0.0, abs_tol=1e-6) and before["accuracy"] == 0.5
    hist = D.dpo_train(model, ref, tok, pairs, cfg, verbose=False)
    after = D.dpo_eval(model, ref, tok, pairs, cfg)
    assert after["margin"] > before["margin"] + 0.05, (before, after)
    assert after["accuracy"] > 0.5 and after["loss"] < before["loss"]
    assert len(hist.margin) == len(hist.step) > 0


@pytest.mark.parametrize("loss", ["simpo", "orpo", "ipo"])
def test_dpo_train_variants_run(tok, model, pairs, loss):
    cfg = D.DPOConfig(steps=3, batch_size=4, lr=1e-4, warmup=1, log_every=1, loss=loss)
    hist = D.dpo_train(model, None, tok, pairs[:8], cfg, verbose=False)
    assert all(math.isfinite(l) for l in hist.train_loss) and len(hist.step) == 3
