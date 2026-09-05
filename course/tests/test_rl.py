"""Tests for llm/rl.py and llm/distill.py (Chapters 18–21). Runs on CPU in well under 90 s."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The nano model's matmuls are too small to benefit from several threads, and on a busy
# machine the extra threads make PyTorch *much* slower (spin-waiting), so use one.
torch.set_num_threads(1)

from llm import chat, tasks
from llm.config import preset
from llm.data import make_corpus, corpus_text
from llm.model import TinyLM
from llm.tokenizer import BPETokenizer, CHAT_SPECIAL_TOKENS
from llm import rl, distill


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(corpus_text(make_corpus(300)), 400, CHAT_SPECIAL_TOKENS)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return TinyLM(preset("nano", vocab_size=tok.vocab_size)).eval()


@pytest.fixture
def cfg():
    return rl.GRPOConfig(group_size=4, prompts_per_step=2, max_new_tokens=8, steps=1, lr=1e-4)


# ------------------------------------------------------------ building blocks
def test_grpo_advantages():
    r = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    adv = rl.grpo_advantages(r)
    assert abs(adv.mean().item()) < 1e-6
    assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-3
    assert adv[0] > 0 > adv[1]
    dr = rl.grpo_advantages(r, normalize_std=False)                 # Dr. GRPO
    assert torch.allclose(dr, r - r.mean())
    assert torch.all(rl.grpo_advantages(torch.ones(4)) == 0)         # all-equal group: no signal


def test_kl_estimators_zero_when_equal():
    lp = torch.randn(3, 5)
    ks = rl.kl_estimators(lp, lp.clone())
    for k in ("k1", "k2", "k3"):
        assert torch.allclose(ks[k], torch.zeros_like(lp), atol=1e-6)
    ks = rl.kl_estimators(lp, lp + 0.3)
    assert torch.all(ks["k3"] >= 0) and torch.all(ks["k2"] >= 0)   # k2, k3 non-negative


def test_ppo_clip_behaviour():
    old = torch.zeros(2, 4)
    mask = torch.ones(2, 4)
    adv = torch.tensor([1.0, -1.0])
    # ratio = 1 everywhere -> nothing clipped, loss = -mean(adv)
    logp = torch.zeros(2, 4, requires_grad=True)
    loss, st = rl.ppo_clip_loss(logp, old, adv, mask, 0.2, 0.2)
    assert st["clip_frac"] == 0.0 and abs(st["approx_kl"]) < 1e-6
    assert abs(loss.item() - 0.0) < 1e-6
    # ratio far above 1 for a positive advantage -> clipped, zero gradient on that row
    logp = torch.tensor([[1.0] * 4, [0.0] * 4], requires_grad=True)
    loss, st = rl.ppo_clip_loss(logp, old, adv, mask, 0.2, 0.28)
    loss.backward()
    assert st["clip_frac"] == 0.5
    assert torch.all(logp.grad[0] == 0)                              # clipped row: no gradient
    assert torch.all(logp.grad[1] != 0)                              # unclipped row: gradient
    # ratio far above 1 for a negative advantage -> NOT clipped (min picks the unclipped term)
    logp = torch.tensor([[0.0] * 4, [1.0] * 4], requires_grad=True)
    _, st = rl.ppo_clip_loss(logp, old, adv, mask, 0.2, 0.28)
    assert st["clip_frac"] == 0.0
    # sequence-level path (GSPO)
    s = rl.gspo_ratio(torch.full((2, 4), 0.5), old, mask)
    assert torch.allclose(s, torch.full((2,), math.exp(0.5)))
    g_loss, g_st = rl.gspo_loss(torch.full((2, 4), 0.5), old, adv, mask)
    assert g_st["clip_frac"] == 0.5


def test_reinforce_loss():
    logps = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
    mask = torch.ones(2, 2)
    loss = rl.reinforce_loss(logps, mask, torch.tensor([1.0, 0.0]))
    assert abs(loss.item() - 1.0) < 1e-6                             # -(1*-2 + 0*-4)/2
    loss_b = rl.reinforce_loss(logps, mask, torch.tensor([1.0, 0.0]), baseline=0.5)
    assert abs(loss_b.item() - (-(0.5 * -2 + -0.5 * -4) / 2)) < 1e-6


def test_response_mask_hand_built():
    PAD, END = 9, 8
    #         prompt (3)  | gen: a b END pad pad
    ids = torch.tensor([[1, 2, 3, 4, 5, END, PAD, PAD],
                        [1, 2, 3, 4, 5, 6, 7, 5]])                    # no END: everything counts
    m = rl.response_mask(ids, prompt_len=3, pad_id=PAD, end_id=END)
    assert m.shape == (2, 7)
    assert m[0].tolist() == [0, 0, 1, 1, 1, 0, 0]                    # targets 3,4,5(END) on
    assert m[1].tolist() == [0, 0, 1, 1, 1, 1, 1]


def test_token_logprobs_matches_manual(model):
    ids = torch.randint(0, model.cfg.vocab_size, (2, 10))
    lp = rl.token_logprobs(model, ids)
    with torch.no_grad():
        logits, _ = model(ids[:, :-1])
        manual = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    assert lp.shape == (2, 9)
    assert torch.allclose(lp, manual, atol=1e-5)
    assert lp.requires_grad


# --------------------------------------------------------------- rollouts
def test_rollout_group_shapes(model, tok, cfg):
    ex = tasks.make_examples(1, seed=1, tasks=["add"], max_value=20)[0]
    ro = rl.rollout_group(model, tok, ex, cfg)
    P = ro.prompt_len
    assert ro.ids.shape[0] == cfg.group_size and ro.ids.shape[1] <= P + cfg.max_new_tokens
    assert ro.mask.shape == (cfg.group_size, ro.ids.shape[1] - 1)
    assert torch.all(ro.mask[:, :P - 1] == 0)                        # nothing in the prompt
    assert len(ro.completions) == cfg.group_size
    assert torch.all(ro.rewards >= 0) and torch.all(ro.rewards <= 1.1)
    # a custom reward function is used
    ro2 = rl.rollout_group(model, tok, ex, cfg, reward_fn=lambda e, t, i: float(len(i)))
    n_end = (ro2.ids[:, P:] == tok.special_tokens["<|end|>"]).sum(-1)   # the mask also covers <|end|>
    assert torch.allclose(ro2.rewards, ro2.mask.sum(-1) - n_end)


def test_grpo_step_runs(model, tok, cfg):
    exs = tasks.make_examples(2, seed=2, tasks=["add"], max_value=20)
    opt = rl.make_optimizer(model, cfg)
    cfg.dynamic_sampling = False                                     # untrained model: rewards mostly 0
    before = [p.detach().clone() for p in model.parameters()]
    st = rl.grpo_step(model, None, tok, exs, cfg, opt, reward_fn=lambda e, t, i: float(sum(i) % 2))
    for k in ("reward", "accuracy", "resp_len", "clip_frac", "approx_kl", "entropy", "skipped_frac", "loss"):
        assert k in st
    assert st["clip_frac"] == 0.0                                    # ppo_epochs=1: ratio == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    # with a reference model and KL, and 2 PPO epochs, the clip path is exercised
    ref = TinyLM(model.cfg).eval(); ref.load_state_dict(model.state_dict())
    cfg.kl_coef = 0.05; cfg.ppo_epochs = 2
    st = rl.grpo_step(model, ref, tok, exs, cfg, opt, reward_fn=lambda e, t, i: float(sum(i) % 2))
    assert "kl_ref" in st and st["kl_ref"] >= 0
    hist = rl.grpo_train(model, tok, exs, cfg, verbose=False)
    assert len(hist) == cfg.steps and "step" in hist[0]


# -------------------------------------------------------------- multi-turn
def test_multi_turn_rollout_calculator(model, tok, cfg):
    env = rl.CalculatorEnv(3, 4)
    traj = rl.multi_turn_rollout(model, tok, env, cfg, max_turns=2)
    assert traj.ids.dim() == 1 and traj.mask.shape[0] == traj.ids.shape[0] - 1
    assert traj.mask.sum() > 0                                       # some generated tokens
    n_assist = sum(t.role == "assistant" for t in traj.turns)
    assert 1 <= n_assist <= 2
    assert traj.done or n_assist == 2
    # the prompt is never trained on
    P = len(traj.turns[0].ids)
    assert torch.all(traj.mask[:P] == 0)
    # the environment itself: a correct tool call then a correct answer
    env.reset()
    obs, r, done = env.step('<|tool_call|>{"name": "calc", "arguments": {"expression": "3 + 4"}}<|end|>')
    assert obs == "7" and r == 0 and not done
    obs, r, done = env.step("3 + 4 = 7<|end|>")
    assert obs is None and abs(r - 1.2) < 1e-9 and done
    env.reset()
    assert env.step("3 + 4 = 8")[1] == 0.0
    st = rl.multi_turn_grpo_step(model, tok, rl.CalculatorEnv.from_seed, cfg, rl.make_optimizer(model, cfg),
                                 n_tasks=1, max_turns=2)
    assert "reward" in st and "tool_call_rate" in st


# ------------------------------------------------------------- distillation
def test_kd_logit_loss_zero_when_identical():
    logits = torch.randn(2, 5, 11)
    mask = torch.ones(2, 5)
    assert abs(distill.kd_logit_loss(logits, logits.clone(), mask).item()) < 1e-6
    assert distill.kd_logit_loss(logits, logits + torch.randn_like(logits), mask).item() > 0
    assert distill.kd_logit_loss(logits, logits + 1.0, mask, temperature=2.0).item() < 1e-6  # shift-invariant


def test_on_policy_distill_step_runs(model, tok):
    teacher = TinyLM(model.cfg).eval(); teacher.load_state_dict(model.state_dict())
    student = TinyLM(model.cfg).eval()
    exs = tasks.make_examples(2, seed=3, tasks=["add"], max_value=20)
    ocfg = distill.OPDConfig(steps=1, group_size=2, prompts_per_step=2, max_new_tokens=6, lr=1e-4)
    opt = torch.optim.AdamW(student.parameters(), lr=ocfg.lr)
    st = distill.on_policy_distill_step(student, teacher, tok, exs, ocfg, opt)
    for k in ("loss", "reverse_kl", "accuracy", "resp_len"):
        assert k in st and math.isfinite(st[k])
    out = distill.offline_distill(student, teacher, tok, exs, n_samples=2, sft_steps_n=1,
                                  max_new_tokens=6, verbose=False)
    assert out["n_total"] == 4 and 0 <= out["keep_rate"] <= 1
