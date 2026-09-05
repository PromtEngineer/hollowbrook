"""Tests for the core library: tokenizer, model, generation, optimizer, training, data, chat."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.config import TinyLMConfig, preset
from llm.data import (make_corpus, add_noise, curate, CurationReport, QualityClassifier, corpus_text,
                      tokenize_and_pack, split_train_val, minhash_signature, shingles, jaccard, gopher_reason,
                      scrub_pii, mix_sources, decontaminate)
from llm.model import TinyLM, RMSNorm, rope_tables, apply_rope, causal_mask
from llm.tokenizer import BPETokenizer, CHAT_SPECIAL_TOKENS, pretokenize
from llm.generate import sample_next, generate, generate_ids, sample_group
from llm.optim import newton_schulz_orthogonalize, Muon, build_optimizer, lr_at, split_params
from llm.train import TrainConfig, train, get_batch, estimate_loss, save_checkpoint, load_checkpoint
from llm.chat import render, build_sft_example, collate, parse_tool_call
from llm.microautograd import Value, train_xor
from llm.tasks import make_examples, verify, extract_answer


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(400, seed=0)


@pytest.fixture(scope="module")
def tok(corpus):
    return BPETokenizer().train(corpus_text(corpus), 512, CHAT_SPECIAL_TOKENS)


# ------------------------------------------------------------------ tokenizer
def test_pretokenize_keeps_leading_space():
    assert pretokenize("Mia had a kite") == ["Mia", " had", " a", " kite"]


def test_tokenizer_roundtrip(tok):
    for s in ["Mia had a red kite.", "What is 12 + 7?\nAnswer: 19", "émoji ✓ bytes", "  spaces   and\ttabs"]:
        assert tok.decode(tok.encode(s)) == s


def test_tokenizer_special_tokens(tok):
    ids = tok.encode("<|user|>hi<|end|>")
    assert ids[0] == tok.special_tokens["<|user|>"] and ids[-1] == tok.special_tokens["<|end|>"]
    # untrusted text must not be able to inject special tokens
    ids2 = tok.encode("<|user|>hi<|end|>", allowed_special=False)
    assert tok.special_tokens["<|user|>"] not in ids2
    assert tok.decode(ids) == "<|user|>hi<|end|>"


def test_tokenizer_save_load(tok, tmp_path):
    p = str(tmp_path / "tok.json")
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    s = "Leo took the boat to the lake."
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.vocab_size == tok.vocab_size


def test_tokenizer_compresses(tok, corpus):
    assert tok.compression_ratio(corpus_text(corpus[:50])) > 2.5   # bytes per token


# ---------------------------------------------------------------------- model
def test_config_presets():
    for name in ["nano", "small", "medium"]:
        cfg = preset(name)
        assert cfg.d_model % cfg.n_heads == 0
    assert preset("nano", vocab_size=300).vocab_size == 300


def test_rmsnorm_unit_rms():
    x = torch.randn(2, 5, 16) * 7
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 5), atol=1e-4)


def test_rope_preserves_norm_and_is_relative():
    cos, sin = rope_tables(8, 16)
    q = torch.randn(1, 1, 16, 8)
    r = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), r.norm(dim=-1), atol=1e-5)
    # dot product depends only on relative position: <rope(q,i), rope(k,j)> == <rope(q,i+d), rope(k,j+d)>
    k = torch.randn(1, 1, 16, 8)
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    d1 = (rq[0, 0, 2] * rk[0, 0, 5]).sum()
    q2 = q.clone(); q2[0, 0, 7] = q[0, 0, 2]
    k2 = k.clone(); k2[0, 0, 10] = k[0, 0, 5]
    rq2, rk2 = apply_rope(q2, cos, sin), apply_rope(k2, cos, sin)
    d2 = (rq2[0, 0, 7] * rk2[0, 0, 10]).sum()
    assert torch.allclose(d1, d2, atol=1e-4)


def test_causal_mask_and_window():
    m = causal_mask(4, 4, 0, None, "cpu")
    assert m.tolist() == [[True, False, False, False], [True, True, False, False],
                          [True, True, True, False], [True, True, True, True]]
    w = causal_mask(4, 4, 0, 2, "cpu")
    assert w[3].tolist() == [False, False, True, True]
    # with a cache of 3 past tokens and 1 new query
    c = causal_mask(1, 4, 3, None, "cpu")
    assert c.tolist() == [[True, True, True, True]]


def test_model_forward_shapes_and_loss(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    logits, loss = model(x, y)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5     # near uniform at init
    assert model.num_params() > 0 and model.flops_per_token() > 6 * model.num_params()


def test_kv_cache_matches_full_forward(tok):
    torch.manual_seed(0)
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    full, _ = model(x)
    cache = model.new_cache()
    out1, _ = model(x[:, :8], cache=cache)
    out2, _ = model(x[:, 8:], cache=cache)
    assert cache.pos == 12
    assert torch.allclose(full[:, :8], out1, atol=1e-4)
    assert torch.allclose(full[:, 8:], out2, atol=1e-4)


def test_causality_future_tokens_do_not_affect_past(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    y = x.clone(); y[0, 7:] = (y[0, 7:] + 1) % cfg.vocab_size
    lx, _ = model(x); ly, _ = model(y)
    assert torch.allclose(lx[:, :7], ly[:, :7], atol=1e-5)
    assert not torch.allclose(lx[:, 7:], ly[:, 7:])


def test_moe_and_mtp_forward(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size, use_moe=True, n_experts=4, n_experts_active=2,
                 n_shared_experts=1, mtp_heads=1)
    model = TinyLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, loss = model(x, x)
    assert logits.shape == (2, 12, cfg.vocab_size) and torch.isfinite(loss)
    counts = model.blocks[0].mlp.last_expert_counts
    assert counts.sum().item() == 2 * 12 * 2                      # every token routed to k experts
    loss.backward()
    assert model.blocks[0].mlp.router.weight.grad is not None


def test_loss_mask_ignores_masked_tokens(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float)
    _, l_masked = model(x, x, loss_mask=mask)
    _, l_tail = model(x[:, 4:], x[:, 4:]) if False else (None, None)
    per_tok = torch.nn.functional.cross_entropy(model(x)[0].reshape(-1, cfg.vocab_size), x.reshape(-1), reduction="none")
    assert torch.allclose(l_masked, per_tok[4:].mean(), atol=1e-5)


def test_save_load_roundtrip(tok, tmp_path):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg).eval()
    p = str(tmp_path / "m.pt")
    model.save(p, extra={"stage": "test"})
    m2 = TinyLM.load(p)
    x = torch.randint(0, cfg.vocab_size, (1, 6))
    assert torch.allclose(model(x)[0], m2(x)[0])


def test_extend_context(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size, max_seq_len=32)
    model = TinyLM(cfg)
    model.extend_context(64, theta=50_000.0)
    x = torch.randint(0, cfg.vocab_size, (1, 60))
    assert model(x)[0].shape == (1, 60, cfg.vocab_size)


# ----------------------------------------------------------------- generation
def test_sampling_knobs():
    logits = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    assert sample_next(logits, temperature=0).item() == 3
    g = torch.Generator().manual_seed(0)
    ids = [sample_next(logits, temperature=1.0, top_k=1, generator=g).item() for _ in range(5)]
    assert ids == [3] * 5
    ids = [sample_next(logits, temperature=1.0, top_p=0.5, generator=g).item() for _ in range(20)]
    assert set(ids) == {3}
    ids = [sample_next(logits, temperature=1.0, min_p=0.5, generator=g).item() for _ in range(20)]
    assert set(ids) == {3}
    # high temperature flattens: all four should appear
    ids = {sample_next(logits, temperature=50.0, generator=g).item() for _ in range(200)}
    assert ids == {0, 1, 2, 3}


def test_generate_and_stop(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg).eval()
    txt = generate(model, tok, "Mia had", max_new_tokens=12, temperature=1.0, seed=1)
    assert isinstance(txt, str)
    ids = torch.tensor([tok.encode("Mia")])
    out = generate_ids(model, ids, 5, temperature=0.0)
    assert out.shape == (1, ids.shape[1] + 5)
    grp = sample_group(model, tok.encode("Mia had"), n=4, max_new_tokens=6, seed=0,
                       stop_ids=[tok.special_tokens["<|end|>"]], pad_id=tok.special_tokens["<|pad|>"])
    assert grp.shape[0] == 4 and grp.shape[1] <= 2 + 6 + 1


def test_cache_and_no_cache_agree_greedy(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg).eval()
    ids = torch.tensor([tok.encode("Leo went")])
    a = generate_ids(model, ids, 8, temperature=0.0, use_cache=True)
    b = generate_ids(model, ids, 8, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


# ------------------------------------------------------------------ optimizer
def test_newton_schulz_orthogonalizes():
    G = torch.randn(16, 8)
    X = newton_schulz_orthogonalize(G, steps=8)
    s = torch.linalg.svdvals(X)
    assert (s > 0.6).all() and (s < 1.3).all()         # singular values driven towards ~1 (not exactly)


def test_muon_steps_and_split(tok):
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg)
    muon_p, adam_p = split_params(model)
    assert all(p.ndim == 2 for p in muon_p) and not any(p is model.embed.weight for p in muon_p)
    opts = build_optimizer(model, "muon")
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    before = model.blocks[0].attn.q_proj.weight.clone()
    _, loss = model(x, x); loss.backward()
    for o in opts: o.step()
    assert not torch.equal(before, model.blocks[0].attn.q_proj.weight)


def test_lr_schedules():
    assert lr_at(0, 100, 1.0, warmup_steps=10) == pytest.approx(0.1)
    assert lr_at(9, 100, 1.0, warmup_steps=10) == pytest.approx(1.0)
    assert lr_at(99, 100, 1.0, warmup_steps=10, kind="cosine", min_ratio=0.1) == pytest.approx(0.1, abs=0.01)
    assert lr_at(50, 100, 1.0, warmup_steps=10, kind="wsd") == 1.0
    assert lr_at(99, 100, 1.0, warmup_steps=10, kind="wsd") < 0.1


# ------------------------------------------------------------------- training
def test_training_reduces_loss_and_resumes(tok, corpus, tmp_path):
    tokens = tokenize_and_pack(corpus, tok)
    tr, va = split_train_val(tokens)
    cfg = preset("nano", vocab_size=tok.vocab_size)
    model = TinyLM(cfg)
    l0 = estimate_loss(model, va, 8, 32, 3)
    ck = str(tmp_path / "ck.pt")
    tc = TrainConfig(steps=40, batch_size=8, seq_len=32, warmup_steps=5, eval_every=20, log_every=10,
                     ckpt_path=ck, optimizer="adamw", lr=2e-3)
    h = train(model, tr, va, tc, verbose=False)
    l1 = estimate_loss(model, va, 8, 32, 3)
    assert l1 < l0 - 0.5
    assert len(h.step) > 0 and len(h.val_loss) >= 1 and os.path.exists(ck)
    # resume: continue to 60 steps from the checkpoint
    tc2 = TrainConfig(**{**tc.to_dict(), "steps": 60})
    h2 = train(TinyLM(cfg), tr, va, tc2, resume_from=ck, verbose=False)
    assert h2.step[-1] == 59


def test_get_batch_shift(corpus, tok):
    tokens = tokenize_and_pack(corpus[:20], tok)
    x, y = get_batch(tokens, 4, 16, torch.Generator().manual_seed(0))
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1])


# ------------------------------------------------------------------------ data
def test_curation_catches_planted_noise(corpus):
    eval_qs = ["What is the color of the sky above the tallest castle in Storyland today?"]
    dirty = add_noise(corpus, eval_questions=eval_qs)
    good = [d["text"] for d in corpus[:150]]
    bad = [d["text"] for d in dirty if d.get("planted") in ("spam", "non_english", "too_short")][:150]
    clf = QualityClassifier().fit(good + bad, [1] * len(good) + [0] * len(bad))
    report = CurationReport()
    clean = curate(dirty, eval_texts=eval_qs, clf=clf, report=report)
    planted_left = [d["planted"] for d in clean if d.get("planted")]
    assert "spam" not in planted_left and "non_english" not in planted_left
    assert "too_short" not in planted_left and "contaminated" not in planted_left
    # deduplication: every clean text is unique, and (almost) no original survives alongside its copy
    assert len({d["text"] for d in clean}) == len(clean)
    from collections import Counter
    per_origin = Counter(d.get("orig_id") for d in clean if d.get("orig_id"))
    assert sum(c - 1 for c in per_origin.values()) <= 3   # MinHash catches nearly all near-duplicates
    assert all("@" not in d["text"] for d in clean)      # PII scrubbed
    assert len(clean) <= len(corpus) + 5
    assert "exact_dedup" in report.table()


def test_minhash_estimates_jaccard():
    a = shingles("mia had a red kite and took it to the park on a sunny day")
    b = shingles("mia had a red kite and took it to the beach on a sunny day")
    est = (minhash_signature(a, 128) == minhash_signature(b, 128)).mean()
    assert abs(est - jaccard(a, b)) < 0.2


def test_gopher_and_pii():
    assert gopher_reason("ok") == "too_short"
    assert gopher_reason("Click here to subscribe now for free stuff and more free stuff") == "boilerplate"
    assert gopher_reason("Mia had a red kite and she took it to the park today.") is None
    t, n = scrub_pii("mail mia@example.com or 555-0123-4567 now")
    assert n == 2 and "<EMAIL>" in t and "<PHONE>" in t


def test_mix_and_decontaminate(corpus):
    mixed = mix_sources(corpus, {"stories": 1.0, "math": 3.0}, 400, seed=0)
    frac_math = sum(d["source"] == "math" for d in mixed) / 400
    assert 0.6 < frac_math < 0.9
    docs = [{"text": "the quick brown fox jumps over the lazy dog every single day"}, {"text": "unrelated text here about kites"}]
    kept = decontaminate(docs, ["quick brown fox jumps over the lazy dog every"], n=5)
    assert len(kept) == 1


# ------------------------------------------------------------------------ chat
def test_chat_template_and_mask(tok):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert render(msgs, add_generation_prompt=False) == "<|bos|><|user|>hi<|end|><|assistant|>hello<|end|>"
    ids, mask = build_sft_example(tok, msgs)
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    assert trained == "hello<|end|>"
    x, y, m = collate([(ids, mask), (ids[:4], mask[:4])], tok.special_tokens["<|pad|>"])
    assert x.shape == y.shape == m.shape and x.shape[1] == len(ids) - 1
    assert parse_tool_call('<|tool_call|>{"name": "calc", "arguments": {"expression": "1+1"}}<|end|>')["name"] == "calc"


# --------------------------------------------------------------- autograd/tasks
def test_microautograd_matches_torch():
    a, b = Value(2.0), Value(-3.0)
    c = (a * b + a ** 2).tanh() * 4 + (a / b).exp()
    c.backward()
    ta, tb = torch.tensor(2.0, requires_grad=True), torch.tensor(-3.0, requires_grad=True)
    tc = torch.tanh(ta * tb + ta ** 2) * 4 + torch.exp(ta / tb)
    tc.backward()
    assert abs(a.grad - ta.grad.item()) < 1e-5 and abs(b.grad - tb.grad.item()) < 1e-5


def test_xor_learns():
    losses = train_xor(steps=150, lr=0.1)
    assert losses[-1] < 0.1 * losses[0] + 0.05


def test_tasks_verify():
    ex = make_examples(20, seed=3)
    assert all(verify(e, e.answer) == 1.0 for e in ex)
    assert extract_answer("add", "So 3 + 4 = 7.") == "7"
    assert extract_answer("add", "I think the answer is 12 or maybe 13") == "13"
