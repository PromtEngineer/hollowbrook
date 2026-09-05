"""Lab 6: assemble TinyLM, count its parameters by hand, check shapes, watch the init, generate.

    python3 labs/lab06_build_tinylm.py            # --quick: nano base model (~1 min)
    python3 labs/lab06_build_tinylm.py --full     # small base model (trains it if runs/base_small.pt is missing)
"""
from _common import setup, check, banner, section, savefig, done, plt
import math
import torch
from llm.config import preset
from llm.model import TinyLM
from llm.pipeline import get_tokenizer, get_base_model, get_tokens
from llm.generate import generate

args = setup("Lab 6: build TinyLM")
torch.manual_seed(args.seed)
PROMPT = "At the park, Mia met"

# ----------------------------------------------------------------------------- 1
section("1. The config of the course model: preset('small')")
cfg = preset("small")
d, V, L, h, kv, hd, ff = (cfg.d_model, cfg.vocab_size, cfg.n_layers, cfg.n_heads,
                          cfg.n_kv_heads, cfg.head_dim, cfg.d_ff)
print(f"d_model={d}  vocab={V}  layers={L}  heads={h}  kv_heads={kv}  head_dim={hd}  d_ff={ff}")
print(f"d_ff by the 8/3 rule: 8/3*{d} = {8*d/3:.1f} -> rounded to a multiple of 64 = {ff}")
model = TinyLM(cfg)

# ----------------------------------------------------------------------------- 2
section("2. Parameter count by hand vs model.num_params()")
hand = {
    "embed (V x d)":            V * d,
    "  attn q_proj (d x h*hd)": d * h * hd,
    "  attn k_proj (d x kv*hd)": d * kv * hd,
    "  attn v_proj (d x kv*hd)": d * kv * hd,
    "  attn o_proj (h*hd x d)": h * hd * d,
    "  mlp gate_proj (d x ff)": d * ff,
    "  mlp up_proj   (d x ff)": d * ff,
    "  mlp down_proj (ff x d)": ff * d,
    "  2 RMSNorm gains (2d)":   2 * d,
}
per_block = sum(v for k, v in hand.items() if k.startswith("  "))
hand_non_embed = L * per_block + d                     # + final_norm
print(f"{'component':30s} {'count':>10s}")
for k, v in hand.items():
    print(f"{k:30s} {v:10,d}")
print(f"{'per block':30s} {per_block:10,d}")
print(f"{'x L blocks + final norm':30s} {hand_non_embed:10,d}")
print(f"{'lm_head (tied -> 0 extra)':30s} {0:10,d}")
print(f"{'TOTAL incl. embedding':30s} {hand_non_embed + V*d:10,d}")
lib_non_embed, lib_total = model.num_params(), model.num_params(non_embedding=False)
print(f"library: non-embedding {lib_non_embed:,}  total {lib_total:,}")
check(hand_non_embed == lib_non_embed, "hand-counted non-embedding params == model.num_params()")
check(hand_non_embed + V * d == lib_total, "hand-counted total == model.num_params(non_embedding=False)")
# every tensor, by shape (block 0 only; the others are identical)
for name, p in model.named_parameters():
    if name.startswith("blocks.0") or not name.startswith("blocks"):
        print(f"  {name:32s} {str(tuple(p.shape)):14s} {p.numel():9,d}")

untied = TinyLM(preset("small", tie_embeddings=False)).num_params(non_embedding=False)
print(f"untied lm_head would add V*d = {V*d:,}: total {untied:,} vs tied {lib_total:,}")
check(untied - lib_total == V * d, "untying the head costs exactly V*d parameters")
check(model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr(), "tied: lm_head and embed share one tensor")

# ----------------------------------------------------------------------------- 3
section("3. FLOPs per token: 6N + attention term")
N = lib_total
attn_term = 12 * L * d * cfg.max_seq_len
hand_flops = 6 * N + attn_term
print(f"6*N = 6*{N:,} = {6*N:,}")
print(f"attention term 12*L*d*T_max = 12*{L}*{d}*{cfg.max_seq_len} = {attn_term:,} ({attn_term/hand_flops:.1%} of total)")
print(f"hand {hand_flops:,}  vs  model.flops_per_token() {model.flops_per_token():,.0f}")
check(abs(hand_flops - model.flops_per_token()) < 1, "6N + 12*L*d*T == model.flops_per_token()")
D = 700 * 32 * 128
print(f"the full course pretraining run sees D = 700 steps x 32 x 128 = {D:,} tokens "
      f"-> C ~= {hand_flops * D:.2e} FLOPs (a laptop does ~1e11 FLOP/s: ~{hand_flops*D/1e11/60:.0f} min)")

# ----------------------------------------------------------------------------- 4
section("4. Init: residual-stream RMS across depth, with and without the 1/sqrt(2L) scaling")
def rms_per_block(n_layers: int, scaled: bool) -> list[float]:
    torch.manual_seed(0)
    m = TinyLM(preset("small", n_layers=n_layers))
    if not scaled:                                   # undo the library's residual scaling
        for name, p in m.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, 0.0, m.cfg.init_std)
    out = []
    hooks = [b.register_forward_hook(lambda mod, i, o: out.append(o.pow(2).mean().sqrt().item()))
             for b in m.blocks]
    with torch.no_grad():
        m(torch.randint(0, m.cfg.vocab_size, (4, 64)))
    for hk in hooks:
        hk.remove()
    return out

fig, ax = plt().subplots(figsize=(6, 3.6))
results = {}
for n_layers in (6, 12):
    for scaled in (True, False):
        r = rms_per_block(n_layers, scaled)
        results[(n_layers, scaled)] = r
        label = f"L={n_layers} {'scaled 0.02/sqrt(2L)' if scaled else 'unscaled 0.02'}"
        print(f"{label:28s} RMS after block 1..{n_layers}: " + " ".join(f"{v:.3f}" for v in r))
        ax.plot(range(1, n_layers + 1), r, marker="o", ls="-" if scaled else "--", label=label)
ax.set_xlabel("block"); ax.set_ylabel("RMS of residual stream at init"); ax.legend(fontsize=8)
ax.set_title("residual scaling keeps the stream's scale flat with depth")
savefig(fig, "lab06_init_rms.png")
s12, u12 = results[(12, True)], results[(12, False)]
print(f"L=12 growth factor (last/first): scaled {s12[-1]/s12[0]:.2f}x   unscaled {u12[-1]/u12[0]:.2f}x")
check(u12[-1] > 2 * s12[-1], "without residual scaling the stream grows > 2x larger by the last block")
check(abs(results[(6, True)][0] - 0.02) < 0.01, "at init the residual stream has RMS ~ init_std = 0.02")

# ----------------------------------------------------------------------------- 5
section("5. Forward pass shapes and the loss at init")
tok = get_tokenizer()
cfg_c = preset("nano" if args.quick else "small", vocab_size=tok.vocab_size)
fresh = TinyLM(cfg_c)
print(f"course tokenizer vocab = {tok.vocab_size} -> {'nano' if args.quick else 'small'} model has "
      f"{fresh.num_params():,} non-embedding / {fresh.num_params(False):,} total params")
train_tokens, val_tokens = get_tokens(tok)
B, T = 8, 64
x = torch.stack([val_tokens[i * T:(i + 1) * T] for i in range(B)])
y = torch.stack([val_tokens[i * T + 1:(i + 1) * T + 1] for i in range(B)])
logits, loss = fresh(x, y)
print(f"idx {tuple(x.shape)} -> logits {tuple(logits.shape)}  (B, T, V)")
print(f"loss at init {loss.item():.4f}  vs  ln(V) = ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}")
check(logits.shape == (B, T, tok.vocab_size), "logits have shape (B, T, V)")
check(abs(loss.item() - math.log(tok.vocab_size)) < 0.05, "untrained loss ~= ln(V): a uniform guess over the vocabulary")
_, _, hidden = fresh(x, return_hidden=True)
check(hidden.shape == (B, T, cfg_c.d_model), "final hidden state has shape (B, T, d)")

# ----------------------------------------------------------------------------- 6
section("6. Generate from the UNTRAINED model")
print(repr(generate(fresh, tok, PROMPT, max_new_tokens=30, temperature=1.0, seed=args.seed)))

# ----------------------------------------------------------------------------- 7
section("7. Load the pretrained base model and generate")
base, _ = get_base_model(quick=args.quick)
print(f"loaded {'nano' if args.quick else 'small'} base: {base.num_params():,} non-embedding params")
_, base_loss = base(x, y)
print(f"val loss {base_loss.item():.3f} (perplexity {math.exp(base_loss.item()):.1f})  vs untrained {loss.item():.3f}")
check(base_loss.item() < loss.item() - 2.0, "trained model's loss is > 2 nats below the untrained one")
for temp in (0.0, 0.8):
    print(f"T={temp}: {PROMPT!r} -> {generate(base, tok, PROMPT, max_new_tokens=30, temperature=temp, seed=args.seed)!r}")

# ----------------------------------------------------------------------------- 8
section("8. What each part does: ablate attention / MLP sub-layers of the trained model")
def loss_with_ablation(which: str, layer: int) -> float:
    mod = base.blocks[layer].attn if which == "attn" else base.blocks[layer].mlp
    hk = mod.register_forward_hook(lambda m, i, o: torch.zeros_like(o))   # write nothing to the stream
    with torch.no_grad():
        _, l = base(x, y)
    hk.remove()
    return l.item()

base_l = base_loss.item()
print(f"{'layer':>5s} {'no attn':>9s} {'no mlp':>9s}   (loss; baseline {base_l:.3f})")
rows = []
for i in range(base.cfg.n_layers):
    la, lm = loss_with_ablation("attn", i), loss_with_ablation("mlp", i)
    rows.append((la, lm))
    print(f"{i:5d} {la:9.3f} {lm:9.3f}")
check(all(la > base_l and lm > base_l for la, lm in rows), "zeroing any sub-layer's write raises the loss")
fig, ax = plt().subplots(figsize=(6, 3.2))
xs = range(base.cfg.n_layers)
ax.bar([i - 0.2 for i in xs], [r[0] - base_l for r in rows], width=0.4, label="attention removed")
ax.bar([i + 0.2 for i in xs], [r[1] - base_l for r in rows], width=0.4, label="MLP removed")
ax.set_xlabel("block"); ax.set_ylabel("loss increase (nats)"); ax.legend(); ax.set_title("ablating one sub-layer at a time")
savefig(fig, "lab06_ablation.png")

# attention is the only thing that moves information between positions:
hooks = [b.attn.register_forward_hook(lambda m, i, o: torch.zeros_like(o)) for b in base.blocks]
a = torch.tensor([tok.encode("Mia had a red kite")])
b = torch.tensor([tok.encode("Tom lost the blue kite")])
with torch.no_grad():
    la, lb = base(a)[0][0, -1], base(b)[0][0, -1]        # logits for the last token of each
for hk in hooks:
    hk.remove()
print(f"with ALL attention zeroed, last-token logits of two different sentences ending in 'kite' "
      f"differ by max {(la - lb).abs().max().item():.2e}")
check(torch.allclose(la, lb, atol=1e-4), "no attention -> each position only sees its own token (a bigram model)")
done()
