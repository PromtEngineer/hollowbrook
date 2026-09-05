# Appendix B: PyTorch in one page

Everything the course's code uses, with the mental model behind it. Run any block with `python3`.

## B.1 Tensors are arrays with a shape

```python
import torch
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])   # shape (2, 3): 2 rows, 3 columns
x.shape            # torch.Size([2, 3])
x.dtype            # torch.float32 (32-bit floats; bf16/fp16 are cheaper, see Chapter 10)
x.T                # transpose -> (3, 2)
x @ x.T            # matrix multiply -> (2, 2)
x.sum(dim=-1)      # add along the LAST dimension -> (2,)  [6., 15.]
x[:, 1]            # column 1 -> tensor([2., 5.])
x.view(3, 2)       # same numbers, new shape (no copy); .reshape also allows copies
```

**Reading shapes is the skill.** The course writes shapes in comments: `# (B, T, d)`. When two tensors
are combined, ask "which dimension lines up with which?".

**Broadcasting**: a `(2, 3)` tensor times a `(3,)` tensor multiplies each row by the vector.
Trailing dimensions are matched from the right; size-1 dimensions stretch.

## B.2 Autograd: every operation records how to compute gradients

```python
w = torch.tensor(0.0, requires_grad=True)
loss = (w - 3) ** 2
loss.backward()     # walks the graph backwards (Chapter 4)
w.grad              # tensor(-6.)
```

`torch.no_grad()` turns recording off (for evaluation and generation: faster, less memory).
`.detach()` gives a tensor that is cut off from the graph — used when we want a *value*
(e.g. old log-probabilities in PPO, Chapter 18) but no gradient through it.

## B.3 Modules hold parameters

```python
import torch.nn as nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)         # a matrix (2, 4) and a bias (2,), both trainable
    def forward(self, x):                  # x: (B, 4)
        return self.lin(x)                 # -> (B, 2)

m = Tiny()
sum(p.numel() for p in m.parameters())    # 10 parameters
```

`nn.Embedding(V, d)` is a lookup table (Chapter 3). `nn.ModuleList` holds a list of sub-modules
(TinyLM's blocks). `model.state_dict()` is the dictionary of all weights; `torch.save` / `torch.load`
serialise it. `model.train()` / `model.eval()` switch modes (dropout etc.).

## B.4 The training loop skeleton

```python
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
for step in range(100):
    x, y = get_batch()                    # your data
    loss = loss_fn(m(x), y)               # forward
    opt.zero_grad(set_to_none=True)       # clear old gradients
    loss.backward()                       # backward: fills .grad on every parameter
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)   # (Chapter 10)
    opt.step()                            # update parameters
```

The course's `llm/train.py` is this loop plus a learning-rate schedule, validation and checkpoints.

## B.5 The functions the course leans on

| Call | What it does |
|---|---|
| `F.softmax(x, dim=-1)` | probabilities along the last dimension |
| `F.log_softmax(x, dim=-1)` | log-probabilities, numerically stable |
| `F.cross_entropy(logits, targets)` | mean `−log p(target)`; `ignore_index=-100` skips positions |
| `x.gather(-1, idx)` | pick one entry per row (used to get `log p(actual token)`) |
| `torch.multinomial(p, 1)` | sample an index according to probabilities `p` |
| `x.topk(k)` | the `k` largest values and their indices |
| `x.masked_fill(mask, value)` | write `value` wherever `mask` is True (the causal mask) |
| `torch.cat([a, b], dim=1)` | concatenate along a dimension (growing the KV cache) |
| `x.repeat_interleave(r, dim=1)` | repeat each entry `r` times (GQA head sharing) |
| `torch.manual_seed(0)` / `torch.Generator().manual_seed(0)` | reproducibility |

## B.6 Speed on a CPU

- Set threads once: `torch.set_num_threads(n)`; the course uses the default (all cores).
- Larger matrix multiplications are proportionally faster than many small ones — batch your work.
- `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` can speed up recent CPUs; the course keeps
  float32 by default for clarity (see `TrainConfig.dtype`).
- Profiling in one line: `python3 -X importtime` for imports, `torch.profiler` for ops; usually you
  only need `time.perf_counter()` around the training step, which is what `llm/train.py` logs as tokens/s.

## B.7 Common errors and what they mean

| Error | Usual cause |
|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | a dimension mismatch — print `.shape` of both |
| `expected scalar type Float but found Long` | you passed integer token ids where floats were expected (or vice versa) |
| `CUDA out of memory` / process killed on CPU | batch or sequence too long; halve `batch_size` |
| loss is `nan` | learning rate too high, or `log(0)` somewhere — see Appendix F |
| `Trying to backward through the graph a second time` | you reused a tensor from a previous step without `.detach()` |
