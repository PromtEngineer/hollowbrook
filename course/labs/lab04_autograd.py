"""Lab 4: how neural networks learn — autograd, optimizers, learning rate, overfitting.

What you will see:
  1. One neuron, by hand, with llm.microautograd.Value — gradients match PyTorch.
  2. A derivative is a slope: finite differences agree with autograd.
  3. XOR learned with the micro autograd engine, then the same thing in PyTorch.
  4. SGD vs momentum vs Adam rolling down a narrow valley (numpy, 2-D).
  5. Learning rate too low / right / too high.
  6. Train/validation split and overfitting on a polynomial fit.

Run:  python3 labs/lab04_autograd.py            (--quick, ~20 s)
      python3 labs/lab04_autograd.py --full     (longer runs, more seeds, ~1-2 min)
"""
from _common import setup, check, banner, section, savefig, done, plt

import math
import time

import numpy as np
import torch
import torch.nn as nn

from llm.microautograd import Value, MLP, sgd_step, train_xor

args = setup("Lab 4: how neural networks learn")
plt = plt()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

# ----------------------------------------------------------------------------
section("1. One neuron by hand: forward values and backward gradients")
# o = tanh(x1*w1 + x2*w2 + b)  — the classic micrograd example
x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")           # chosen so that n = 0.8814 and o = 0.7071
x1w1 = x1 * w1; x1w1.label = "x1*w1"
x2w2 = x2 * w2; x2w2.label = "x2*w2"
n = x1w1 + x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"
o.backward()                                       # fills every .grad with d(o)/d(node)
for v in (x1, w1, x2, w2, b, n, o):
    print(f"  {v.label:<5} data = {v.data:8.4f}   grad = {v.grad:8.4f}")

# the same graph in PyTorch
tx1, tx2 = torch.tensor(2.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
tw1, tw2 = torch.tensor(-3.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)
tb = torch.tensor(6.8813735870195432, requires_grad=True)
to = torch.tanh(tx1 * tw1 + tx2 * tw2 + tb)
to.backward()
diff = max(abs(a.grad - t.grad.item()) for a, t in [(x1, tx1), (w1, tw1), (x2, tx2), (w2, tw2), (b, tb)])
print(f"  torch: o = {to.item():.4f}; max |grad difference| vs microautograd = {diff:.1e}")
check(diff < 1e-5, "microautograd gradients match torch.autograd")
check(abs(w1.grad - 1.0) < 1e-3 and abs(x1.grad + 1.5) < 1e-3, "chain rule: dw1 = x1 * (1 - o^2) = 2 * 0.5 = 1.0")

# ----------------------------------------------------------------------------
section("2. A derivative is a slope: finite differences vs autograd")


def f(x):                      # any function built from Value ops
    return (x * x * 3.0 + x * -2.0 + 1.0).relu() * (x * 0.5).tanh()


x0 = 1.7
xv = Value(x0)
y = f(xv); y.backward()
h = 1e-5
slope = (f(Value(x0 + h)).data - f(Value(x0 - h)).data) / (2 * h)
print(f"  f({x0}) = {y.data:.4f};  autograd df/dx = {xv.grad:.5f};  (f(x+h)-f(x-h))/2h = {slope:.5f}")
check(abs(xv.grad - slope) < 1e-4, "autograd derivative equals the numerical slope")

# ----------------------------------------------------------------------------
section("3. Learning XOR: the micro autograd engine, then PyTorch")
steps = 200 if args.quick else 1000
t0 = time.perf_counter()
micro_losses = train_xor(steps=steps, lr=0.1, seed=args.seed)
micro_time = time.perf_counter() - t0
marks = [0, 25, 50, 100, steps - 1]
print("  microautograd loss:", "  ".join(f"step {s}: {micro_losses[s]:.4f}" for s in marks))
print(f"  {steps} steps in {micro_time:.2f}s ({1000 * micro_time / steps:.1f} ms/step)")
check(micro_losses[-1] < 0.05, f"microautograd XOR loss {micro_losses[-1]:.4f} < 0.05 after {steps} steps")

# The same network, the same data, the same optimizer — but every line is PyTorch's job now.
xs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])         # (4, 2)
ys = torch.tensor([[-1.], [1.], [1.], [-1.]])                       # (4, 1)
net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))    # same shape as MLP([2, 4, 1])
opt = torch.optim.SGD(net.parameters(), lr=0.1)
torch_losses = []
t0 = time.perf_counter()
for _ in range(steps):
    pred = net(xs)                                                  # (4, 1)
    loss = ((pred - ys) ** 2).mean()                                # same MSE as train_xor
    opt.zero_grad()                                                 # net.zero_grad()
    loss.backward()                                                 # loss.backward()
    opt.step()                                                      # sgd_step(params, lr)
    torch_losses.append(loss.item())
torch_time = time.perf_counter() - t0
print("  torch          loss:", "  ".join(f"step {s}: {torch_losses[s]:.4f}" for s in marks))
print(f"  {steps} steps in {torch_time:.2f}s ({1000 * torch_time / steps:.1f} ms/step)")
check(torch_losses[-1] < 0.05, f"torch XOR loss {torch_losses[-1]:.4f} < 0.05 after {steps} steps")
with torch.no_grad():
    print("  torch predictions:", [round(v, 2) for v in net(xs).squeeze(1).tolist()], " targets:", ys.squeeze(1).tolist())

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(micro_losses, label="llm.microautograd (Value)")
ax.plot(torch_losses, label="torch (nn.Module + SGD)", alpha=0.8)
ax.set_xlabel("step"); ax.set_ylabel("MSE loss"); ax.set_yscale("log")
ax.set_title("XOR: same network, same data, two autograd engines"); ax.legend(); ax.grid(alpha=0.3)
savefig(fig, "lab04_xor_loss.png")

# ----------------------------------------------------------------------------
section("4. SGD vs momentum vs Adam on a narrow valley (numpy)")
# loss(x, y) = x^2/20 + y^2 : a bowl that is 20x wider along x than along y.
# The gradient points mostly along y (the steep wall), barely along x (the valley floor).


def loss_fn(p):
    return p[0] ** 2 / 20 + p[1] ** 2


def grad_fn(p):
    return np.array([p[0] / 10, 2 * p[1]])


def run(optimizer: str, lr: float, n_steps: int, start=(-9.0, 3.0)):
    p = np.array(start, dtype=float)
    v = np.zeros(2)                       # momentum buffer
    m, s = np.zeros(2), np.zeros(2)       # Adam: first and second moments
    b1, b2, eps = 0.9, 0.999, 1e-8
    path = [p.copy()]
    for t in range(1, n_steps + 1):
        g = grad_fn(p)
        if optimizer == "sgd":
            p = p - lr * g
        elif optimizer == "momentum":
            v = 0.9 * v + g                   # remember the direction we were going
            p = p - lr * v
        elif optimizer == "adam":
            m = b1 * m + (1 - b1) * g         # running mean of the gradient
            s = b2 * s + (1 - b2) * g * g     # running mean of its square
            m_hat, s_hat = m / (1 - b1 ** t), s / (1 - b2 ** t)
            p = p - lr * m_hat / (np.sqrt(s_hat) + eps)   # per-coordinate step ≈ lr in size
        path.append(p.copy())
    return np.array(path)


n_opt_steps = 100 if args.quick else 300
runs = {"SGD (lr 0.4)": run("sgd", 0.4, n_opt_steps),
        "momentum (lr 0.04, β 0.9)": run("momentum", 0.04, n_opt_steps),
        "Adam (lr 0.3)": run("adam", 0.3, n_opt_steps)}
steps_to = {}
for name, path in runs.items():
    vals = np.array([loss_fn(p) for p in path])
    hit = np.argmax(vals < 1e-3) if (vals < 1e-3).any() else None
    steps_to[name] = hit
    print(f"  {name:<28} final loss {vals[-1]:.2e}   steps to loss<1e-3: {hit if hit is not None else '>' + str(n_opt_steps)}")
check(steps_to["Adam (lr 0.3)"] is not None and (steps_to["SGD (lr 0.4)"] is None or steps_to["Adam (lr 0.3)"] < steps_to["SGD (lr 0.4)"]),
      "Adam reaches the bottom of the valley in fewer steps than plain SGD")

gx, gy = np.meshgrid(np.linspace(-10, 3, 200), np.linspace(-3.5, 3.5, 200))
fig, ax = plt.subplots(figsize=(7, 4))
ax.contour(gx, gy, gx ** 2 / 20 + gy ** 2, levels=np.logspace(-2, 1, 14), cmap="Greys", alpha=0.7)
for (name, path), col in zip(runs.items(), ["#2563eb", "#f59e0b", "#16a34a"]):
    ax.plot(path[:, 0], path[:, 1], "-o", ms=2.5, lw=1, color=col, label=name)
ax.plot([0], [0], "k*", ms=12, label="minimum")
ax.set_xlabel("x (flat direction)"); ax.set_ylabel("y (steep direction)")
ax.set_title("loss = x²/20 + y²: three optimizers, same start"); ax.legend(fontsize=8); ax.set_aspect("equal")
savefig(fig, "lab04_optimizers.png")

# ----------------------------------------------------------------------------
section("5. Learning rate: too low, about right, too high (loss = y^2)")
# One step of gradient descent on y^2 multiplies y by (1 - 2*lr).
lrs = {"too low (0.05)": 0.05, "good (0.4)": 0.4, "too high (1.05)": 1.05}
traces = {}
for name, lr in lrs.items():
    y = 3.0
    tr = [y]
    for _ in range(15):
        y = y - lr * 2 * y
        tr.append(y)
    traces[name] = tr
    print(f"  lr {name:<16}: y after 15 steps = {tr[-1]:10.3f}   (each step multiplies y by {1 - 2 * lr:+.2f})")
check(abs(traces["good (0.4)"][-1]) < 1e-6 and abs(traces["too high (1.05)"][-1]) > 3.0,
      "lr 0.4 converges, lr 1.05 diverges (|1 - 2 lr| > 1)")
fig, ax = plt.subplots(figsize=(6, 3.4))
yy = np.linspace(-4, 4, 100)
ax.plot(yy, yy ** 2, "k-", alpha=0.4)
for (name, tr), col in zip(traces.items(), ["#2563eb", "#16a34a", "#dc2626"]):
    tr = np.array(tr[:8])
    ax.plot(tr, tr ** 2, "-o", ms=4, color=col, label=name)
ax.set_ylim(-0.5, 16); ax.set_xlabel("y"); ax.set_ylabel("loss = y²"); ax.legend(); ax.set_title("gradient descent steps at three learning rates")
savefig(fig, "lab04_learning_rate.png")

# ----------------------------------------------------------------------------
section("6. Train/validation split and overfitting (polynomial fit)")


def true_fn(x):
    return np.sin(2 * np.pi * x)


n_seeds = 3 if args.quick else 10
degrees = [1, 3, 5, 7, 11, 15]
train_err = np.zeros((n_seeds, len(degrees)))
val_err = np.zeros((n_seeds, len(degrees)))
fits = {}                                                  # seed-0 fits, for the plot
for s in range(n_seeds):
    r = np.random.default_rng(args.seed + s)
    x = np.sort(r.uniform(-1, 1, 20))
    y = true_fn(x) + 0.15 * r.standard_normal(20)         # noisy measurements
    idx = r.permutation(20)
    tr_i, va_i = idx[:12], idx[12:]                        # 12 points to train on, 8 held out
    for j, deg in enumerate(degrees):
        # least-squares fit of a degree-`deg` polynomial (the closed-form "training")
        A_tr = np.vander(x[tr_i], deg + 1)
        coef, *_ = np.linalg.lstsq(A_tr, y[tr_i], rcond=None)
        train_err[s, j] = np.mean((A_tr @ coef - y[tr_i]) ** 2)
        val_err[s, j] = np.mean((np.vander(x[va_i], deg + 1) @ coef - y[va_i]) ** 2)
        if s == 0:
            fits[deg] = (coef, x[tr_i], y[tr_i], x[va_i], y[va_i])
tr_m, va_m = np.median(train_err, 0), np.median(val_err, 0)   # median: one wild fit should not hide the trend
print(f"  {'degree':>6} {'train MSE':>10} {'val MSE':>12}   (median over {n_seeds} seeds; 12 train / 8 validation points)")
for deg, a, b_ in zip(degrees, tr_m, va_m):
    print(f"  {deg:>6} {a:10.4f} {b_:12.4f}")
best_j = int(np.argmin(va_m))
print(f"  best degree by validation error: {degrees[best_j]}  (train error keeps falling all the way to degree {degrees[-1]})")
check(tr_m[-1] <= tr_m[best_j] + 1e-9 and va_m[-1] > va_m[best_j],
      f"degree {degrees[-1]} fits the training points at least as well as degree {degrees[best_j]} but generalises worse")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
xx = np.linspace(-1, 1, 300)
axes[0].plot(xx, true_fn(xx), "k--", alpha=0.5, label="true function")
for deg, col in zip([1, 5, 15], ["#64748b", "#16a34a", "#dc2626"]):
    coef, xt, yt, xv, yv = fits[deg]
    axes[0].plot(xx, np.vander(xx, deg + 1) @ coef, color=col, label=f"degree {deg}")
axes[0].scatter(fits[5][1], fits[5][2], color="#2563eb", zorder=5, label="train")
axes[0].scatter(fits[5][3], fits[5][4], color="#f59e0b", zorder=5, marker="s", label="validation")
axes[0].set_ylim(-2, 2); axes[0].legend(fontsize=7); axes[0].set_title("polynomial fits to 12 noisy points")
axes[1].plot(degrees, tr_m, "-o", color="#2563eb", label="train MSE")
axes[1].plot(degrees, va_m, "-s", color="#f59e0b", label="validation MSE")
axes[1].set_yscale("log"); axes[1].set_xlabel("polynomial degree (model capacity)"); axes[1].legend()
axes[1].set_title("overfitting: train keeps falling, validation turns up"); axes[1].grid(alpha=0.3)
savefig(fig, "lab04_overfitting.png")

done()
