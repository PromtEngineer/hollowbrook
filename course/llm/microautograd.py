"""A scalar automatic-differentiation engine in ~120 lines (Chapter 4).

This is the whole idea behind PyTorch's autograd, small enough to read in one
sitting. Every ``Value`` remembers how it was made (its parents and the local
derivative rule), so calling ``.backward()`` on the final loss walks the graph in
reverse and accumulates d(loss)/d(value) into every ``.grad``.

Inspired by Andrej Karpathy's micrograd.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Iterable


class Value:
    """A single number that tracks its gradient."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data: float, _children: tuple = (), _op: str = "", label: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0                        # d(loss) / d(self), filled by backward()
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                       # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                       # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k: float):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():                       # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():                       # d(e^a)/da = e^a
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():                       # d(ln a)/da = 1/a
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():                       # gradient passes only where a > 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():                       # d(tanh a)/da = 1 - tanh^2 a
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # convenience operators built from the ones above
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return Value(other) + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)
    def __rtruediv__(self, other): return Value(other) * self ** -1
    def __repr__(self): return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------- backward
    def backward(self) -> None:
        """Topologically order the graph, then apply each node's local rule in reverse."""
        order: list[Value] = []
        seen: set[int] = set()

        def build(v: Value) -> None:
            if id(v) not in seen:
                seen.add(id(v))
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0                        # d(loss)/d(loss) = 1
        for v in reversed(order):
            v._backward()


# ------------------------------------------------------------ a tiny network
class Neuron:
    def __init__(self, n_in: int, rng: random.Random, nonlin: bool = True) -> None:
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Iterable[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int, rng: random.Random, nonlin: bool = True) -> None:
        self.neurons = [Neuron(n_in, rng, nonlin) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """A multi-layer perceptron: sizes=[3, 4, 4, 1] means 3 inputs, two hidden layers of 4, 1 output."""

    def __init__(self, sizes: list[int], seed: int = 0) -> None:
        rng = random.Random(seed)
        self.layers = [Layer(sizes[i], sizes[i + 1], rng, nonlin=(i < len(sizes) - 2))
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


def sgd_step(params: Iterable[Value], lr: float) -> None:
    """The optimizer: move each parameter a little *against* its gradient."""
    for p in params:
        p.data -= lr * p.grad


def train_xor(steps: int = 200, lr: float = 0.1, seed: int = 0) -> list[float]:
    """Learn XOR — the classic problem a single linear neuron cannot solve."""
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1, 1, 1, -1]
    net = MLP([2, 4, 1], seed)
    losses = []
    for _ in range(steps):
        preds = [net(x) for x in xs]
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0)) * (1 / len(xs))
        net.zero_grad()
        loss.backward()
        sgd_step(net.parameters(), lr)
        losses.append(loss.data)
    return losses
