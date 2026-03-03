from micrograd.engine import Value
import random


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1) for _ in range(nin))]
        self.b = Value(0)

    def __call__(self, x):
        act = Value(sum([xi * bi for xi, bi in zip(x, self.w)], self.b))
        out = act.tanh()
        return out[0] if len(out) == 1 else out


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]


def MLP(self):
    def __init__(self, nin: int, nouts: list[int]):
        self.sz = [nin] + nouts
        self.layers = [Layer(self.sz[i], self.sz[i + 1]) for i in range(len(self.sz))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
