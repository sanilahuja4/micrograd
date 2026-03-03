from micrograd.engine import Value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum([xi * bi for xi, bi in zip(x, self.w)], self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return [self.b] + self.w


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        param_list = []
        for neuron in self.neurons:
            param_list.extend(neuron.parameters())
        return param_list


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]):
        self.sz = [nin] + nouts
        self.layers = [
            Layer(self.sz[i], self.sz[i + 1]) for i in range(len(self.sz) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        param_list = []
        for layer in self.layers:
            param_list.extend(layer.parameters())
        return param_list
