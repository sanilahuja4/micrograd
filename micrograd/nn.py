from micrograd.engine import Value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activation="tanh"):
        # Use He initialization for ReLU, Xavier for tanh
        if activation == "relu":
            # He initialization: scale by sqrt(2/nin)
            scale = (2.0 / nin) ** 0.5
            self.w = [Value(random.gauss(0, scale)) for _ in range(nin)]
            self.b = Value(0.01)  # Small positive bias to prevent dead neurons
        else:
            # Xavier/Glorot initialization for tanh
            scale = (1.0 / nin) ** 0.5
            self.w = [Value(random.uniform(-scale, scale)) for _ in range(nin)]
            self.b = Value(0)
        self.activation = activation

    def __call__(self, x):
        act = sum([xi * bi for xi, bi in zip(x, self.w)], self.b)
        if self.activation == "tanh":
            out = act.tanh()
        elif self.activation == "relu":
            out = act.relu()
        else:
            out = act  # linear activation (no activation)
        return out

    def parameters(self):
        return [self.b] + self.w


class Layer(Module):
    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        param_list = []
        for neuron in self.neurons:
            param_list.extend(neuron.parameters())
        return param_list


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int], activation="tanh"):
        self.sz = [nin] + nouts

        # Support both single activation or list of activations (one per layer)
        if isinstance(activation, str):
            activations = [activation] * len(nouts)
        else:
            activations = activation
            assert len(activations) == len(
                nouts
            ), "Number of activations must match number of layers"

        self.layers = [
            Layer(self.sz[i], self.sz[i + 1], activation=activations[i])
            for i in range(len(self.sz) - 1)
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
