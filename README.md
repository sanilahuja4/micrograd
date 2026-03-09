# micrograd

A lightweight, educational autograd engine and neural network library built from scratch in Python. Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), with extended features including multiple activation functions, proper weight initialization, and flexible neural network architectures.

## 🌟 Features

- **Automatic Differentiation**: Scalar-valued autograd engine with dynamic computation graph
- **Neural Network Library**: Multi-layer perceptron (MLP) with configurable architectures
- **Multiple Activations**: Support for ReLU, tanh, and linear activations with proper initialization
  - He initialization for ReLU
  - Xavier initialization for tanh
- **Training Utilities**: SVM max-margin loss with L2 regularization
- **Visualization**: Computation graph visualization using Graphviz
- **Type-safe**: Full type hints throughout the codebase

## 📦 Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/micrograd.git
cd micrograd

# Install dependencies
uv sync

# Run in development mode
uv run python -m micrograd.train
```

### Using pip

```bash
pip install -e .
```

### Optional Dependencies

```bash
# For development (pre-commit hooks, linting)
uv sync --group dev
```

## 🚀 Quick Start

### Basic Usage

```python
from micrograd.engine import Value

# Create values
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Build computation graph
e = a * b
e.label = 'e'
d = e + c
d.label = 'd'

# Compute gradients
d.backward()

print(f"a.grad = {a.grad}")  # -3.0
print(f"b.grad = {b.grad}")  # 2.0
```

### Training a Neural Network

```python
from micrograd.nn import MLP
from micrograd.engine import Value
from sklearn.datasets import make_moons
import numpy as np

# Create dataset
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert to -1/+1 labels

# Create model: 2 inputs -> 16 hidden (ReLU) -> 16 hidden (ReLU) -> 1 output (linear)
model = MLP(2, [16, 16, 1], activation=['relu', 'relu', 'linear'])

# Training loop
for epoch in range(100):
    # Forward pass
    inputs = [list(map(Value, xrow)) for xrow in X]
    scores = [model(x) for x in inputs]

    # SVM max-margin loss
    losses = [(Value(1) + Value(-yi)*scorei).relu() for yi, scorei in zip(y, scores)]
    data_loss = sum(losses) / len(losses)

    # L2 regularization
    reg_loss = 1e-4 * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # Backward pass
    model.zero_grad()
    total_loss.backward()

    # Update (with learning rate decay)
    lr = 1.0 - 0.9 * epoch / 100
    for p in model.parameters():
        p.data -= lr * p.grad

    # Compute accuracy
    accuracy = sum((yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)) / len(y)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={total_loss.data:.4f}, accuracy={accuracy*100:.1f}%")
```

### Using the Training Utility

```python
from micrograd.train import get_dataset, train_model
from micrograd.nn import MLP

# Get dataset
X, y = get_dataset(n_samples=100, noise=0.1, output_activation='linear', visualize=False)

# Create model
model = MLP(2, [16, 16, 1], activation=['relu', 'relu', 'linear'])

# Train
final_loss, final_accuracy = train_model(X, y, model, epochs=100, initial_lr=1.0)

print(f"Final accuracy: {final_accuracy*100:.1f}%")
```

## 📚 API Reference

### Value

The core autograd engine. Supports basic arithmetic operations with automatic gradient computation.

**Operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Activations: `.tanh()`, `.relu()`, `.exp()`
- Backpropagation: `.backward()`

```python
a = Value(2.0)
b = Value(3.0)
c = (a + b) * a  # Builds computation graph
c.backward()     # Computes gradients
```

### Neural Network Components

#### Neuron

```python
from micrograd.nn import Neuron

neuron = Neuron(nin=3, activation='relu')
output = neuron([1.0, 2.0, 3.0])
```

#### Layer

```python
from micrograd.nn import Layer

layer = Layer(nin=3, nout=5, activation='relu')
outputs = layer([1.0, 2.0, 3.0])  # Returns list of 5 Values
```

#### MLP (Multi-Layer Perceptron)

```python
from micrograd.nn import MLP

# Single activation for all layers
model = MLP(2, [16, 16, 1], activation='relu')

# Different activation per layer
model = MLP(2, [16, 16, 1], activation=['relu', 'relu', 'linear'])

# Get all parameters
params = model.parameters()

# Zero gradients
model.zero_grad()
```

**Architecture Notes:**
- For classification with -1/+1 labels: use `'linear'` or `'tanh'` for output layer
- For classification with 0/1 labels: use `'relu'` for output layer (not recommended)
- Hidden layers typically use `'relu'` activation

### Training Utilities

#### get_dataset

```python
from micrograd.train import get_dataset

X, y = get_dataset(
    n_samples=100,
    noise=0.1,
    output_activation='tanh',  # 'tanh'/'linear' -> -1/+1, 'relu' -> 0/1
    visualize=True
)
```

#### train_model

```python
from micrograd.train import train_model

final_loss, final_acc = train_model(
    X, y, model,
    epochs=100,
    initial_lr=1.0,
    batch_size=None  # None = full batch
)
```

## 🏗️ Project Structure

```
micrograd/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py          # Autograd engine (Value class)
│   ├── nn.py              # Neural network components (MLP, Layer, Neuron)
│   └── train.py           # Training utilities
├── notebooks/
│   └── micrograd_from_scratch.ipynb  # Interactive tutorial
├── tests/
│   ├── test_engine.py     # Tests for autograd
│   ├── test_nn.py         # Tests for neural networks
│   └── test_train.py      # Tests for training
├── .github/
│   └── workflows/
│       └── ci.yml         # CI/CD pipeline
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## 🧪 Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=micrograd --cov-report=html

# Run specific test file
uv run pytest tests/test_engine.py

# Run with verbose output
uv run pytest -v
```

## 🎨 Examples

### Gradient Checking

Compare analytical gradients with numerical gradients:

```python
from micrograd.engine import Value

def gradient_check(f, x, epsilon=1e-5):
    # Analytical gradient
    y = f(x)
    y.backward()
    analytical = x.grad

    # Numerical gradient
    x_plus = Value(x.data + epsilon)
    x_minus = Value(x.data - epsilon)
    numerical = (f(x_plus).data - f(x_minus).data) / (2 * epsilon)

    print(f"Analytical: {analytical}")
    print(f"Numerical: {numerical}")
    print(f"Difference: {abs(analytical - numerical)}")

# Test
x = Value(3.0)
gradient_check(lambda x: x**2, x)
```

### Visualizing Computation Graph

```python
from micrograd.engine import Value
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Example
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b
c.label = 'c'
c.backward()

dot = draw_dot(c)
dot.render('computation_graph', view=True)
```

### Custom Loss Functions

```python
from micrograd.engine import Value

def mse_loss(predictions, targets):
    """Mean squared error loss"""
    return sum((pred - tgt)**2 for pred, tgt in zip(predictions, targets)) / len(predictions)

def binary_crossentropy_loss(predictions, targets):
    """Binary cross-entropy (for 0/1 labels)"""
    epsilon = 1e-7
    loss = Value(0)
    for pred, tgt in zip(predictions, targets):
        if tgt == 1:
            loss += -((pred + epsilon).log())
        else:
            loss += -((Value(1) - pred + epsilon).log())
    return loss / len(predictions)
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Install with dev dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Run linting
uv run ruff check micrograd/
uv run ruff format micrograd/

# Type checking
uv run mypy micrograd/
```

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality:
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Python AST validation
- Ruff linting and formatting

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📖 Learning Resources

### Tutorials

Check out the Jupyter notebook for an interactive tutorial:
```bash
uv run jupyter notebook notebooks/micrograd_from_scratch.ipynb
```

### Key Concepts

**Autograd Engine:**
- Builds a dynamic computation graph
- Each operation creates a new node
- Backward pass traverses graph in reverse topological order
- Chain rule applied automatically

**Weight Initialization:**
- **He initialization** (ReLU): `w ~ N(0, sqrt(2/n_in))`
- **Xavier initialization** (tanh): `w ~ U(-sqrt(1/n_in), sqrt(1/n_in))`

**Training Tips:**
- Use linear or tanh output for -1/+1 classification
- ReLU hidden layers + linear output is a good default
- Start with high learning rate (1.0) and decay
- Add L2 regularization to prevent overfitting

## 🐛 Known Issues / Limitations

- **Scalar-only operations**: This is a scalar autograd engine (no vectorization)
- **Performance**: Not optimized for speed (educational purpose)
- **Memory**: Keeps entire computation graph in memory
- **No GPU support**: CPU-only implementation

For production use cases, consider PyTorch, TensorFlow, or JAX.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original micrograd
- [The Spelled-Out Intro to Neural Networks and Backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)

## 📬 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project. For production machine learning, use established frameworks like PyTorch or TensorFlow.
