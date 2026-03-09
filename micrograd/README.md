# micrograd Module

This directory contains the core implementation of the micrograd autograd engine and neural network library.

## Module Structure

### `engine.py` - Autograd Engine

The `Value` class is the heart of micrograd. It wraps scalars and tracks operations to build a dynamic computation graph.

**Key Components:**
- `Value.__init__()` - Creates a new value node in the computation graph
- `Value._backward()` - Local gradient function (set by operations)
- `Value.backward()` - Triggers backpropagation through the entire graph
- Arithmetic operations (`__add__`, `__mul__`, `__pow__`, etc.)
- Activation functions (`.tanh()`, `.relu()`, `.exp()`)

**Example:**
```python
from micrograd.engine import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x**2
z.backward()

print(x.grad)  # dz/dx = y + 2x = 3 + 4 = 7
print(y.grad)  # dz/dy = x = 2
```

**Implementation Details:**
- Uses reverse-mode automatic differentiation (backpropagation)
- Builds computation graph dynamically during forward pass
- Topological sort ensures correct gradient computation order
- Each operation stores references to its inputs in `_prev`

### `nn.py` - Neural Network Library

Provides building blocks for constructing neural networks.

#### Module (Base Class)

Abstract base class for all neural network components.

**Methods:**
- `parameters()` - Returns list of all trainable parameters
- `zero_grad()` - Resets all gradients to zero

#### Neuron

Single neuron with configurable activation function.

**Parameters:**
- `nin` - Number of input features
- `activation` - Activation function: `'tanh'`, `'relu'`, or `'linear'`

**Weight Initialization:**
- ReLU: He initialization `w ~ N(0, sqrt(2/nin))`, `b = 0.01`
- Tanh: Xavier initialization `w ~ U(-sqrt(1/nin), sqrt(1/nin))`, `b = 0`
- Linear: Same as tanh

**Example:**
```python
from micrograd.nn import Neuron

neuron = Neuron(nin=3, activation='relu')
x = [Value(1.0), Value(2.0), Value(3.0)]
output = neuron(x)
```

#### Layer

Collection of neurons forming a single layer.

**Parameters:**
- `nin` - Number of input features
- `nout` - Number of neurons in the layer
- `activation` - Activation function for all neurons

**Returns:**
- Single `Value` if `nout == 1`
- List of `Value` objects if `nout > 1`

**Example:**
```python
from micrograd.nn import Layer

layer = Layer(nin=3, nout=5, activation='relu')
x = [Value(1.0), Value(2.0), Value(3.0)]
outputs = layer(x)  # List of 5 Values
```

#### MLP (Multi-Layer Perceptron)

Complete neural network with multiple layers.

**Parameters:**
- `nin` - Number of input features
- `nouts` - List of neurons per layer (e.g., `[16, 16, 1]`)
- `activation` - Either:
  - String: same activation for all layers
  - List: different activation per layer (must match length of `nouts`)

**Example:**
```python
from micrograd.nn import MLP

# Same activation for all layers
model = MLP(2, [16, 16, 1], activation='relu')

# Different activation per layer (recommended)
model = MLP(2, [16, 16, 1], activation=['relu', 'relu', 'linear'])

# Forward pass
x = [Value(1.0), Value(2.0)]
output = model(x)

# Get all parameters
params = model.parameters()  # List of all weights and biases

# Zero gradients before backprop
model.zero_grad()
```

**Architecture Best Practices:**
```python
# Classification (-1/+1 labels)
model = MLP(nin, [...hidden...], activation=[...'relu'..., 'linear'])

# Regression
model = MLP(nin, [...hidden...], activation=[...'relu'..., 'linear'])

# Not recommended: ReLU on output for signed classification
model = MLP(nin, [...hidden...], activation='relu')  # ❌ Can't output negative
```

### `train.py` - Training Utilities

Functions for dataset creation, loss computation, and training loops.

#### get_dataset()

Creates the make_moons dataset with label format compatible with output activation.

**Parameters:**
- `n_samples` - Number of samples (default: 100)
- `noise` - Noise level (default: 0.1)
- `output_activation` - Determines label format:
  - `'relu'`: labels are 0 and 1
  - `'tanh'` or `'linear'`: labels are -1 and +1
- `visualize` - Whether to plot the dataset (default: True)

**Returns:**
- `X` - Input features, shape (n_samples, 2)
- `y` - Labels (numpy array)

#### loss_max_margin()

Computes SVM max-margin loss (hinge loss).

**Parameters:**
- `pred` - List of prediction Values
- `labels` - List of labels (int)

**Returns:**
- `loss` - Average loss (Value)
- `accuracy` - Classification accuracy (float)

**Implementation:**
- Automatically detects label format (0/1 vs -1/+1)
- For -1/+1: `loss = max(0, 1 - y*pred)`
- For 0/1: Different loss for each class

#### train_model()

Training loop with SVM max-margin loss and learning rate decay.

**Parameters:**
- `X` - Input features
- `y` - Labels
- `nn` - Neural network (MLP)
- `epochs` - Number of training epochs (default: 100)
- `initial_lr` - Starting learning rate (default: 1.0)
- `batch_size` - Mini-batch size (default: None for full batch)

**Returns:**
- `final_loss` - Loss at final epoch (float)
- `final_accuracy` - Accuracy at final epoch (float)

**Features:**
- Learning rate decay: `lr = initial_lr * (1 - 0.9 * epoch/epochs)`
- Prints progress every epoch
- Supports mini-batch training

**Example:**
```python
from micrograd.train import get_dataset, train_model
from micrograd.nn import MLP

X, y = get_dataset(n_samples=100, output_activation='linear', visualize=False)
model = MLP(2, [16, 16, 1], activation=['relu', 'relu', 'linear'])

final_loss, final_acc = train_model(
    X, y, model,
    epochs=100,
    initial_lr=1.0
)

print(f"Final accuracy: {final_acc*100:.1f}%")
```

## Design Patterns

### Computation Graph

The autograd engine builds a directed acyclic graph (DAG):
```
     x1        x2
      \        /
       \      /
        \    /
         add
          |
        relu
          |
        loss
```

Each node stores:
- `data` - Forward pass value
- `grad` - Gradient (initialized to 0)
- `_prev` - Parent nodes
- `_op` - Operation that created this node
- `_backward` - Function to compute local gradients

### Gradient Accumulation

Gradients are accumulated (`+=`) not assigned (`=`) because:
- A variable can be used multiple times in the computation
- Each usage contributes to the total gradient
- Example: `z = x + x` means `dz/dx = 1 + 1 = 2`

### Parameter Management

The `Module` base class provides a hierarchical parameter collection:
```python
MLP.parameters() calls Layer.parameters() for each layer
    Layer.parameters() calls Neuron.parameters() for each neuron
        Neuron.parameters() returns [b, w1, w2, ..., wn]
```

This allows easy parameter updates:
```python
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

## Performance Considerations

### Why is it slow?

1. **Scalar operations**: Each operation creates a new Python object
2. **No vectorization**: No use of NumPy's optimized operations
3. **Dynamic graph**: Graph is rebuilt for every forward pass
4. **Python overhead**: Pure Python implementation

### When to use micrograd?

- ✅ Learning how autograd works
- ✅ Understanding backpropagation
- ✅ Small educational experiments
- ❌ Production ML workloads
- ❌ Large datasets
- ❌ Deep networks (>5 layers)

For production, use PyTorch or TensorFlow.

## Extending micrograd

### Adding New Operations

To add a new operation (e.g., sigmoid):

```python
# In engine.py
def sigmoid(self):
    s = 1 / (1 + (-self).exp())
    out = Value(s.data, _children=(self,), _op='sigmoid')

    def _backward():
        self.grad += out.grad * s.data * (1 - s.data)

    out._backward = _backward
    return out

Value.sigmoid = sigmoid
```

### Adding New Loss Functions

```python
# In train.py
def cross_entropy_loss(predictions, targets):
    """Binary cross-entropy loss"""
    epsilon = 1e-7
    loss = Value(0)
    for pred, target in zip(predictions, targets):
        if target == 1:
            loss += -(pred + epsilon).log()
        else:
            loss += -(Value(1) - pred + epsilon).log()
    return loss / len(predictions)
```

### Adding New Layers

```python
# In nn.py
class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training:
            return x
        # Simplified dropout (not truly random in scalar form)
        return [xi * (1 / (1 - self.p)) if random.random() > self.p else Value(0)
                for xi in x]
```

## Debugging Tips

### Gradient Checking

Always verify your gradients numerically:
```python
def gradient_check(f, x, epsilon=1e-5):
    # Analytical
    y = f(x)
    y.backward()
    analytical = x.grad

    # Numerical
    x_plus = Value(x.data + epsilon)
    x_minus = Value(x.data - epsilon)
    numerical = (f(x_plus).data - f(x_minus).data) / (2 * epsilon)

    diff = abs(analytical - numerical)
    assert diff < 1e-5, f"Gradient check failed: {diff}"
```

### Visualizing Gradients

```python
# Print all gradients
for i, p in enumerate(model.parameters()):
    print(f"param {i}: data={p.data:.4f}, grad={p.grad:.4f}")

# Check for vanishing/exploding gradients
grads = [abs(p.grad) for p in model.parameters()]
print(f"Min grad: {min(grads):.6f}")
print(f"Max grad: {max(grads):.6f}")
print(f"Mean grad: {sum(grads)/len(grads):.6f}")
```

### Common Issues

**Issue**: Gradients are all zero
- **Cause**: Using ReLU on negative inputs (dead neurons)
- **Fix**: Check weight initialization, reduce learning rate

**Issue**: Loss is NaN
- **Cause**: Exploding gradients or division by zero
- **Fix**: Add gradient clipping, reduce learning rate

**Issue**: Model not learning
- **Cause**: Wrong output activation, bad initialization, or learning rate
- **Fix**: Use linear/tanh output for classification, check Karpathy's setup

## References

- [Original micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
