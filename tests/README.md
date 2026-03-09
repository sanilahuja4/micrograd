# Tests

Comprehensive test suite for the micrograd library.

## Overview

This directory contains unit tests and integration tests for all micrograd components:
- Autograd engine (`test_engine.py`)
- Neural network components (`test_nn.py`)
- Training utilities (`test_train.py`)

## Running Tests

### Run All Tests

```bash
# Using uv
uv run pytest

# Using pytest directly
pytest
```

### Run Specific Test File

```bash
uv run pytest tests/test_engine.py
uv run pytest tests/test_nn.py
uv run pytest tests/test_train.py
```

### Run Specific Test

```bash
# Run a specific test function
uv run pytest tests/test_engine.py::test_addition

# Run tests matching a pattern
uv run pytest -k "gradient"
```

### Verbose Output

```bash
# Show all test names and results
uv run pytest -v

# Show print statements
uv run pytest -s

# Show detailed failure info
uv run pytest -vv
```

### Coverage Report

```bash
# Generate coverage report
uv run pytest --cov=micrograd

# Generate HTML coverage report
uv run pytest --cov=micrograd --cov-report=html

# View HTML report
open htmlcov/index.html
```

## Test Files

### `test_engine.py` - Autograd Engine Tests

Tests for the `Value` class and autograd functionality.

**Test Categories:**

1. **Basic Operations**
   - Addition, subtraction, multiplication, division
   - Power operations
   - Negation

2. **Activation Functions**
   - tanh
   - ReLU
   - exp

3. **Gradient Computation**
   - Single operations
   - Chained operations
   - Multiple usage of same variable

4. **Gradient Checking**
   - Numerical vs analytical gradients
   - Various operations and combinations

**Example Tests:**

```python
def test_addition():
    """Test addition operation and gradients"""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b

    c.backward()

    assert c.data == 5.0
    assert a.grad == 1.0  # dc/da = 1
    assert b.grad == 1.0  # dc/db = 1


def test_multiplication():
    """Test multiplication operation and gradients"""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b

    c.backward()

    assert c.data == 6.0
    assert a.grad == 3.0  # dc/da = b
    assert b.grad == 2.0  # dc/db = a
```

### `test_nn.py` - Neural Network Tests

Tests for neural network components (Neuron, Layer, MLP).

**Test Categories:**

1. **Component Creation**
   - Neuron initialization
   - Layer initialization
   - MLP initialization

2. **Forward Pass**
   - Neuron output
   - Layer output
   - MLP output

3. **Parameter Management**
   - Counting parameters
   - Parameter collection
   - Zero gradients

4. **Activation Functions**
   - Different activations per layer
   - ReLU initialization
   - Tanh initialization

**Example Tests:**

```python
def test_neuron_creation():
    """Test neuron creates correct number of parameters"""
    neuron = Neuron(nin=3, activation='relu')
    params = neuron.parameters()

    assert len(params) == 4  # 3 weights + 1 bias


def test_mlp_forward():
    """Test MLP forward pass"""
    model = MLP(2, [4, 1], activation='tanh')
    x = [Value(1.0), Value(2.0)]
    output = model(x)

    assert isinstance(output, Value)
    assert -1 <= output.data <= 1  # tanh output range
```

### `test_train.py` - Training Utilities Tests

Tests for training functions and utilities.

**Test Categories:**

1. **Dataset Creation**
   - get_dataset with different parameters
   - Label format (0/1 vs -1/+1)

2. **Loss Functions**
   - Max-margin loss
   - Loss with different label formats

3. **Training Loop**
   - Training convergence
   - Learning rate decay
   - Parameter updates

**Example Tests:**

```python
def test_get_dataset():
    """Test dataset creation"""
    X, y = get_dataset(n_samples=50, noise=0.1, output_activation='tanh', visualize=False)

    assert X.shape == (50, 2)
    assert len(y) == 50
    assert set(y) == {-1, 1}


def test_training_reduces_loss():
    """Test that training reduces loss"""
    X, y = get_dataset(n_samples=50, output_activation='linear', visualize=False)
    model = MLP(2, [8, 1], activation=['relu', 'linear'])

    # Train for a few epochs
    final_loss, final_acc = train_model(X, y, model, epochs=10, initial_lr=1.0)

    assert final_acc > 0.5  # Should be better than random
```

## Writing New Tests

### Test Structure

Follow this structure for consistency:

```python
def test_descriptive_name():
    """
    Brief description of what this test does.

    Additional details about the test case,
    edge cases, or expected behavior.
    """
    # Arrange: Set up test inputs
    x = Value(2.0)
    y = Value(3.0)

    # Act: Perform the operation
    z = x * y
    z.backward()

    # Assert: Check results
    assert z.data == 6.0
    assert x.grad == 3.0
    assert y.grad == 2.0
```

### Gradient Checking Tests

Always include numerical gradient checks for new operations:

```python
def test_operation_gradients():
    """Test gradients against numerical approximation"""
    x = Value(2.0)
    epsilon = 1e-5

    # Analytical gradient
    y = x ** 2
    y.backward()
    analytical = x.grad

    # Numerical gradient
    x_plus = Value(x.data + epsilon)
    x_minus = Value(x.data - epsilon)
    numerical = (x_plus.data**2 - x_minus.data**2) / (2 * epsilon)

    assert abs(analytical - numerical) < 1e-5
```

### Parameterized Tests

Use pytest's `@pytest.mark.parametrize` for testing multiple cases:

```python
import pytest

@pytest.mark.parametrize("activation,expected_range", [
    ('tanh', (-1, 1)),
    ('relu', (0, float('inf'))),
    ('linear', (-float('inf'), float('inf'))),
])
def test_activation_range(activation, expected_range):
    """Test activation function output ranges"""
    neuron = Neuron(nin=2, activation=activation)
    x = [Value(-10.0), Value(10.0)]
    output = neuron(x)

    min_val, max_val = expected_range
    if activation == 'tanh':
        assert min_val < output.data < max_val
```

### Fixture Usage

Create reusable test fixtures:

```python
import pytest

@pytest.fixture
def simple_model():
    """Fixture providing a simple trained model"""
    return MLP(2, [4, 1], activation=['relu', 'linear'])


@pytest.fixture
def sample_data():
    """Fixture providing sample dataset"""
    X, y = get_dataset(n_samples=20, visualize=False)
    return X, y


def test_with_fixtures(simple_model, sample_data):
    """Test using fixtures"""
    X, y = sample_data
    inputs = [list(map(Value, xrow)) for xrow in X]
    outputs = [simple_model(x) for x in inputs]

    assert len(outputs) == 20
```

## Test Coverage Goals

Aim for the following coverage:

- **Overall**: > 90%
- **engine.py**: > 95% (critical component)
- **nn.py**: > 90%
- **train.py**: > 85%

### Check Current Coverage

```bash
uv run pytest --cov=micrograd --cov-report=term-missing
```

This shows which lines are not covered by tests.

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- See `.github/workflows/ci.yml` for CI configuration

### Local CI Simulation

Run the same checks as CI:

```bash
# Linting
uv run ruff check micrograd/

# Formatting check
uv run ruff format --check micrograd/

# Type checking
uv run mypy micrograd/

# Tests
uv run pytest
```

## Common Testing Patterns

### Testing for Exceptions

```python
def test_invalid_activation():
    """Test that invalid activation raises error"""
    with pytest.raises(ValueError):
        Neuron(nin=3, activation='invalid')
```

### Testing Floating Point Values

Use `pytest.approx` for floating point comparisons:

```python
def test_floating_point():
    """Test with floating point tolerance"""
    x = Value(1.0)
    y = x * 0.1 * 10  # May not be exactly 1.0

    assert y.data == pytest.approx(1.0, rel=1e-5)
```

### Testing Randomness

Set random seed for reproducibility:

```python
def test_random_initialization():
    """Test random initialization is deterministic with seed"""
    import numpy as np

    np.random.seed(42)
    model1 = MLP(2, [4, 1])

    np.random.seed(42)
    model2 = MLP(2, [4, 1])

    # Should have same parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert p1.data == p2.data
```

## Debugging Failed Tests

### Run Failed Tests Only

```bash
# Run only tests that failed last time
uv run pytest --lf

# Run failed tests first, then others
uv run pytest --ff
```

### Interactive Debugging

```bash
# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger on first failure
uv run pytest -x --pdb
```

### Print Debugging

```python
def test_with_debug():
    """Test with debug output"""
    model = MLP(2, [4, 1])

    # Print parameter info
    for i, p in enumerate(model.parameters()):
        print(f"Param {i}: data={p.data}, grad={p.grad}")

    # Run test
    assert len(model.parameters()) == 17
```

Run with `-s` to see print output:
```bash
uv run pytest -s tests/test_nn.py::test_with_debug
```

## Performance Testing

### Benchmark Tests

Use `pytest-benchmark` for performance testing:

```python
def test_forward_pass_performance(benchmark):
    """Benchmark forward pass performance"""
    model = MLP(2, [16, 16, 1])
    x = [Value(1.0), Value(2.0)]

    result = benchmark(model, x)
    assert isinstance(result, Value)
```

### Memory Usage

Profile memory usage:

```python
import tracemalloc

def test_memory_usage():
    """Test memory usage of computation graph"""
    tracemalloc.start()

    model = MLP(2, [16, 16, 1])
    X = [[Value(1.0), Value(2.0)] for _ in range(100)]
    outputs = [model(x) for x in X]

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    assert peak < 100 * 1024 * 1024  # Less than 100 MB
```

## Best Practices

1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should explain what they test
3. **Keep Tests Fast**: Tests should run in milliseconds, not seconds
4. **Isolate Tests**: Tests should not depend on each other
5. **Test Edge Cases**: Include boundary conditions and corner cases
6. **Document Tests**: Add docstrings explaining complex test logic
7. **Use Fixtures**: Avoid code duplication with pytest fixtures
8. **Check Coverage**: Aim for high coverage but focus on meaningful tests

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Best Practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html)
- [Testing Machine Learning Code](https://www.jeremyjordan.me/testing-ml/)
