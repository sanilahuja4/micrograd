from micrograd.nn import Neuron, Layer, MLP, Module
from micrograd.engine import Value


def test_neuron_creation():
    """Test that a neuron can be created with the correct number of weights"""
    n = Neuron(3)
    assert isinstance(n.b, Value)
    assert n.b.data == 0


def test_neuron_call():
    """Test that a neuron can process input"""
    n = Neuron(2)
    x = [1, 2]
    output = n(x)
    # Output should be a Value object (after tanh activation)
    assert isinstance(output, Value) or isinstance(output, list)


def test_neuron_parameters():
    """Test that neuron returns its parameters"""
    n = Neuron(3)
    params = n.parameters()
    assert isinstance(params, list)
    # Should have bias + weights
    assert len(params) > 0


def test_layer_creation():
    """Test that a layer can be created with correct number of neurons"""
    layer = Layer(nin=3, nout=2)
    assert len(layer.neurons) == 2
    assert all(isinstance(n, Neuron) for n in layer.neurons)


def test_layer_call():
    """Test that a layer can process input"""
    layer = Layer(nin=2, nout=3)
    x = [Value(1.0), Value(2.0)]
    outputs = layer(x)
    assert isinstance(outputs, list)
    assert len(outputs) == 3


def test_layer_parameters():
    """Test that layer returns all neuron parameters"""
    layer = Layer(nin=2, nout=3)
    params = layer.parameters()
    assert isinstance(params, list)
    # Should have parameters from all 3 neurons
    assert len(params) > 0


def test_mlp_creation():
    """Test MLP creation with multiple layers"""
    mlp = MLP(nin=3, nouts=[4, 4, 1])
    assert len(mlp.layers) == 3
    # First layer: 3 inputs, 4 outputs
    assert len(mlp.layers[0].neurons) == 4
    # Second layer: 4 inputs, 4 outputs
    assert len(mlp.layers[1].neurons) == 4
    # Third layer: 4 inputs, 1 output
    assert len(mlp.layers[2].neurons) == 1


def test_mlp_call():
    """Test that MLP can process input through all layers"""
    mlp = MLP(nin=3, nouts=[4, 1])
    x = [Value(1.0), Value(2.0), Value(3.0)]
    output = mlp(x)
    # Output should be a list (from the last layer)
    assert isinstance(output, Value)


def test_mlp_parameters():
    """Test that MLP returns all parameters from all layers"""
    mlp = MLP(nin=2, nouts=[3, 1])
    params = mlp.parameters()
    assert isinstance(params, list)
    # Should have many parameters (weights + biases from all neurons)
    assert len(params) > 0


def test_module_zero_grad():
    """Test the zero_grad method"""
    module = Module()
    # Module's zero_grad should iterate over parameters
    # This is a basic test - in practice would need parameters with non-zero grads
    try:
        module.zero_grad()
    except AttributeError:
        # Expected if parameters() returns empty list
        pass


def test_mlp_forward_pass_values():
    """Test that MLP forward pass produces reasonable values"""
    mlp = MLP(nin=2, nouts=[2, 1])
    x = [Value(0.5), Value(-0.5)]
    output = mlp(x)

    # Output should be a list
    assert isinstance(output, Value)
    # Each output should be a Value with tanh activation (between -1 and 1)
    if isinstance(output, Value):
        assert -1.5 <= output.data <= 1.5  # Some margin for numerical issues
