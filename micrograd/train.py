from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from micrograd.nn import MLP
from micrograd.engine import Value
import numpy as np


def get_dataset(n_samples=100, noise=0.1, output_activation="tanh", visualize=True):
    """
    Generate moon dataset with labels compatible with the output activation.

    Args:
        n_samples: Number of samples
        noise: Noise level
        output_activation: 'relu' for [0, 1] labels, 'tanh'/'linear' for [-1, 1] labels
        visualize: Whether to plot the dataset

    Returns:
        X, y: Features and labels
    """

    X, y = make_moons(n_samples=n_samples, noise=noise)

    # Adjust labels based on output activation
    if output_activation == "relu":
        # ReLU can only output positive values, use 0 and 1
        y = y  # already 0 or 1 from make_moons
    else:
        # tanh/linear can output negative values, use -1 and 1
        y = y * 2 - 1

    # Visualize if requested
    if visualize:
        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
        plt.title(f"Labels: {set(y)} (output_activation={output_activation})")
        plt.show()

    return X, y


def loss_max_margin(pred: list[Value], labels: list[int]):
    """
    Max-margin loss compatible with both 0/1 and -1/+1 labels.

    For -1/+1 labels: maximize margin = prediction * label
    For 0/1 labels: maximize margin for class 1, minimize for class 0
    """
    loss = Value(0)

    # Check if labels are 0/1 or -1/+1
    label_set = set(labels)
    is_binary_01 = (0 in label_set or 1 in label_set) and (-1 not in label_set)

    for y_i, label_i in zip(pred, labels):
        if is_binary_01:
            # For 0/1 labels with ReLU output
            # Push predictions towards 0 or 1
            if label_i == 1:
                # Want prediction >= 1
                if y_i.data < 1:
                    loss += Value(1) - y_i
            else:  # label_i == 0
                # Want prediction <= 0 (but ReLU outputs >= 0, so push towards 0)
                if y_i.data > 0:
                    loss += y_i
        else:
            # For -1/+1 labels with tanh/linear output
            margin = y_i * label_i
            if margin.data < 1:
                loss += Value(1) - margin

    # Calculate accuracy
    if is_binary_01:
        accuracy = [
            (score_i.data > 0.5) == (label_i == 1)
            for score_i, label_i in zip(pred, labels)
        ]
    else:
        accuracy = [
            (score_i.data > 0) == (label_i > 0)
            for score_i, label_i in zip(pred, labels)
        ]

    return loss / len(pred), sum(accuracy) / len(accuracy)


def train_model(
    X: list[Value],
    y: list[int],
    nn: MLP,
    epochs: int,
    initial_lr: float = 1.0,
    batch_size: int = None,
):
    """
    Train the model using max-margin loss.

    Returns:
        tuple: (final_loss, final_accuracy)
    """
    final_loss = None
    final_acc = None

    for ep in range(epochs):
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]

        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # forward pass
        scores = [nn(x) for x in inputs]  # each is Value, not list

        # caluclate loss
        loss, acc = loss_max_margin(pred=scores, labels=yb)

        # backward propapagation
        nn.zero_grad()
        loss.backward()

        # learning rate schedule
        lr = initial_lr - 0.9 * initial_lr * ep / epochs

        # update weights
        for p in nn.parameters():
            # print(p.data, p.grad)
            p.data -= lr * p.grad

        print(f"step {ep} lr {lr:.3f} loss {loss.data:.4f}, accuracy {acc * 100:.1f}%")

        # Store final values
        final_loss = loss.data
        final_acc = acc

    return final_loss, final_acc


def main():
    X, y = get_dataset()
    nn = MLP(2, [16, 16, 1], activation=["relu", "relu", "linear"])
    final_loss, final_acc = train_model(
        X=X, y=y, nn=nn, epochs=100, initial_lr=1.0, batch_size=None
    )
    print(f"\nFinal: loss={final_loss:.4f}, accuracy={final_acc*100:.1f}%")


if __name__ == "__main__":
    main()
