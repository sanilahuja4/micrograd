from sklearn.datasets import make_moons


def make_dataset():
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1
