import pandas as pd
import numpy as np


def load_data(path='../data/zipcombo.dat'):
    """
    Load data set
    """
    mnist = pd.read_csv(path, header=None, sep='\s', engine='python')
    for col in mnist.columns[1:]:
        mnist[col] = mnist[col].astype(np.float_)
    mnist[mnist.columns[0]] = mnist[mnist.columns[0]].astype(np.int_)
    mnist_xs = mnist.iloc[:, 1:].to_numpy()
    mnist_ys = mnist.iloc[:, 0].to_numpy()
    return mnist_xs, mnist_ys


def train_test_split(xs, ys):
    """
    Shuffle data set. Split into 80% train and 20% test.
    """
    split = int(len(xs) * .8)
    permutation = np.random.permutation(len(ys))
    train_xs, test_xs = xs[permutation][0:split], xs[permutation][split:]
    train_ys, test_ys = ys[permutation][0:split], ys[permutation][split:]
    return train_xs, train_ys, test_xs, test_ys
