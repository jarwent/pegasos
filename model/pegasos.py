import numpy as np
import numba


@numba.njit()
def _fit(xs, ys, lmbda, kernel):
    alpha = np.zeros(len(xs))
    t = 0
    for i in range(len(xs)):
        t += 1
        z = np.dot(np.multiply(alpha, ys), kernel[i, :])
        z *= ys[i] / (lmbda * t)
        if z < 1:
            alpha[i] += 1
    return alpha / (lmbda * t)


def fit(train_xs, train_ys, lmbda, gram):
    """
    Fit kernelized Pegasos algorithm to training data using one-vs-the-rest approach

    :param train_xs: train set
    :param train_ys: train labels
    :param lmbda:    regularization parameter
    :param gram:     kernel matrix

    :return: binary weights, binary data
    """
    binary_labels = {}
    weights = []
    for k in range(10):
        xs = train_xs.copy()
        ys = train_ys.copy()
        class_pos = k
        # set instances with class k to +1, all other classes to -1
        class_pos_indices = np.where((ys == class_pos))[0]
        class_neg_indices = np.where((ys != class_pos))[0]
        ys[class_pos_indices] = 1
        ys[class_neg_indices] = -1
        # keep track of two-class data to be used later for prediction
        binary_labels[k] = ys
        w = _fit(xs, ys, lmbda, gram)
        weights.append(w)
    return weights, binary_labels


def predict(xs, ys, weights, data, kernel, num_classes=10):
    """
    Predict class for test set

    :param xs:      test set
    :param ys:      test labels
    :param weights: binary weights
    :param data:    binary data
    :param kernel:  kernel matrix

    :return: test error
    """
    num_correct = 0
    for i, x in enumerate(xs):
        predictions = np.zeros(num_classes)
        for k in range(num_classes):
            if kernel is not None:
                predictions[k] = np.dot(np.multiply(weights[k], data[k]), kernel[i, :])
            else:
                predictions[k] = np.dot(weights[k], x)
        predicted = np.argmax(predictions)
        correct = ys[i]
        if predicted == correct:
            num_correct += 1
    return 1 - num_correct / len(ys)
