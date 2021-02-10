import numpy as np

from model.kernels import gram_matrix, kernel_matrix, gaussian_kernel
from collections import defaultdict
from model.pegasos import fit, predict
from config.config import *
from experiment.utils import load_data, train_test_split
from experiment.kfold import KFold


if __name__ == "__main__":
    np.random.seed(seed)
    mnist_xs, mnist_ys = load_data()
    optimal_dimensions, test_errors = [], []

    for run in range(runs):
        print(f"run: {run}")
        train_xs, train_ys, test_xs, test_ys = train_test_split(mnist_xs, mnist_ys)
        kf = KFold(train_xs, 5)
        results = defaultdict(list)

        for train_index, test_index in kf:
            xs, ys = train_xs[train_index], train_ys[train_index]
            xt, yt = train_xs[test_index], train_ys[test_index]

            # gaussian width
            for c in [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1]:
                gram = gram_matrix(xs, c, gaussian_kernel)
                kernel = kernel_matrix(xt, xs, c, gaussian_kernel)

                # SVM regularization parameter
                for lmbda in [1e-6, 1e-5, 1e-4, 1e-3]:
                    weights, binary_labels = fit(xs, ys, lmbda, gram)
                    test_error = predict(xt, yt, weights, binary_labels, kernel)
                    results[(c, lmbda)].append(test_error)

        # optimal width and regularization parameter
        c, lmbda = min(results.items(), key=lambda x: np.average(x[1]))[0]
        gram = gram_matrix(train_xs, c, gaussian_kernel)
        kernel = kernel_matrix(test_xs, train_xs, c, gaussian_kernel)
        weights, binary_labels = fit(train_xs, train_ys, lmbda, gram)
        test_error = predict(test_xs, test_ys, weights, binary_labels, kernel)
        test_errors.append(test_error)
        optimal_dimensions.append((c, lmbda))
        print(f"c={c} lmbda={lmbda} error={test_error}")

    optimal_c = list(map(lambda x: x[0], optimal_dimensions))
    optimal_lmbda = list(map(lambda x: x[1], optimal_dimensions))

    print(f"average={np.average(optimal_c)} std={np.std(optimal_c)}")
    print(f"average={np.average(optimal_lmbda)} std={np.std(optimal_lmbda)}")
    print(f"average={np.average(test_errors)} std={np.std(test_errors)}")
