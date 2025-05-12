import numpy as np


def euclidean_distance(X, Y):
    X_norms_square = np.linalg.norm(X, axis=1) ** 2
    Y_norms_square = np.linalg.norm(Y, axis=1) ** 2
    euclidean_distance_matrix = np.sqrt(X_norms_square.reshape(X.shape[0], 1) +
                                        Y_norms_square.reshape(1, Y.shape[0]) - 2 * X @ Y.T)
    return euclidean_distance_matrix


def cosine_distance(X, Y):
    X_norms = np.linalg.norm(X, axis=1)
    Y_norms = np.linalg.norm(Y, axis=1)
    # transpose for broadcasting
    X_transposed_and_normalized = X.T / np.where(X_norms != 0, X_norms, 1)
    Y_transposed_and_normalized = Y.T / np.where(Y_norms != 0, Y_norms, 1)
    cosine_distance_matrix = X_transposed_and_normalized.T @ Y_transposed_and_normalized
    return 1 - cosine_distance_matrix
