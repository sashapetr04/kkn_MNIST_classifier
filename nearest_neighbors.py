import numpy as np
import distances as dist
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.X_samples = None
        self.y_samples = None
        self.NN_model = None
        if self.strategy in ['kd_tree', 'ball_tree', 'brute']:
            self.NN_model = NearestNeighbors(n_neighbors=self.k,
                                             algorithm=self.strategy,
                                             metric=self.metric)

    def fit(self, X, y):
        self.X_samples = X
        self.y_samples = y
        if self.strategy in ['kd_tree', 'ball_tree', 'brute']:
            self.NN_model.fit(X, y)

    def my_own_find_kneighbors(self, X, k, return_distance):
        if self.metric == 'cosine':
            distance_function = dist.cosine_distance
        else:
            distance_function = dist.euclidean_distance

        distances_matrix = distance_function(X, self.X_samples)
        indices = np.argpartition(distances_matrix, kth=np.arange(k),
                                  axis=1)[:, :k]
        distances = np.take_along_axis(distances_matrix, indices, axis=1)
        if return_distance is False:
            return indices
        return distances, indices

    def find_kneighbors(self, X, return_distance):
        if self.metric == 'cosine':
            distance_function = dist.cosine_distance
        else:
            distance_function = dist.euclidean_distance

        if self.strategy == 'my_own':
            find_function = self.my_own_find_kneighbors
        else:
            find_function = self.NN_model.kneighbors

        if self.test_block_size and X.shape[0] > self.test_block_size:
            num_blocks = X.shape[0] // self.test_block_size
            X_sections = np.array_split(X, num_blocks)
            if return_distance:
                distances, indices = find_function(X_sections[0], self.k, True)
                for section in X_sections[1:]:
                    distance, index = find_function(section, self.k, True)
                    indices = np.vstack((indices, index))
                    distances = np.vstack((distances, distance))
                return distances, indices

            else:
                indices = find_function(X_sections[0], self.k, False)
                for section in X_sections[1:]:
                    index = find_function(section, self.k, False)
                    indices = np.vstack((indices, index))
                return indices
        else:
            return find_function(X, self.k, return_distance)

    def predict(self, X):
        eps = 10 ** (-5)
        distances, indices = self.find_kneighbors(X, return_distance=True)
        if not self.weights:
            weights = np.ones_like(distances)
        else:
            weights = 1 / (distances + eps)

        targets = self.y_samples[indices]
        unique_targets = np.unique(targets)
        weighted_targets = np.array([])
        for index in range(len(unique_targets)):
            target = unique_targets[index]
            only_target = np.where(targets == target, 1, 0)
            weighted_target = np.sum(weights * only_target, axis=1)
            weighted_target = weighted_target.reshape((len(weighted_target),
                                                       1))
            if index == 0:
                weighted_targets = weighted_target
            else:
                weighted_targets = np.hstack((weighted_targets,
                                              weighted_target))
        max_indices = np.argmax(weighted_targets, axis=1)
        return unique_targets[max_indices]
