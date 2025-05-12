import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    indices = np.arange(n)
    split = np.array_split(indices, n_folds)
    cv = list()
    for i in range(n_folds):
        cv.append((np.hstack(split[:i] + split[i+1:]), split[i]))
    return cv


def predict_with_distance(knn_model, y, distances, indices):
    eps = 10 ** (-5)
    if not knn_model.weights:
        weights = np.ones_like(distances)
    else:
        weights = 1 / (distances + eps)

    targets = y[indices]
    unique_targets = np.unique(y[indices])
    weighted_targets = np.array([])
    for index in range(len(unique_targets)):
        target = unique_targets[index]
        only_target = np.where(targets == target, 1, 0)
        weighted_target = np.sum(weights * only_target, axis=1)
        weighted_target = weighted_target.reshape((len(weighted_target), 1))
        if index == 0:
            weighted_targets = weighted_target
        else:
            weighted_targets = np.hstack((weighted_targets, weighted_target))
    max_indices = np.argmax(weighted_targets, axis=1)
    return unique_targets[max_indices]


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    k_dict = {}
    if cv is None:
        cv = kfold(X.shape[0], 3)

    k_max = k_list[-1]
    for train, valid in cv:
        knn_model = KNNClassifier(k_max, **kwargs)
        knn_model.fit(X[train], y[train])
        distances, indices = knn_model.find_kneighbors(X[valid],
                                                       return_distance=True)
        for k in k_list:
            if k not in k_dict.keys():
                k_dict[k] = np.array([])
            y_predict = predict_with_distance(knn_model, y[train],
                                              distances[:, :k], indices[:, :k])
            if score == 'accuracy':
                ans_num = y[valid].shape[0]
                true_ans_num = np.sum(y[valid] == y_predict)
                k_dict[k] = np.append(k_dict[k], true_ans_num / ans_num)
    return k_dict
