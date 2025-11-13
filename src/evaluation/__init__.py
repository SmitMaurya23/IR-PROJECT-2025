import numpy as np


def f2_score(precision, recall):
    return 5 * precision * recall / (4 * precision + recall + 1e-5)


def task3_metrics(groundtruth, predictions):
    correct = 0
    n_retrived = 1e-5
    n_relevant = 1e-5

    coverages = []

    for target, preds in zip(groundtruth, predictions):
        coverages.append(len(preds))

        n_retrived += len(preds)
        n_relevant += len(target)

        for p in preds:
            if p in target:
                correct += 1

    precision = correct / n_retrived
    recall = correct / n_relevant

    return {
        'average_candidates': np.mean(coverages),
        'precision': precision,
        'recall': recall,
        'f2': f2_score(precision, recall)
    }
    