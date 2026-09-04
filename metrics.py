"""Evaluation helpers: Hungarian-matched accuracy, NMI and macro-F1.

Cluster indices are matched to class indices by the Hungarian algorithm on the
confusion matrix (the protocol of the paper: matching fitted on the evaluated set).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, normalized_mutual_info_score


def hungarian_map(y_true, y_pred, n_classes):
    """Cluster -> class mapping maximising the matched count."""
    M = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(M, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    r, c = linear_sum_assignment(-M)
    mapping = np.arange(n_classes)
    mapping[c] = r
    return mapping, M


def accuracy(y_true, y_pred, n_classes):
    mapping, M = hungarian_map(y_true, y_pred, n_classes)
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / M.sum())


def macro_f1(y_true, y_pred, n_classes):
    mapping, _ = hungarian_map(y_true, y_pred, n_classes)
    return float(f1_score(y_true, mapping[np.asarray(y_pred, int)], average="macro"))


def nmi(y_true, y_pred):
    return float(normalized_mutual_info_score(y_true, y_pred))


def evaluate(y_true, y_pred, n_classes):
    """{'acc', 'f1', 'nmi'} on the given pixels (e.g. the spatial test blocks)."""
    return dict(acc=accuracy(y_true, y_pred, n_classes),
                f1=macro_f1(y_true, y_pred, n_classes),
                nmi=nmi(y_true, y_pred))
