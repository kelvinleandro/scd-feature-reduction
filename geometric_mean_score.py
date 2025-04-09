import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, make_scorer


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-08)


def geometric_mean_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return np.sqrt(recall * specificity)


gmean_scorer = make_scorer(geometric_mean_score)
