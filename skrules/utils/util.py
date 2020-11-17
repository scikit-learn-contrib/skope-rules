# -*- coding: utf8 -*-

import numpy as np
from sklearn.utils import indices_to_mask


def get_confusion_matrix(rule, test_x, test_y):
    """Return the confusion matrix of a rule given the data (X,y)
    in the form (tn, fp, fn, tp)

    Parameters
    ----------
        rule: Rule
            The rule which is being evaluated on the data (X,y)
        test_x: array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features
        test_y: array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention
            0 for normal data, 1 for anomalies

    Returns
    -------
        tuple
            Confusion matrix
    """
    # not using confusion_matrix function from sklearn.metrics
    # for computation time perfs reasons
    n_samples, _ = test_x.shape

    detected_indices = list(test_x.query(rule).index)
    if len(detected_indices) < 1:
        return 0, 0, 0, 0
    y_detected = test_y[detected_indices]

    undetected_indices = ~indices_to_mask(detected_indices, n_samples)
    y_undetected = test_y[undetected_indices]

    tp = len(y_detected[y_detected == 1])
    fp = len(y_detected[y_detected == 0])
    tn = len(y_undetected[y_undetected == 0])
    fn = len(y_undetected[y_undetected == 1])

    return tn, fp, fn, tp


def precision(confusion_matrix):
    """Compute the precision of a given rule according
    to its confusion matrix

    Parameters
    ----------
    confusion_matrix : tuple
        Confusion matrix (tn, fp, fn, tp)

    Returns
    -------
        float
            Precision
    """
    tn, fp, fn, tp = confusion_matrix

    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(confusion_matrix):
    """Compute the recall of a given rule according
    to its confusion matrix

    Parameters
    ----------
    confusion_matrix : tuple
        Confusion matrix (tn, fp, fn, tp)

    Returns
    -------
        float
            Recall
    """
    tn, fp, fn, tp = confusion_matrix

    return tp / (tp + fn) if (tp + fn) > 0 else 0


def mcc_score(confusion_matrix):
    """Compute the Matthews Correlation Coefficient of
    a given rule according to its confusion matrix

    Parameters
    ----------
        confusion_matrix : tuple
            Confusion matrix (tn, fp, fn, tp)


    Returns
    -------
        float
            MCC score : value is between -1 and +1
                +1: perfect prediction
                0: no better than random prediction
                -1: total disagreement between prediction and observation
    """
    # retrieving confusion matrix
    tn, fp, fn, tp = confusion_matrix

    numerator = (tp*tn - fp*fn)
    if all(x > 0 for x in (tp + fp, tp + fn, tn + fp, tn + fn)):
        # using log and exp to prevent overflow
        denominator = (np.log(tp + fp)
                       + np.log(tp + fn)
                       + np.log(tn + fp)
                       + np.log(tn + fn))
        denominator = np.exp(1 / 2*denominator)

        return numerator / denominator

    else:
        return 0


def f1_score(confusion_matrix):
    """Compute the F1-score of a given rule
    (harmonic mean of the precision and recall)

    Parameters
    ----------
        confusion_matrix : tuple
            Confusion matrix (tn, fp, fn, tp)

    Returns
    -------
        Float
            F1-score : value is between 0 and 1
    """
    pre = precision(confusion_matrix)
    rec = recall(confusion_matrix)

    return 2*pre*rec / (pre + rec) if (pre + rec) > 0 else 0
