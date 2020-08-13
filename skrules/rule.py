# -*- coding: utf8 -*-
import re
import numpy as np
from sklearn.utils import indices_to_mask
from decimal import Decimal as D

from .utils import check_features_to_round


def replace_feature_name(rule, replace_dict):
    def replace(match):
        return replace_dict[match.group(0)]

    rule = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replace_dict),
                  replace, rule)
    return rule


class Rule:
    """ An object modelling a logical rule and add factorization methods.
    It is used to simplify rules and deduplicate them.

    Parameters
    ----------

    rule : str
        The logical rule that is interpretable by a pandas query.

    args : object, optional
        Arguments associated to the rule, it is not used for factorization
        but it takes part of the output when the rule is converted to an array.
    """

    def __init__(self, rule, args=None):
        self.rule = rule
        self.args = args
        self.terms = [t.split(' ') for t in self.rule.split(' and ')]
        self.agg_dict = {}
        self.factorize()
        self.rule = str(self)

    def __eq__(self, other):
        return self.agg_dict == other.agg_dict

    def __hash__(self):
        # FIXME : Easier method ?
        return hash(tuple(sorted(((i, j) for i, j in self.agg_dict.items()))))

    def factorize(self):
        for feature, symbol, value in self.terms:
            if (feature, symbol) not in self.agg_dict:
                if symbol != '==':
                    self.agg_dict[(feature, symbol)] = str(float(value))
                else:
                    self.agg_dict[(feature, symbol)] = value
            else:
                if symbol[0] == '<':
                    self.agg_dict[(feature, symbol)] = str(min(
                                float(self.agg_dict[(feature, symbol)]),
                                float(value)))
                elif symbol[0] == '>':
                    self.agg_dict[(feature, symbol)] = str(max(
                                float(self.agg_dict[(feature, symbol)]),
                                float(value)))
                else:  # Handle the c0 == c0 case
                    self.agg_dict[(feature, symbol)] = value

    def __iter__(self):
        yield str(self)
        yield self.args

    def __repr__(self):
        return ' and '.join([' '.join(
                [feature, symbol, str(self.agg_dict[(feature, symbol)])])
                for feature, symbol in sorted(self.agg_dict.keys())
                ])


def get_confusionMatrix(rule, X, y):
    """Return the confusion matrix of a rule given the data (X,y)
    in the form (tn, fp, fn, tp)

    Arguments:
        rule: Rule
            The rule which is being evaluated on the data (X,y)
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features
        y: array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention
            0 for normal data, 1 for anomalies

    Returns:
        tuple: confusion matrix
    """
    # not using confusion_matrix function from sklearn.metrics
    # for computation time perfs reasons
    n_samples = y.shape[0]

    detected_indices = list(X.query(rule).index)
    if len(detected_indices) < 1:
        return (0, 0, 0, 0)
    y_detected = y[detected_indices]

    undetected_indices = ~indices_to_mask(detected_indices, n_samples)
    y_undetected = y[undetected_indices]

    tp = len(y_detected[y_detected == 1])
    fp = len(y_detected[y_detected == 0])
    tn = len(y_undetected[y_undetected == 0])
    fn = len(y_undetected[y_undetected == 1])

    return (tn, fp, fn, tp)


def precision(rule):
    """Compute the precision of a given rule according
    to its confusion matrix
    """
    # retrieving confusion matrix
    tn, fp, fn, tp = rule[1]

    return tp/(tp+fp) if (tp+fp) > 0 else 0


def recall(rule):
    """Compute the recall of a given rule according to
    its confusion matrix
    """
    # retrieving confusion matrix
    tn, fp, fn, tp = rule[1]

    return tp/(tp+fn) if (tp+fn) > 0 else 0


def mcc_score(rule):
    """Compute the Matthews Correlation Coefficient of
    a given rule according to its confusion matrix

    Arguments:
        rule: Rule

    Returns:
        mcc : float
            The value is between -1 and +1
                +1: perfect prediction
                0: no better than random prediction
                -1: total disagreement between prediction and observation
    """
    # retrieving confusion matrix
    tn, fp, fn, tp = rule[1]

    numerator = (tp*tn - fp*fn)
    if all(x > 0 for x in (tp+fp, tp+fn, tn+fp, tn+fn)):
        # using log and exp to prevent overflow
        denominator = (np.log(tp+fp)
                       + np.log(tp+fn)
                       + np.log(tn+fp)
                       + np.log(tn+fn))
        denominator = np.exp(1/2*denominator)

        return numerator/denominator

    else:
        return 0


def f1_score(rule):
    """Compute the F1-score of a given rule
    (harmonic mean of the precision and recall)

    Arguments:
        rule: Rule

    Returns:
        f1-score: float
            The value is between 0 and 1
    """
    p = precision(rule)
    r = recall(rule)

    return 2*p*r / (p+r) if (p+r) > 0 else 0


def round_rule(rule, features_to_round):
    """Approximate the values of the features of a rule
    according to a dict containing the features to round.

    Arguments:
        rule: Rule
            Tuple containing the rule string and confusion matrix

        features_to_round: dict
            A dict containing the names of the quantitative features
            whose values are to be rounded.
            Should be in the form {var_name: power_of_ten_exponent}
            with types {str: int/float}.
            The power_of_ten_exponent should be an int
            (if float then rounded) either positive or negative.
            It is the exponent of the power of ten at which the value
            of the feature is rounded.

            example:
                - power_of_ten_exponent = 1 => 1357.914 becomes 1.36E+3
                - power_of_ten_exponent = 0 => 1357.914 becomes 1358
                - power_of_ten_exponent = -2 => 1357.914 becomes 1357.91

    Returns:
        rounded rule: str
            A string representation of the rule with rounded features values.
    """
    check_features_to_round(features_to_round)

    def truncate(n, order=None):
        if order is not None:
            power_of_ten = 10**(round(order))
            return D(str(n)).quantize(D('{:.0e}'.format(power_of_ten)))
        else:
            return n

    terms = [t.split(' ') for t in rule[0].split(' and ')]

    for j in range(len(terms)):
        if terms[j][0] in features_to_round:
            terms[j][2] = truncate(terms[j][2],
                                   order=features_to_round[terms[j][0]]
                                   )

    rounded_rule = ' and '.join([' '.join([elt[0], elt[1], str(elt[2])])
                                 for elt in terms
                                 ]
                                )

    return rounded_rule
