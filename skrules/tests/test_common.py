from sklearn.utils.estimator_checks import check_estimator
from skrules import SkopeRules


def test_classifier():
    return check_estimator(SkopeRules)
