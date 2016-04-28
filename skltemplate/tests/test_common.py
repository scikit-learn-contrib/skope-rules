from sklearn.utils.estimator_checks import check_estimator
from skltemplate import TemplateEstimator, TemplateClassifier


def test_estimator():
    return check_estimator(TemplateEstimator)


def test_classifier():
    return check_estimator(TemplateClassifier)
