from sklearn.utils.estimator_checks import check_estimator
from skltemplate import TemplateEstimator

def test_common():
    return check_estimator(TemplateEstimator)
