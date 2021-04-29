from sklearn.utils.estimator_checks import check_estimator
from skrules import SkopeRules
from skrules.datasets import load_credit_data
import sklearn


def test_classifier():
    try:
        check_estimator(SkopeRules)
    except TypeError:
        # For sklearn >= 0.24.0 compatibility
        from sklearn.utils._testing import SkipTest
        from sklearn.utils.estimator_checks import check_sample_weights_invariance

        checks = check_estimator(SkopeRules(), generate_only=True)
        for estimator, check in checks:
            # Here we ignore this particular estimator check because
            # sample weights are treated differently in skope-rules
            if check.func != check_sample_weights_invariance:
                try:
                    check(estimator)
                except SkipTest as exception:
                    # SkipTest is thrown when pandas can't be imported, or by checks
                    # that are in the xfail_checks tag
                    warnings.warn(str(exception), SkipTestWarning)


def test_load_credit_data():
    assert load_credit_data().data.shape[0] == 30000
