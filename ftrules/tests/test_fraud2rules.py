"""
Testing for FraudToRules algorithm (ftrules.fraud2rules).
"""


import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_boston, load_iris
from sklearn.utils import check_random_state

from ftrules import FraudToRules

rng = check_random_state(0)

# load the iris dataset
# and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_fraudetorules():
    """Check various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid({"n_estimators": [1],
                          "max_samples": [0.5, 1.0, 3],
                          "bootstrap": [True, False]})

    with ignore_warnings():
        for params in grid:
            FraudToRules(random_state=rng,
                         **params).fit(X_train).predict(X_test)


def test_fraudetorules_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data

    # Test max_samples
    assert_raises(ValueError,
                  FraudToRules(max_samples=-1).fit, X)
    assert_raises(ValueError,
                  FraudToRules(max_samples=0.0).fit, X)
    assert_raises(ValueError,
                  FraudToRules(max_samples=2.0).fit, X)
    # The dataset has less than 256 samples, explicitly setting
    # max_samples > n_samples should result in a warning. If not set
    # explicitly there should be no warning
    assert_warns_message(UserWarning,
                         "max_samples will be set to n_samples for estimation",
                         FraudToRules(max_samples=1000).fit, X)
    assert_no_warnings(FraudToRules(max_samples='auto').fit, X)
    assert_no_warnings(FraudToRules(max_samples=np.int64(2)).fit, X)
    assert_raises(ValueError, FraudToRules(max_samples='foobar').fit, X)
    assert_raises(ValueError, FraudToRules(max_samples=1.5).fit, X)


def test_max_samples_attribute():
    X = iris.data
    clf = FraudToRules().fit(X)
    assert_equal(clf.max_samples_, X.shape[0])

    clf = FraudToRules(max_samples=500)
    assert_warns_message(UserWarning,
                         "max_samples will be set to n_samples for estimation",
                         clf.fit, X)
    assert_equal(clf.max_samples_, X.shape[0])

    clf = FraudToRules(max_samples=0.4).fit(X)
    assert_equal(clf.max_samples_, 0.4*X.shape[0])


def test_fraudetorules_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, 7]]

    # Test LOF
    for contamination in [0.25, "auto"]:
        clf = FraudToRules(random_state=rng, contamination=contamination)
        clf.fit(X)
        decision_func = - clf.decision_function(X)
        pred = clf.predict(X)
        # assert detect outliers:
        assert_greater(np.min(decision_func[-2:]), np.max(decision_func[:-2]))
        assert_array_equal(pred, 6 * [1] + 2 * [-1])


# XXX TODO: add test for parallel computation (n_jobs)
# add performance test
