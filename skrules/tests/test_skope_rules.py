"""
Testing for SkopeRules algorithm (skrules.skope_rules).
"""

import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_iris, load_boston, make_blobs
from sklearn.metrics import accuracy_score

from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import ignore_warnings


from skrules import SkopeRules

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


def test_skope_rules():
    """Check various parameter settings."""
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
               [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid({
        "feature_names": [None, ['a', 'b']],
        "precision_min": [0.],
        "recall_min": [0.],
        "n_estimators": [1],
        "max_samples": [0.5, 4],
        "max_samples_features": [0.5, 2],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "max_depth": [2],
        "max_features": ["auto", 1, 0.1],
        "min_samples_split": [2, 0.1],
        "n_jobs": [-1, 2]})

    with ignore_warnings():
        for params in grid:
            SkopeRules(random_state=rng,
                       **params).fit(X_train, y_train).predict(X_test)

    # additional parameters:
    SkopeRules(n_estimators=50,
               max_samples=1.,
               recall_min=0.,
               precision_min=0.).fit(X_train, y_train).predict(X_test)


def test_skope_rules_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data
    y = iris.target
    y = (y != 0)

    # Test similarity_thres:
    assert_raises(ValueError,
                  SkopeRules(similarity_thres=2).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(similarity_thres=0).fit, X, y)

    # Test max_samples
    assert_raises(ValueError,
                  SkopeRules(max_samples=-1).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(max_samples=0.0).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(max_samples=2.0).fit, X, y)
    # explicitly setting max_samples > n_samples should result in a warning.
    assert_warns_message(UserWarning,
                         "max_samples will be set to n_samples for estimation",
                         SkopeRules(max_samples=1000).fit, X, y)
    assert_no_warnings(SkopeRules(max_samples=np.int64(2)).fit, X, y)
    assert_raises(ValueError, SkopeRules(max_samples='foobar').fit, X, y)
    assert_raises(ValueError, SkopeRules(max_samples=1.5).fit, X, y)
    assert_raises(ValueError, SkopeRules().fit(X, y).predict, X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).decision_function,
                  X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).rules_vote, X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).separate_rules_score,
                  X[:, 1:])


def test_max_samples_attribute():
    X = iris.data
    y = iris.target
    y = (y != 0)

    clf = SkopeRules(max_samples=1.).fit(X, y)
    assert_equal(clf.max_samples_, X.shape[0])

    clf = SkopeRules(max_samples=500)
    assert_warns_message(UserWarning,
                         "max_samples will be set to n_samples for estimation",
                         clf.fit, X, y)
    assert_equal(clf.max_samples_, X.shape[0])

    clf = SkopeRules(max_samples=0.4).fit(X, y)
    assert_equal(clf.max_samples_, 0.4*X.shape[0])


def test_skope_rules_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1.)
    clf.fit(X, y)
    decision_func = clf.decision_function(X_test)
    rules_vote = clf.rules_vote(X_test)
    separate_rules_score = clf.separate_rules_score(X_test)
    pred = clf.predict(X_test)
    # assert detect outliers:
    assert_greater(np.min(decision_func[-2:]), np.max(decision_func[:-2]))
    assert_greater(np.min(rules_vote[-2:]), np.max(rules_vote[:-2]))
    assert_greater(np.min(separate_rules_score[-2:]),
                   np.max(separate_rules_score[:-2]))
    assert_array_equal(pred, 6 * [0] + 2 * [1])


def test_performances():
    X, y = make_blobs(n_samples=1000, random_state=0, centers=2)

    # make labels imbalanced by remove all but 100 instances from class 1
    indexes = np.ones(X.shape[0]).astype(bool)
    ind = np.array([False] * 100 + list(((y == 1)[100:])))
    indexes[ind] = 0
    X = X[indexes]
    y = y[indexes]
    n_samples, n_features = X.shape

    clf = SkopeRules()
    # fit
    clf.fit(X, y)
    # with lists
    clf.fit(X.tolist(), y.tolist())
    y_pred = clf.predict(X)
    assert_equal(y_pred.shape, (n_samples,))
    # training set performance
    assert_greater(accuracy_score(y, y_pred), 0.83)

    # decision_function agrees with predict
    decision = -clf.decision_function(X)
    assert_equal(decision.shape, (n_samples,))
    dec_pred = (decision.ravel() < 0).astype(np.int)
    assert_array_equal(dec_pred, y_pred)
