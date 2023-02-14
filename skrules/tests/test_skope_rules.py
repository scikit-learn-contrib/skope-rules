"""
Testing for SkopeRules algorithm (skrules.skope_rules).
"""

import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_iris, load_boston, make_blobs
from sklearn.metrics import accuracy_score

from sklearn.utils import check_random_state
from numpy.testing import assert_array_equal

from unittest import TestCase
import warnings

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

_dummy = TestCase("__init__")


def test_skope_rules():
    """Check various parameter settings."""
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {
            "feature_names": [None, ["a", "b"]],
            "precision_min": [0.0],
            "recall_min": [0.0],
            "n_estimators": [1],
            "max_samples": [0.5, 4],
            "max_samples_features": [0.5, 2],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
            "max_depth": [2],
            "max_features": ["auto", 1, 0.1],
            "min_samples_split": [2, 0.1],
            "n_jobs": [-1, 2],
        }
    )

    for params in grid:
        SkopeRules(random_state=rng, **params).fit(X_train, y_train).predict(X_test)

    # additional parameters:
    SkopeRules(n_estimators=50, max_samples=1.0, recall_min=0.0, precision_min=0.0).fit(
        X_train, y_train
    ).predict(X_test)


def test_skope_rules_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data
    y = iris.target
    y = y != 0

    # Test max_samples
    _dummy.assertRaises(ValueError, SkopeRules(max_samples=-1).fit, X, y)
    _dummy.assertRaises(ValueError, SkopeRules(max_samples=0.0).fit, X, y)
    _dummy.assertRaises(ValueError, SkopeRules(max_samples=2.0).fit, X, y)
    # explicitly setting max_samples > n_samples should result in a warning.
    assert_warns_message(
        UserWarning,
        "max_samples will be set to n_samples for estimation",
        SkopeRules(max_samples=1000).fit,
        X,
        y,
    )
    assert_no_warnings(SkopeRules(max_samples=np.int64(2)).fit, X, y)
    _dummy.assertRaises(ValueError, SkopeRules(max_samples="foobar").fit, X, y)
    _dummy.assertRaises(ValueError, SkopeRules(max_samples=1.5).fit, X, y)
    _dummy.assertRaises(ValueError, SkopeRules(max_depth_duplication=1.5).fit, X, y)
    _dummy.assertRaises(ValueError, SkopeRules().fit(X, y).predict, X[:, 1:])
    _dummy.assertRaises(ValueError, SkopeRules().fit(X, y).decision_function, X[:, 1:])
    _dummy.assertRaises(ValueError, SkopeRules().fit(X, y).rules_vote, X[:, 1:])
    _dummy.assertRaises(ValueError, SkopeRules().fit(X, y).score_top_rules, X[:, 1:])


def test_max_samples_attribute():
    X = iris.data
    y = iris.target
    y = y != 0

    clf = SkopeRules(max_samples=1.0).fit(X, y)
    assert clf.max_samples_ == X.shape[0]

    clf = SkopeRules(max_samples=500)
    assert_warns_message(
        UserWarning,
        "max_samples will be set to n_samples for estimation",
        clf.fit,
        X,
        y,
    )
    assert clf.max_samples_ == X.shape[0]

    clf = SkopeRules(max_samples=0.4).fit(X, y)
    assert clf.max_samples_ == 0.4 * X.shape[0]


def test_skope_rules_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1.0)
    clf.fit(X, y)
    decision_func = clf.decision_function(X_test)
    rules_vote = clf.rules_vote(X_test)
    score_top_rules = clf.score_top_rules(X_test)
    pred = clf.predict(X_test)
    pred_score_top_rules = clf.predict_top_rules(X_test, 1)
    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    assert np.min(rules_vote[-2:]) > np.max(rules_vote[:-2])
    assert np.min(score_top_rules[-2:]) > np.max(score_top_rules[:-2])
    assert_array_equal(pred, 6 * [0] + 2 * [1])
    assert_array_equal(pred_score_top_rules, 6 * [0] + 2 * [1])


def test_deduplication_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1.0, max_depth_duplication=3)
    clf.fit(X, y)
    decision_func = clf.decision_function(X_test)
    rules_vote = clf.rules_vote(X_test)
    score_top_rules = clf.score_top_rules(X_test)
    pred = clf.predict(X_test)
    pred_score_top_rules = clf.predict_top_rules(X_test, 1)


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
    assert y_pred.shape == (n_samples,)
    # training set performance
    assert accuracy_score(y, y_pred) > 0.83

    # decision_function agrees with predict
    decision = -clf.decision_function(X)
    assert decision.shape == (n_samples,)
    dec_pred = (decision.ravel() < 0).astype(np.int)
    assert_array_equal(dec_pred, y_pred)


def test_similarity_tree():
    # Test that rules are well splitted
    rules = [
        ("a <= 2 and b > 45 and c <= 3 and a > 4", (1, 1, 0)),
        ("a <= 2 and b > 45 and c <= 3 and a > 4", (1, 1, 0)),
        ("a > 2 and b > 45", (0.5, 0.3, 0)),
        ("a > 2 and b > 40", (0.5, 0.2, 0)),
        ("a <= 2 and b <= 45", (1, 1, 0)),
        ("a > 2 and c <= 3", (1, 1, 0)),
        ("b > 45", (1, 1, 0)),
    ]

    sk = SkopeRules(max_depth_duplication=2)
    rulesets = sk._find_similar_rulesets(rules)
    # Assert some couples of rules are in the same bag
    idx_bags_rules = []
    for idx_rule, r in enumerate(rules):
        idx_bags_for_rule = []
        for idx_bag, bag in enumerate(rulesets):
            if r in bag:
                idx_bags_for_rule.append(idx_bag)
        idx_bags_rules.append(idx_bags_for_rule)

    assert idx_bags_rules[0] == idx_bags_rules[1]
    assert idx_bags_rules[0] != idx_bags_rules[2]
    # Assert the best rules are kept
    final_rules = sk.deduplicate(rules)
    assert rules[0] in final_rules
    assert rules[2] in final_rules
    assert rules[3] not in final_rules


def test_f1_score():
    clf = SkopeRules()
    rule0 = ("a > 0", (0, 0, 0))
    rule1 = ("a > 0", (0.5, 0.5, 0))
    rule2 = ("a > 0", (0.5, 0, 0))

    assert clf.f1_score(rule0) == 0
    assert clf.f1_score(rule1) == 0.5
    assert clf.f1_score(rule2) == 0


def assert_warns_message(warning_class, message, func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    """Test that a certain warning occurs and with a certain message.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    message : str | callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`.

    Returns
    -------
    result : the return value of `func`

    """
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        if hasattr(np, "FutureWarning"):
            # Let's not catch the numpy internal DeprecationWarnings
            warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s" % func.__name__)

        found = [issubclass(warning.category, warning_class) for warning in w]
        if not any(found):
            raise AssertionError(
                "No warning raised for %s with class "
                "%s" % (func.__name__, warning_class)
            )

        message_found = False
        # Checks the message of all warnings belong to warning_class
        for index in [i for i, x in enumerate(found) if x]:
            # substring will match, the entire message with typo won't
            msg = w[index].message  # For Python 3 compatibility
            msg = str(msg.args[0] if hasattr(msg, "args") else msg)
            if callable(message):  # add support for certain tests
                check_in_message = message
            else:

                def check_in_message(msg):
                    return message in msg

            if check_in_message(msg):
                message_found = True
                break

        if not message_found:
            raise AssertionError(
                "Did not receive the message you expected "
                "('%s') for <%s>, got: '%s'" % (message, func.__name__, msg)
            )

    return result


def assert_no_warnings(func, *args, **kw):
    """
    Parameters
    ----------
    func
    *args
    **kw
    """
    # very important to avoid uncontrolled state propagation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = func(*args, **kw)
        if hasattr(np, "FutureWarning"):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w if e.category is not np.VisibleDeprecationWarning]

        if len(w) > 0:
            raise AssertionError(
                "Got warnings when calling %s: [%s]"
                % (func.__name__, ", ".join(str(warning) for warning in w))
            )
    return result
