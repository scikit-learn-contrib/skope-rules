"""
Testing for SkopeRules algorithm (skrules.skope_rules).
"""
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns
from numpy.testing import assert_no_warnings

from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_iris, load_boston, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state


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

    grid = ParameterGrid([{
        "feature_names": [None, ['a', 'b']],
        "filtering_criteria": [{'precision': 0., 'recall': 0.}],
        "duplication_criterion": ['f1', 'mcc'],
        "custom_func": [None],
        "n_estimators": [1],
        "max_samples": [0.5, 4],
        "max_samples_features": [0.5, 2],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "max_depth": [2],
        "max_features": ["auto", 1, 0.1],
        "min_samples_split": [2, 0.1],
        "n_jobs": [-1, 2]},
        # grid with custom_func parameter
        {
        "feature_names": [None, ['a', 'b']],
        "filtering_criteria": [{'precision': 0., 'recall': 0., 'custom_func': 0}],
        "duplication_criterion": ['f1', 'mcc', 'custom_func'],
        "custom_func": [lambda mtx: mtx[0]],
        "n_estimators": [1],
        "max_samples": [0.5, 4],
        "max_samples_features": [0.5, 2],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "max_depth": [2],
        "max_features": ["auto", 1, 0.1],
        "min_samples_split": [2, 0.1],
        "n_jobs": [-1, 2]}]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for params in grid:
            SkopeRules(random_state=rng,
                       **params).fit(X_train, y_train).predict(X_test)

    # additional parameters:
    SkopeRules(n_estimators=50,
               max_samples=1.,
               filtering_criteria={'precision': 0., 'recall': 0.}
               ).fit(X_train, y_train).predict(X_test)


def test_skope_rules_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data
    y = iris.target
    y = (y != 0)

    # Test max_samples
    assert_raises(ValueError,
                  SkopeRules(max_samples=-1).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(max_samples=0.0).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(max_samples=2.0).fit, X, y)
    # explicitly setting max_samples > n_samples should result in a warning.
    assert_warns(UserWarning,
                 SkopeRules(max_samples=1000).fit, X, y)
    assert_no_warnings(SkopeRules(max_samples=np.int64(2)).fit, X, y)
    assert_raises(ValueError, SkopeRules(max_samples='foobar').fit, X, y)
    assert_raises(ValueError, SkopeRules(max_samples=1.5).fit, X, y)
    with assert_raises(TypeError):
        SkopeRules(max_depth_duplication=1.5).fit(X, y)
    assert_raises(ValueError, SkopeRules().fit(X, y).predict, X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).decision_function,
                  X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).rules_vote, X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).score_top_rules,
                  X[:, 1:])
    # check filtering_criteria errors
    with assert_raises(TypeError):
        SkopeRules(filtering_criteria=[]).fit(X, y)
    with assert_raises(TypeError):
        SkopeRules(filtering_criteria={}).fit(X, y)
    with assert_raises(ValueError):
        SkopeRules(filtering_criteria={'foobar': 0.}).fit(X, y)
    with assert_raises(TypeError):
        SkopeRules(filtering_criteria={'foo': 'bar'}).fit(X, y)
    # check duplication_criterion errors
    with assert_raises(ValueError):
        SkopeRules(duplication_criterion=0).fit(X, y)
    with assert_raises(ValueError):
        SkopeRules(duplication_criterion='foobar').fit(X, y)
    # check custom_func errors
    with assert_raises(TypeError):
        SkopeRules(duplication_criterion='custom_func', custom_func=None).fit(X, y)
    with assert_raises(TypeError):
        SkopeRules(filtering_criteria={'custom_func': 0.}, custom_func=None).fit(X, y)
    with assert_raises(TypeError):
        SkopeRules(duplication_criterion='custom_func',
                   custom_func='foobar').fit(X, y)
    with assert_raises(TypeError):
        SkopeRules(duplication_criterion='custom_func',
                   custom_func=lambda x: x).fit(X, y)


def test_max_samples_attribute():
    X = iris.data
    y = iris.target
    y = (y != 0)

    clf = SkopeRules(max_samples=1.).fit(X, y)
    assert clf.max_samples_ == X.shape[0]

    clf = SkopeRules(max_samples=500)
    assert_warns(UserWarning,
                 clf.fit, X, y)
    assert clf.max_samples_ == X.shape[0]

    clf = SkopeRules(max_samples=0.4).fit(X, y)
    assert clf.max_samples_ == 0.4*X.shape[0]


def test_skope_rules_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3],
         [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1.)
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
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3],
         [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1., max_depth_duplication=3)
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
    # tn, fp, fn, tp
    rules = [("a <= 2 and b > 45 and c <= 3 and a > 4", (10, 0, 0, 10)),
             ("a <= 2 and b > 45 and c <= 3 and a > 4", (10, 0, 0, 10)),
             ("a > 2 and b > 45", (7, 3, 3, 7)),
             ("a > 2 and b > 40", (6, 4, 4, 6)),
             ("a <= 2 and b <= 45", (10, 0, 0, 10)),
             ("a > 2 and c <= 3", (10, 0, 0, 10)),
             ("b > 45", (10, 0, 0, 10)),
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
    final_rules = sk._deduplicate(rules)
    assert rules[0] in final_rules
    assert rules[2] in final_rules
    assert rules[3] not in final_rules
