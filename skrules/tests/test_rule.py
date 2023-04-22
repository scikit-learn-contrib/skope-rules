import pandas as pd
import numpy as np
from skrules import Rule, replace_feature_name

from skrules.utils import precision, recall, f1_score, mcc_score, get_confusion_matrix


def test_rule():
    assert(Rule('a <= 10 and a <= 12')
           == Rule('a <= 10'))

    assert(Rule('a <= 10 and a <= 12 and a > 3')
           == Rule('a > 3 and a <= 10'))

    assert(Rule('a <= 10 and a <= 10 and a > 3')
           == Rule('a > 3 and a <= 10'))

    assert(Rule('a <= 10 and a <= 12 and b > 3 and b > 6')
           == Rule('a <= 10 and b > 6'))

    assert(len({Rule('a <= 2 and a <= 3'), Rule('a <= 2')})
           == 1)

    assert(len({Rule('a > 2 and a > 3 and b <= 2 and b <= 3'),
                Rule('a > 3 and b <= 2')
                })
           == 1)

    assert(len({Rule('a <= 3 and b <= 2'), Rule('b <= 2 and a <= 3')})
           == 1)


def test_hash_rule():
    assert(len({Rule('a <= 2 and a <= 3'),
                Rule('a <= 2')
                })
           == 1)
    assert(len({Rule('a <= 4 and a <= 3'),
                Rule('a <= 2')
                })
           != 1)


def test_str_rule():
    rule = 'a <= 10.0 and b > 3.0'
    assert(rule == str(Rule(rule)))


def test_equals_rule():
    rule = "a == a"
    assert(rule == str(Rule(rule)))

    rule2 = "a == a and a == a"
    assert(rule == str(Rule(rule2)))

    rule3 = "a < 3.0 and a == a"
    assert(rule3 == str(Rule(rule3)))


def test_replace_feature_name():
    rule = "__C__0 <= 3 and __C__1 > 4"
    real_rule = "$b <= 3 and c(4) > 4"
    replace_dict = {
                    "__C__0": "$b",
                    "__C__1": "c(4)"
                    }
    assert(replace_feature_name(rule, replace_dict=replace_dict)
           == real_rule)


def test_precision():
    rule0 = ('a > 0', (0, 0, 0, 0))
    rule1 = ('a > 0', (0, 1, 0, 1))

    assert precision(rule0[1]) == 0
    assert precision(rule1[1]) == 0.5


def test_recall():
    rule0 = ('a > 0', (0, 0, 0, 0))
    rule1 = ('a > 0', (0, 0, 1, 1))

    assert recall(rule0[1]) == 0
    assert recall(rule1[1]) == 0.5


def test_f1_score():
    rule0 = ('a > 0', (0, 0, 0, 0))
    rule1 = ('a > 0', (0, 1, 1, 1))

    assert f1_score(rule0[1]) == 0
    assert f1_score(rule1[1]) == 0.5


def test_mcc_score():
    rule0 = ('a > 0', (0, 1, 0, 1))
    rule1 = ('a > 0', (0, 0, 1, 1))
    rule2 = ('a > 0', (1, 1, 0, 0))
    rule3 = ('a > 0', (1, 0, 1, 0))
    rule4 = ('a > 0', (0, 0, 0, 0))
    rule5 = ('a > 0', (1, 0, 0, 1))

    assert mcc_score(rule0[1]) == 0
    assert mcc_score(rule1[1]) == 0
    assert mcc_score(rule2[1]) == 0
    assert mcc_score(rule3[1]) == 0
    assert mcc_score(rule4[1]) == 0
    assert mcc_score(rule5[1]) == 1


def test_get_confusion_matrix():
    X = pd.DataFrame([[0, 1, 1], [1, 0, 1]], columns=['a', 'b', 'c'])
    y = np.array([1, 0])

    rule0 = 'a > 0'
    rule1 = 'b > 0'
    rule2 = 'c > 0'

    assert get_confusion_matrix(rule0, X, y) == (0, 1, 1, 0)
    assert get_confusion_matrix(rule1, X, y) == (1, 0, 0, 1)
    assert get_confusion_matrix(rule2, X, y) == (0, 1, 0, 1)
