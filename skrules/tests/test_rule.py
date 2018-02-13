from sklearn.utils.testing import assert_equal, assert_not_equal

from skrules import Rule


def test_rule():
    assert_equal(Rule('a <= 10 and a <= 12'),
                 Rule('a <= 10'))
    assert_equal(Rule('a <= 10 and a <= 12 and a > 3'),
                 Rule('a > 3 and a <= 10'))

    assert_equal(Rule('a <= 10 and a <= 10 and a > 3'),
                 Rule('a > 3 and a <= 10'))

    assert_equal(Rule('a <= 10 and a <= 12 and b > 3 and b > 6'),
                 Rule('a <= 10 and b > 6'))

    assert_equal(len({Rule('a <= 2 and a <= 3'),
                      Rule('a <= 2')
                      }), 1)

    assert_equal(len({Rule('a > 2 and a > 3 and b <= 2 and b <= 3'),
                      Rule('a > 3 and b <= 2')
                      }), 1)

    assert_equal(len({Rule('a <= 3 and b <= 2'),
                      Rule('b <= 2 and a <= 3')
                      }), 1)


def test_hash_rule():
    assert_equal(len({
                        Rule('a <= 2 and a <= 3'),
                        Rule('a <= 2')
                      }), 1)
    assert_not_equal(len({
                        Rule('a <= 4 and a <= 3'),
                        Rule('a <= 2')
                      }), 1)


def test_str_rule():
    rule = 'a <= 10.0 and b > 3.0'
    assert_equal(rule, str(Rule(rule)))


def test_equals_rule():
    rule = "a == a"
    assert_equal(rule, str(Rule(rule)))

    rule2 = "a == a and a == a"
    assert_equal(rule, str(Rule(rule2)))

    rule3 = "a < 3.0 and a == a"
    assert_equal(rule3, str(Rule(rule3)))
