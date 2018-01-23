.. -*- mode: rst -*-

|Travis|_  |Coveralls|_ |CircleCI|_ |Python27|_ |Python35|_

.. |Travis| image:: https://api.travis-ci.org/skope-rules/skope-rules.svg?branch=master
.. _Travis: https://travis-ci.org/skope-rules/skope-rules

.. |Coveralls| image:: https://coveralls.io/repos/github/skope-rules/skope-rules/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/skope-rules/skope-rules?branch=master

.. |CircleCI| image:: https://circleci.com/gh/skope-rules/skope-rules/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/skope-rules/skope-rules

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/skope-rules

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/skope-rules


skope-rules
===========

skope-rules is a Python machine learning module built on top of
scikit-learn and distributed under the 3-Clause BSD license.

It aims at learning logical, interpretable rules for "scoping" a target
class, i.e. detecting with high precision instances of this class.

See the `AUTHORS.rst <AUTHORS.rst>`_ file for a list of contributors.

Links with existing litterature
-------------------------------

The main advantage of decision rules is that they are offering interpretable models. The problem of generating such rules has been widely considered in machine learning, see e.g. RuleFit [1], Slipper [2], LRI [3], MLRules[4].

A decision rule is a logical expression of the form "IF conditions THEN reponse". In a binary classification setting, if an instance satisfies conditions of the rule, then it is assigned to one of the two classes. If this instance does not satisfy conditions, it remains unassigned.

1) In [2, 3, 4], rules induction is done by considering each single decision rule as a base classifier in an ensemble, which is built by greedily minimizing some loss function.

2) In [1], rules are extracted from an ensemble of trees; a weighted combination of these rules is then built by solving a L1-regularized optimization problem over the weights as described in [5].

In this package, we use the second approach. Rules are extracted from tree ensemble, which allow us to take advantage of existing fast algorithms (such as bagged decision trees, or gradient boosting) to produce such tree ensemble. Too similar or duplicated rules are then removed, based on a similarity threshold of their supports..
The main goal of this package is to provide rules verifying precision and recall conditions. It still implement a score (`decision_function`) method, but which does not solve the L1-regularized optimization problem as in [1]. Instead, weights are simply proportional to the OOB associated precision of the rule.

This package also offers convenient methods to compute predictions with the k most precise rules (cf score_top_rules() and predict_top_rules() functions).


[1] Friedman and Popescu, Predictive learning via rule ensembles,Technical Report, 2005.

[2] Cohen and Singer, A simple, fast, and effective rule learner, National Conference on Artificial Intelligence, 1999.

[3] Weiss and Indurkhya, Lightweight rule induction, ICML, 2000.

[4] Dembczyński, Kotłowski and Słowiński, Maximum Likelihood Rule Ensembles, ICML, 2008.

[5] Friedman and Popescu, Gradient directed regularization, Technical Report, 2004.

Dependencies
------------

skope-rules requires:

- Python (>= 2.7 or >= 3.3)
- NumPy (>= 1.10.4)
- SciPy (>= 0.17.0)
- Pandas (>= 0.18.1)
- Scikit-Learn (>= 0.17.1)

For running the examples Matplotlib >= 1.1.1 is required.

Installation
------------

You can get the latest sources with the command::

    pip install skope-rules
    
Documentation
--------------

You can access the full project documentation `here <http://skope-rules.readthedocs.io/en/latest/>`_

You can also check the notebooks/ folder which contains some examples of utilisation.