.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |Python27|_ |Python35|_ |PyPi|_ |DOI|_

.. |Travis| image:: https://api.travis-ci.org/skope-rules/skope-rules.svg?branch=master
.. _Travis: https://travis-ci.org/skope-rules/skope-rules

.. |Codecov| image:: https://coveralls.io/repos/github/skope-rules/skope-rules/badge.svg?branch=master
.. _Codecov: https://coveralls.io/github/skope-rules/skope-rules?branch=master

.. |CircleCI| image:: https://circleci.com/gh/skope-rules/skope-rules/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/skope-rules/skope-rules

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/skope-rules

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/skope-rules

.. |PyPi| image:: https://badge.fury.io/py/skope-rules.svg
.. _PyPi: https://badge.fury.io/py/skope-rules

skope-rules
============

skope-rules is a Python machine learning module built on top of
scikit-learn and distributed under the 3-Clause BSD license.

It aims at learning logical, interpretable rules for "scoping" a target
class, i.e. detecting with high precision instances of this class.

See the `AUTHORS.rst <AUTHORS.rst>`_ file for a complete list of contributors.


Installation
------------

Dependencies
~~~~~~~~~~~~

skope-rules requires:

- Python (>= 2.7 or >= 3.3)
- NumPy (>= 1.10.4)
- SciPy (>= 0.17.0)
- Pandas (>= 0.18.1)
- Scikit-Learn (>= 0.17.1)

For running the examples Matplotlib >= 1.1.1 is required.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install skope-rules is using ``pip`` ::

    pip install skope-rules

Source code
~~~~~~~~~~~

You can get the latest sources with the command::

    git clone https://github.com/skope-rules/skope-rules.git

Then you just need to execute in the skope-rules directory::

    python setup.py install
